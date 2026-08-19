#!/usr/bin/env python3
"""Recipe for probing a Transformer ASR system for phonemic information.
The system trains a linear probe on all the hidden layers in the encoder
to classify phonemes, and records the speed at which they all learn.
"""

import os
import sys
import pathlib

import torch
import polars
import textgrid
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from torch.nn.functional import cross_entropy

import data_sb
import speechbrain as sb
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.logger import get_logger
from speechbrain.utils.data_pipeline import takes, provides
from speechbrain.decoders.ctc import ctc_greedy_decode


logger = get_logger(__name__)


class PhonemeProbe(torch.nn.Module):
    """Create a probe for each hidden representation."""
    def __init__(self, phoneme_count, cnn_size, enc_size, enc_layers):
        super().__init__()
        self.cnn_probe = torch.nn.Linear(cnn_size, phoneme_count)
        self.t_probes = torch.nn.ModuleList()
        for _ in range(enc_layers):
            self.t_probes.append(torch.nn.Linear(enc_size, phoneme_count))

    def forward(self, cnn_output, hidden_layers):
        """Apply probe to both cnn and hidden layers"""
        batch, time, _, _ = cnn_output.shape
        cnn_out_flat = cnn_output.view(batch, time, -1)
        probes = [self.cnn_probe(cnn_out_flat)]

        # Hidden layers includes the input, but we handle that separately above
        # This is the reason for the "+1" on the hidden layer index.
        probes += [p(hidden_layers[i + 1]) for i, p in enumerate(self.t_probes)]
        return probes


# Define training procedure
class ProbeBrain(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Apply the probes to the encoded representations."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.signal

        # compute features
        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.hparams.normalize(feats, wav_lens, epoch=current_epoch)

        # forward modules
        src = self.modules.CNN(feats)
        enc_out, hidden_layers = self.modules.Transformer.encode(src, wav_lens)

        return self.modules.phoneme_probe(src, hidden_layers)

    def compute_objectives(self, predictions, batch, stage):
        """Computes the phoneme loss against all probes."""

        _, wav_lens = batch.signal

        loss = 0
        for i, p in enumerate(predictions):
            #p = self.hparams.log_softmax(p)
            #loss += self.hparams.ctc_cost(p, y, wav_lens, y_lens)

            # Loss and metrics both expect shape [batch, classes, ...]
            # After transpose, shape is [batch, phonemes, time]
            p = p.transpose(1, 2)
            loss += cross_entropy(p, batch.phone_targets[0], ignore_index=0)

            if stage == sb.Stage.VALID:
            #    predicted = ctc_greedy_decode(p, wav_lens, blank_id=self.hparams.blank_index)
            #    self.phoneme_metrics[i].append(batch.id, predicted, y)

                self.phoneme_metrics[i](p, batch.phone_targets[0])

        return loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.phoneme_metrics = [
                self.hparams.PhonemeMetric().to(self.device)
                for i in range(self.hparams.num_encoder_layers + 1)
            ]

    def on_fit_start(self):
        super().on_fit_start()
        self.metrics_log = []

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        
        def compute_metric(metric):
            score = round(metric.compute().item(), 5)
            metric.reset()
            return score

        if stage == sb.Stage.VALID:
            row = {
                f"layer_{i}": compute_metric(m)
                for i, m in enumerate(self.phoneme_metrics)
            }
            row["epoch"] = epoch
            print("Validation loss", stage_loss)
            print(row)

            self.metrics_log.append(row)
           

if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Load pretrained weights if available
    if "pretrainer" in hparams:
        hparams["pretrainer"].collect_files()
        hparams["pretrainer"].load_collected()

        # Freeze params if requested
        if "freeze_model" in hparams and hparams["freeze_model"]:
            for p in hparams["model"].parameters():
                p.requires_grad = False


    data_sb.make_manifests(hparams)
    data_sb.make_encoders(hparams)
    datasets = data_sb.make_datasets(hparams)

    # Trainer initialization
    probe_brain = ProbeBrain(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Training
    probe_brain.fit(
        probe_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    df = polars.DataFrame(probe_brain.metrics_log)
    df.write_csv(probe_brain.hparams.metrics_log)
