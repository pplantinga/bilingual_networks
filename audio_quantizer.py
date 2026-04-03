#!/usr/bin/env python3
"""Recipe for generating discrete audio tokens using k-means on mfcc features.

Author
 * Peter Plantinga
"""

import sys
import pathlib

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

import data_sb
import speechbrain as sb
from speechbrain.utils.logger import get_logger
from speechbrain.utils.data_pipeline import takes, provides
from speechbrain.utils.data_utils import undo_padding


logger = get_logger(__name__)


def flatten_no_padding(data, lengths):
    B, L = data.shape[:2]
    abs_lengths = (lengths * L).round().long()
    indices = torch.arange(L, device=data.device).unsqueeze(0)
    mask = indices < abs_lengths.unsqueeze(1)

    return data[mask]

    #extra_dims = data.shape[2:]
    #flat_data = data.reshape(B * L, *extra_dims)
    #flat_mask = mask.reshape(B * L)
    #return flat_data[flat_mask]



class QuantizerBrain(sb.core.Brain):
    """Look ma, no gradients!"""

    @torch.no_grad()
    def compute_forward(self, batch, stage):
        """Simply compute features used in clustering"""
        audio, audio_len = batch.signal
        features = self.hparams.compute_features(audio)
        features = self.hparams.normalize(features, audio_len)
        features = flatten_no_padding(features, audio_len)
        return features

    @torch.no_grad()
    def compute_objectives(self, features, batch, stage):
        """Performs minibatch updates during training, returns inertia during validation"""

        # For tracking across epochs, use inertia as a measure of goodness-of-fit
        if stage != sb.Stage.TRAIN:
            return self.hparams.quantizer.inertia(features)

        # Store centroids to compute drift
        pre_update_centroids = self.hparams.quantizer.cluster_centers

        # Update centroids with however many batches we can extract
        for batch in self.iterate_features(features):
            self.hparams.quantizer.partial_fit(batch)

        # Fake loss is computed as drift, SpeechBrain expects tensor with gradient
        post_update_centroids = self.hparams.quantizer.cluster_centers
        loss = (post_update_centroids - pre_update_centroids).norm()
        loss.requires_grad_()

        return loss

    @torch.no_grad()
    def iterate_features(self, features):
        """Iterator producing batches of the size the quantizer expects, saving
        any remainders and prepending to the next batch."""
        features = torch.cat((self.leftovers, features), dim=0)
        batches = features.split(self.hparams.kmeans_batch_size, dim=0)
        self.leftovers = batches[-1]
        for batch in batches[:-1]:
            yield batch

    def on_stage_start(self, stage, epoch):
        """Initialize leftover features"""
        self.leftovers = torch.tensor([])

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        elif stage == sb.Stage.VALID:
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_checkpoint(
                meta={"epoch": epoch, "loss": stage_loss},
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )


# Define custom data procedure
def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    @takes("path")
    @provides("signal")
    def audio_pipeline(path):
        """Returns full audio sequence."""
        base = hparams["data_folder"] / pathlib.Path(path)
        wav_path = base.with_suffix(".mp3")
        return data_sb.load_audio_and_resample(wav_path, hparams["sample_rate"])

    # Define datasets before the text pipeline so that
    # we can use it to load all the text and write to file
    # for training the tokenizer then used in the text pipeline
    datasets = {}
    lang = hparams["train_languages"][0]
    pipelines = [audio_pipeline]
    for dataset in ["train", "valid", "test"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            hparams[f"{dataset}_{lang}_manifest"],
            dynamic_items=pipelines,
            output_keys=["signal"],
        )

    return datasets


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

    data_sb.make_manifests(hparams)
    datasets = dataio_prepare(hparams)

    # Trainer initialization
    quantizer_brain = QuantizerBrain(
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Training
    quantizer_brain.fit(
        hparams["epoch_counter"],
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )
