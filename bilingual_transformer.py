#!/usr/bin/env python3
"""Keep separate decoders per language, swap based on input lang.

Author: Peter Plantinga 2026
"""

import os
import sys
import pathlib
import textgrid
from copy import deepcopy

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

import data_sb
import speechbrain as sb
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.dataio.sampler import DynamicBatchSampler
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.data_pipeline import takes, provides


logger = sb.utils.logger.get_logger(__name__)


# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.signal
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        tokens_bos, _ = batch.tokens_bos

        # compute features
        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.hparams.normalize(feats, wav_lens, epoch=current_epoch)

        # Add feature augmentation if specified.
        if (
            stage == sb.Stage.TRAIN
            and hasattr(self.hparams, "fea_augment")
            and self.optimizer_step > self.hparams.augment_warmup
        ):
            feats, fea_lens = self.hparams.fea_augment(feats, wav_lens)
            tokens_bos = self.hparams.fea_augment.replicate_labels(tokens_bos)

        # Encoder is shared between all languages
        src = self.modules.CNN(feats)
        enc_out, pred = self.modules.Transformer(
            src, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index,
        )

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(enc_out)
        p_ctc = self.hparams.log_softmax(logits)

        # output layer for seq2seq log-probabilities
        pred = self.modules.seq_lin(pred)
        p_seq = self.hparams.log_softmax(pred)

        # Compute outputs
        hyps = None
        current_epoch = self.hparams.epoch_counter.current
        is_valid_search = (
            stage == sb.Stage.VALID
            and current_epoch % self.hparams.valid_search_interval == 0
        )
        is_test_search = stage == sb.Stage.TEST

        if is_valid_search:
            hyps, _, _, _ = self.hparams.valid_search(
                enc_out.detach(), wav_lens
            )

        elif is_test_search:
            hyps, _, _, _ = self.hparams.test_search(enc_out.detach(), wav_lens)

        return p_ctc, p_seq, wav_lens, hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        (p_ctc, p_seq, wav_lens, predicted_tokens) = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        # Augment Labels
        if stage == sb.Stage.TRAIN:
            # Labels must be extended if parallel augmentation or concatenated
            # augmentation was performed on the input (increasing the time dimension)
            if (
                hasattr(self.hparams, "fea_augment")
                and self.optimizer_step > self.hparams.augment_warmup
            ):
                (
                    tokens,
                    tokens_lens,
                    tokens_eos,
                    tokens_eos_lens,
                ) = self.hparams.fea_augment.replicate_multiple_labels(
                    tokens, tokens_lens, tokens_eos, tokens_eos_lens
                )

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        )
        loss_ctc = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)
        loss = (
            self.hparams.ctc_weight * loss_ctc
            + (1 - self.hparams.ctc_weight) * loss_seq
        )

        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0 or (
                stage == sb.Stage.TEST
            ):
                # Decode token terms to words
                predicted_words = self.hparams.tokenizer(
                    predicted_tokens, task="decode_from_list"
                )

                # Convert indices to words
                target_words = undo_padding(tokens, tokens_lens)
                target_words = self.hparams.tokenizer(
                    target_words, task="decode_from_list"
                )
                self.wer_metric.append(ids, predicted_words, target_words)
                self.cer_metric.append(ids, predicted_words, target_words)

            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)
        return loss

    def init_optimizers(self):
        """Initialize optimzers and load pretrained optimizer if needed"""
        all_params = self.modules.parameters()
        self.optimizer = self.opt_class(all_params)
        self.optimizers_dict = {"opt_class": self.optimizer}
        self.checkpointer.add_recoverable("optimizer", self.optimizer)
        self.scheduler = self.hparams.scheduler(self.optimizer)

        # Load optimizer parameters
        if hasattr(self.hparams, "pretrainer"):
            opt_file = self.hparams.pretrained_path + "/optimizer.ckpt"
            opt_params = torch.load(opt_file)
            self.optimizer.load_state_dict(opt_params)

    def fit(
        self,
        epoch_counter,
        train_set,
        valid_sets={},
        progressbar=None,
        train_loader_kwargs={},
        valid_loader_kwargs={},
    ):
        """Iterate train and multiple validation sets"""
        if not (
            isinstance(train_set, sb.dataio.dataloader.DataLoader)
            or isinstance(train_set, sb.dataio.dataloader.LoopedLoader)
        ):
            train_set = self.make_dataloader(
                train_set, stage=sb.Stage.TRAIN, **train_loader_kwargs
            )

        for key, valid_set in valid_sets.items():
            if valid_set is not None and not (
                isinstance(valid_set, sb.dataio.dataloader.DataLoader)
                or isinstance(valid_set, sb.dataio.dataloader.LoopedLoader)
            ):
                valid_sets[key] = self.make_dataloader(
                    valid_set,
                    stage=sb.Stage.VALID,
                    ckpt_prefix=None,
                    **valid_loader_kwargs[key],
                )

        self.on_fit_start()

        if progressbar is None:
            progressbar = not self.noprogressbar

        # Only show progressbar if requested and main_process
        enable = progressbar and sb.utils.distributed.if_main_process()

        # Iterate epochs
        for epoch in epoch_counter:
            self._fit_train(train_set=train_set, epoch=epoch, enable=enable)

            # Iterate all valid sets
            for key, valid_set in valid_sets.items():
                self.lang = key
                self.modules.Transformer.decoder = self.hparams.decoders[key]
                self._fit_valid(valid_set=valid_set, epoch=epoch, enable=enable)

            # Debug mode only runs a few epochs
            if (
                self.debug
                and epoch == self.debug_epochs
                or self._optimizer_step_limit_exceeded
            ):
                break

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """At the end of the optimizer step, apply annealing."""
        if should_step:
            #self.hparams.noam_annealing(self.optimizer)
            self.scheduler.step()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats[f"ACC_{self.lang}"] = self.acc_metric.summarize()
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if (
                current_epoch % valid_search_interval == 0
                or stage == sb.Stage.TEST
            ):
                stage_stats[f"WER_{self.lang}"] = self.wer_metric.summarize("error_rate")
                stage_stats[f"CER_{self.lang}"] = self.cer_metric.summarize("error_rate")

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID:
            # report different epoch stages according current stage
            current_epoch = self.hparams.epoch_counter.current
            #lr = self.hparams.noam_annealing.current_lr
            #steps = self.hparams.noam_annealing.n_steps

            epoch_stats = {
                "epoch": epoch,
                "lr": self.scheduler.get_lr(),
                "steps": self.optimizer_step,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            #self.checkpointer.save_checkpoint(
            #    meta={"ACC": stage_stats["ACC"], "epoch": epoch},
            #)

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
    @provides("lang", "wav_path", "grid_path")
    def path_pipeline(path):
        """Create full path to files"""
        lang = path[:2]
        base = hparams["data_folder"] / pathlib.Path(path)
        return lang, base.with_suffix(".mp3"), base.with_suffix(".TextGrid")

    @takes("wav_path")
    @provides("signal")
    def audio_pipeline(wav_path):
        """Returns full audio sequence."""
        return data_sb.load_audio_and_resample(wav_path, hparams["sample_rate"])

    @takes("grid_path")
    @provides("words")
    def word_pipeline(grid_path):
        """Load words from file."""
        grid = textgrid.TextGrid.fromFile(grid_path)
        words = [i.mark.strip() for i in grid.getList("words")[0]]
        words = list(filter(lambda x: x and x != "<unk>", words))
        return " ".join(words)

    # Define train dataset before the text pipeline so that
    # we can use it to load all the text and write to file
    # for training the tokenizer then used in the text pipeline
    train_lang = hparams["train_languages"][0]
    pipelines = [path_pipeline, audio_pipeline, word_pipeline]
    train_dataset = DynamicItemDataset.from_json(
        hparams[f"train_{train_lang}_manifest"], dynamic_items=pipelines
    )
    datasets = {"train": train_dataset}

    # Write words to file
    text_file = pathlib.Path(hparams[f"{train_lang}_combined"])
    if not text_file.exists():
        with datasets["train"].output_keys_as(["words"]):
            text = "\n".join(d["words"] for d in datasets["train"])
            text_file.write_text(text, encoding="utf-8")

    # A different tokenizer per language
    for lang in hparams["test_languages"]:
        # Defining tokenizer and loading it, training it if not done already
        hparams[f"{lang}_tokenizer"] = SentencePiece(
            model_dir=hparams[f"tokenizer_{lang}_folder"],
            vocab_size=hparams["output_neurons"],
            model_type=hparams["token_type"],
            character_coverage=hparams["character_coverage"],
            bos_id=hparams["bos_index"],
            eos_id=hparams["eos_index"],
            text_file=str(text_file),
        )

    # Now that tokenizer is finally defined, create our token pipeline
    token_keys = ["tokens_bos", "tokens_eos", "tokens"]

    @sb.utils.data_pipeline.takes("words", "lang")
    @sb.utils.data_pipeline.provides(*token_keys)
    def token_pipeline(words, lang):
        tokens_list = hparams[f"{lang}_tokenizer"].sp.encode_as_ids(words)
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + tokens_list)
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    # Add the last remaining pipeline
    datasets["train"].add_dynamic_item(token_pipeline)
    datasets["train"].set_output_keys(["id", "signal"] + token_keys)

    # Create remaining datasets
    for dataset_name in ["valid", "test"]:
        opts = deepcopy(hparams[f"{dataset_name}_dataloader_opts"])
        datasets[dataset_name] = {}
        for test_lang in hparams["test_languages"]:
            datasets[dataset_name][test_lang] = DynamicItemDataset.from_json(
                hparams[f"{dataset_name}_{test_lang}_manifest"],
                dynamic_items=pipelines + [token_pipeline],
                output_keys=["id", "signal"] + token_keys,
            )
            
            # Copy options to language-specific dict
            hparams[f"{dataset_name}_dataloader_opts"][test_lang] = opts

    # If Dynamic Batching is used, we instantiate the needed samplers.
    if hparams["dynamic_batching"]:
        # Remove keys incompatible with batch_sampler
        for incompatible_option in ["shuffle", "batch_size", "sampler", "drop_last"]:
            for key in ["train_dataloader_opts", "valid_dataloader_opts"]:
                if incompatible_option in hparams[key]:
                    del hparams[key][incompatible_option]

        hparams["train_dataloader_opts"]["batch_sampler"] = DynamicBatchSampler(
            datasets["train"],
            length_func=lambda x: x["duration"],
            **hparams["dynamic_batch_sampler_train"],
        )
        # Different sampler per language
        for lang in hparams["test_languages"]:
            hparams["valid_dataloader_opts"][lang]["batch_sampler"] = DynamicBatchSampler(
                datasets["valid"][lang],
                length_func=lambda x: x["duration"],
                **hparams["dynamic_batch_sampler_valid"],
            )

    return datasets

def copy_decoder(hparams):
    """Copy the decoder for testing languages"""
    train_lang = hparams["train_languages"][0]
    hparams["decoders"] = {train_lang: hparams["Transformer"].decoder}

    # Copy and freeze decoder for testing languages
    for lang in set(hparams["test_languages"]) - {train_lang}:
        hparams["decoders"][lang] = deepcopy(hparams["Transformer"].decoder)
        hparams["modules"][f"{lang}_decoder"] = hparams["decoders"][lang]
        for p in hparams["decoders"][lang].parameters():
            p.requires_grad = False

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

    data_sb.make_manifests(hparams)
    datasets = dataio_prepare(hparams)
    copy_decoder(hparams)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )
