#!/usr/bin/env python3
"""Recipe for training a Transformer ASR system with MFA-aligned phonemes
from Commonvoice. The phonemes are used as simple discrete inputs.
The system employs an encoder, a decoder, and an attention mechanism
between them. Decoding is performed with (CTC/Att joint) beamsearch.


Authors
* Peter Plantinga 2026
"""

import os
import sys
import pathlib
import textgrid

import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml

import data_sb
import speechbrain as sb
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.logger import get_logger
from speechbrain.utils.data_pipeline import takes, provides


logger = get_logger(__name__)


# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the phonemes to the token probabilities."""
        batch = batch.to(self.device)
        phones, phone_lens = batch.phone_targets
        tokens_bos, _ = batch.tokens_bos

        # compute features
        src = self.modules.embedding(phones)

        # forward modules
        enc_out, pred = self.modules.Transformer(
            src, tokens_bos, phone_lens, pad_idx=self.hparams.pad_index
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
                enc_out.detach(), phone_lens
            )

        elif is_test_search:
            hyps, _, _, _ = self.hparams.test_search(enc_out.detach(), phone_lens)

        return p_ctc, p_seq, phone_lens, hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        (p_ctc, p_seq, phone_lens, predicted_tokens) = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        )
        loss_ctc = self.hparams.ctc_cost(p_ctc, tokens, phone_lens, tokens_lens)
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
            stage_stats["ACC"] = self.acc_metric.summarize()
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if (
                current_epoch % valid_search_interval == 0
                or stage == sb.Stage.TEST
            ):
                stage_stats["WER"] = self.wer_metric.summarize("error_rate")
                stage_stats["CER"] = self.cer_metric.summarize("error_rate")

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
            #self.checkpointer.save_and_keep_only(
            self.checkpointer.save_checkpoint(
                meta={"ACC": stage_stats["ACC"], "epoch": epoch},
            #    max_keys=["ACC"],
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            def write_file():
                with open(
                    self.hparams.test_wer_file, "w", encoding="utf-8"
                ) as w:
                    self.wer_metric.write_stats(w)
            run_on_main(write_file)


# Define custom data procedure
def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """

    # Re-use make_datasets first
    data_sb.make_encoders(hparams)
    datasets = data_sb.make_datasets(hparams)

    # First add words so that we can use them for training a tokenizer
    @takes("grid_path")
    @provides("words")
    def word_pipeline(grid_path):
        """Load words from file."""
        grid = textgrid.TextGrid.fromFile(grid_path)
        words = [i.mark.strip() for i in grid.getList("words")[0]]
        words = list(filter(lambda x: x and x != "<unk>", words))
        return " ".join(words)

    for dataset in datasets.values():
        dataset.add_dynamic_item(word_pipeline)

    # Write words to file
    lang = hparams["train_language"]
    text_file = pathlib.Path(hparams[f"{lang}_combined"])
    if not text_file.exists():
        with datasets["train"].output_keys_as(["words"]):
            text = "\n".join(d["words"] for d in datasets["train"])
            text_file.write_text(text, encoding="utf-8")

    # Defining tokenizer and loading it, training it if not done already
    hparams["tokenizer"] = SentencePiece(
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

    @sb.utils.data_pipeline.takes("words")
    @sb.utils.data_pipeline.provides(*token_keys)
    def token_pipeline(words):
        tokens_list = hparams["tokenizer"].sp.encode_as_ids(words)
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + tokens_list)
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    # Add the last remaining pipeline
    for dataset in datasets.values():
        dataset.add_dynamic_item(token_pipeline)
        dataset.set_output_keys(["id", "phone_targets"] + token_keys)

    # If Dynamic Batching is used, we instantiate the needed samplers.
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

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
        hparams["valid_dataloader_opts"]["batch_sampler"] = DynamicBatchSampler(
            datasets["valid"],
            length_func=lambda x: x["duration"],
            **hparams["dynamic_batch_sampler_valid"],
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

    # Load pretrained weights if available
    if "pretrainer" in hparams:
        hparams["pretrainer"].collect_files()
        hparams["pretrainer"].load_collected()

    data_sb.make_manifests(hparams)
    datasets = dataio_prepare(hparams)

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
