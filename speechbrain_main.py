"""
Speechbrain recipe for training a multilingual phonme/homolog/word recognition model
for investigating the effects of language attrition from lack of exposure.

To run:

> python speechbrain_main.py experiments/pretrain_fr.yaml --data_folder data

Author:
 * Peter Plantinga
"""
import sys
import torch
import data_sb
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from torch.nn.functional import cross_entropy

logger = sb.utils.logger.get_logger("speechbrain_main.py")


class BilingualBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Computes forward pass from wavs to phonmes and word"""
        batch.to(self.device)
        signal, lens = batch.signal
        feats = self.hparams.compute_features(signal)
        return self.modules.model(feats)

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss between predicted and actual word and phonmes."""
        phon_out, lang_out, word_out = predictions
        # Ignore lengths, they should match the predictions by design
        phon_targets, _ = batch.phon_targets
        lang_targets, _ = batch.lang_targets
        word_targets, _ = batch.word_targets

        # Phones and words ignore empty frames, which have a label of "0"
        phon_loss = cross_entropy(phon_out.transpose(1, 2), phon_targets, ignore_index=0)
        word_loss = cross_entropy(word_out.transpose(1, 2), word_targets, ignore_index=0)
        # But languages are just 0, 1, 2, so we don't ignore "0"
        lang_loss = cross_entropy(lang_out.transpose(1, 2), lang_targets)

        if stage != sb.Stage.TRAIN:
            # Where targets are nonzero, compute accuracy, expects [batch, class, time]
            self.hparams.phon_metric(phon_out.transpose(1, 2), phon_targets)
            self.hparams.lang_metric(lang_out.transpose(1, 2), lang_targets)
            self.hparams.word_metric(word_out.transpose(1, 2), word_targets)

        return phon_loss + lang_loss + word_loss

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """Update LR after every batch"""
        if should_step:
            self.hparams.lr_annealing(self.optimizer)

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Compute metrics and save progress"""

        if stage != sb.Stage.TRAIN:
            stats={
                "loss": stage_loss,
                "phon_acc": round(self.hparams.phon_metric.compute().item(), 3),
                "word_acc": round(self.hparams.word_metric.compute().item(), 3),
                "lang_acc": round(self.hparams.lang_metric.compute().item(), 3),
            }

        if stage == sb.Stage.VALID:
            self.hparams.phon_metric.reset()
            self.hparams.lang_metric.reset()
            self.hparams.word_metric.reset()

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch},
                valid_stats=stats,
            )
            self.checkpointer.save_and_keep_only()

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )


#######################################
# MAIN
#######################################
if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides
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

    # Manifests will only be made once, encoders and datasets every time
    data_sb.make_manifests(hparams)
    data_sb.make_encoders(hparams)
    datasets = data_sb.make_datasets(hparams)

    # Create trainer
    bilingual_brain = BilingualBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Training/validation loop
    bilingual_brain.fit(
        bilingual_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["test_loader_options"],
    )

    # Test
    bilingual_brain.evaluate(
        datasets["test"],
        test_loader_kwargs=hparams["test_loader_options"],
    )
