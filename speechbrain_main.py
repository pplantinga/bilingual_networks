"""
Speechbrain recipe for training a multilingual phoneme/homolog/word recognition model
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

logger = sb.utils.logger.get_logger("speechbrain_main.py")


class BilingualBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Computes forward pass from wavs to phonemes and wrd"""
        batch.to(self.device)
        signal, lens = batch.signal
        feats = self.hparams.compute_features(signal)
        return self.modules.model(feats, batch.lang_enc)

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss between predicted and actual wrd and phonemes."""
        wrd_out, phn_out, hlg_out = predictions
        # Ignore lengths, they should match the predictions by design
        wrd_targets, _ = batch.wrd_targets
        phn_targets, _ = batch.phn_targets
        hlg_targets, _ = batch.hlg_targets
        wrd_loss = self.compute_loss(predictions=wrd_out, targets=wrd_targets)
        phn_loss = self.compute_loss(predictions=phn_out, targets=phn_targets)
        hlg_loss = self.compute_loss(predictions=hlg_out, targets=hlg_targets)

        if stage != sb.Stage.TRAIN:
            # Where targets are nonzero, compute accuracy, expects [batch, class, time]
            self.hparams.word_metric(wrd_out.transpose(1, 2), wrd_targets)
            self.hparams.phone_metric(phn_out.transpose(1, 2), phn_targets)
            self.hparams.homolog_metric(hlg_out.transpose(1, 2), hlg_targets)

        return wrd_loss + phn_loss + hlg_loss

    def compute_loss(self, predictions, targets):
        """Compute cross-entropy loss, ignoring the "silence" and padding index: 0"""
        # Move time dimension to end for predictions
        predictions = predictions.transpose(1, 2)

        # Ignore silences and padding
        return torch.nn.functional.cross_entropy(input=predictions, target=targets, ignore_index=0)

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """Update LR after every batch"""
        if should_step:
            self.hparams.lr_annealing(self.optimizer)

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Compute metrics and save progress"""

        if stage != sb.Stage.TRAIN:
            stats={
                "loss": stage_loss,
                "wrd_acc": round(self.hparams.word_metric.compute().item(), 3),
                "phn_acc": round(self.hparams.phone_metric.compute().item(), 3),
                "hlg_acc": round(self.hparams.homolog_metric.compute().item(), 3),
            }

        if stage == sb.Stage.VALID:
            self.hparams.word_metric.reset()
            self.hparams.phone_metric.reset()
            self.hparams.homolog_metric.reset()

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

    # Manifests will only be made once, encoders and datasets every time
    data_sb.make_manifests(hparams)
    data_sb.make_encoders(hparams)
    datasets = data_sb.make_datasets(hparams)

    # Create trainer
    bilingual_brain = BilingualBrain(
        modules={k: hparams[k] for k in ["model", "word_metric", "phone_metric", "homolog_metric"]},
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
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # Test
    bilingual_brain.evaluate(
        datasets["test"],
        test_loader_kwargs=hparams["dataloader_options"],
    )
