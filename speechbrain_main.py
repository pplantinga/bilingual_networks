"""
Speechbrain recipe for training a multilingual phonme/homolog/word recognition model
for investigating the effects of language attrition from lack of exposure.

To run:

> python speechbrain_main.py experiments/pretrain_fr.yaml --data_folder data

Author:
 * Peter Plantinga
"""
import sys
import json
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

        # Losses expect time last
        phone_out = predictions[0].transpose(1, 2)
        word_out = predictions[1].transpose(1, 2)

        # Ignore target lengths, they should match the predictions by design
        word_targets, _ = batch.word_targets

        # Phones and words ignore empty frames, which have a label of "0"
        word_loss = cross_entropy(word_out, word_targets, ignore_index=0)
        phone_loss = 0
        if self.hparams.phone_feedback:
            phone_targets, _ = batch.phone_targets
            phone_loss = cross_entropy(phone_out, phone_targets, ignore_index=0)

        if stage != sb.Stage.TRAIN:
            for lang in self.hparams.train_languages:
                lang_mask = getattr(batch, f"{lang}_mask")
                if not lang_mask.any():
                    continue

                self.hparams.word_metrics[lang](
                    word_out[lang_mask], word_targets[lang_mask]
                )

                if self.hparams.phone_feedback:
                    self.hparams.phone_metrics[lang](
                        phone_out[lang_mask], phone_targets[lang_mask]
                    )

            if hasattr(self.hparams, "phone_bin_metrics"):
                predictions = phone_out.argmax(dim=1)
                for phoneme, metric in self.hparams.phone_bin_metrics.items():
                    metric.to(self.device)
                    phoneme_index = self.hparams.phone_encoder.encode_label_torch(phoneme).item()
                    bin_preds = predictions == phoneme_index
                    bin_targs = phone_targets == phoneme_index
                    metric(bin_preds, bin_targs)

        return phone_loss + word_loss

    def on_fit_start(self):
        super().on_fit_start()

        self.metric_tracker = []

    def init_optimizers(self):
        all_params = self.modules.parameters()
        self.optimizer = self.opt_class(all_params)
        self.optimizers_dict = {"opt_class": self.optimizer}
        self.checkpointer.add_recoverable("optimizer", self.optimizer)

        # Load optimizer parameters
        if hasattr(self.hparams, "pretrainer"):
            opt_file = self.hparams.pretrained_path + "/optimizer.ckpt"
            opt_params = torch.load(opt_file)
            self.optimizer.load_state_dict(opt_params)

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """Update LR after every batch"""
        if should_step:
            if hasattr(self.hparams, "lr_annealing"):
                self.hparams.lr_annealing(self.optimizer)
            if self.modules.model.phone_bottleneck:
                self.modules.model.step_bottleneck_temp()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Compute metrics and save progress"""

        def compute_metric(metric):
            score = round(metric.compute().item(), 5)
            metric.reset()
            return score

        if stage != sb.Stage.TRAIN:
            stats={"loss": round(stage_loss, 5)}
            stats.update({
                f"word_{lang}_acc": compute_metric(self.hparams.word_metrics[lang])
                for lang in self.hparams.train_languages
            })
            if self.hparams.phone_feedback:
                stats.update({
                    f"phone_{lang}_acc": compute_metric(self.hparams.phone_metrics[lang])
                    for lang in self.hparams.train_languages
                })

                # Save confusions to file
                #confusions = self.hparams.phone_confusion.compute().cpu()
                #if stage == sb.Stage.VALID:
                #    torch.save(confusions, self.hparams.confusions_valid + "." + lang)
                #else:
                #    torch.save(confusions, self.hparams.confusions_test + "." + lang)

            if hasattr(self.hparams, "phone_bin_metrics"):
                stats.update({
                    f"phone_{p}_f1": compute_metric(self.hparams.phone_bin_metrics[p])
                    for p in self.hparams.phone_bin_metrics
                })

        if stage == sb.Stage.VALID:
            if hasattr(self.hparams, "lr_annealing"):
                lr = self.hparams.lr_annealing.current_lr
            else:
                lr = self.hparams.lr

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch}, train_stats={"lr": lr}, valid_stats=stats,
            )
            self.metric_tracker.append({"epoch": epoch, **stats})

            if epoch % self.hparams.checkpoint_after_epochs == 0:
                self.checkpointer.save_checkpoint()

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

            with open(self.hparams.metric_log, "w") as f:
                json.dump(self.metric_tracker, f, indent=2)


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

        # Freeze params if requested
        if "freeze_model" in hparams and hparams["freeze_model"]:
            for p in hparams["model"].parameters():
                p.requires_grad = False

            # Unfreeze phoneme output layer
            for p in hparams["model"].phone_out.parameters():
                p.requires_grad = True


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
