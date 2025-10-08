import sys
import json
import torch
import pathlib
import textgrid
import torchaudio
import pandas as pd
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml


class BilingualBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Computes forward pass from wavs to phonemes and wrd"""
        batch.to(self.device)
        signal, lens = batch.signal
        feats = self.hparams.compute_features(signal)
        # Permute shape to expected shape
        data = (feats.transpose(1, 2), batch.lang_enc)
        phn_out, wrd_out, hlg_out, _, _, _ = self.modules.model(data, lens)

        return wrd_out, phn_out, hlg_out

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
            wrd_pred = torch.argmax(wrd_out, dim=-1)
            phn_pred = torch.argmax(phn_out, dim=-1)
            hlg_pred = torch.argmax(hlg_out, dim=-1)

            self.wrd_count += wrd_targets.count_nonzero()
            self.phn_count += phn_targets.count_nonzero()
            self.hlg_count += hlg_targets.count_nonzero()

            wrd_correct = torch.logical_and(wrd_targets == wrd_pred, wrd_targets)
            phn_correct = torch.logical_and(phn_targets == phn_pred, phn_targets)
            hlg_correct = torch.logical_and(hlg_targets == hlg_pred, hlg_targets)
            self.wrd_correct += wrd_correct.sum()
            self.phn_correct += phn_correct.sum()
            self.hlg_correct += hlg_correct.sum()

        return wrd_loss + phn_loss + hlg_loss

    def compute_loss(self, predictions, targets):
        # Move time dimension to end for predictions
        predictions = predictions.transpose(1, 2)
        # Ignore silences and padding
        return torch.nn.functional.cross_entropy(predictions, targets, ignore_index=0)

    def on_stage_start(self, stage, epoch=None):
        """Set up metric computation"""
        self.wrd_count = torch.ones(1, device=self.device)
        self.phn_count = torch.ones(1, device=self.device)
        self.hlg_count = torch.ones(1, device=self.device)

        self.wrd_correct = torch.ones(1, device=self.device)
        self.phn_correct = torch.ones(1, device=self.device)
        self.hlg_correct = torch.ones(1, device=self.device)

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Compute metrics and save progress"""
        wrd_acc = self.wrd_correct / self.wrd_count
        phn_acc = self.phn_correct / self.phn_count
        hlg_acc = self.hlg_correct / self.hlg_count
        stats={
            "loss": stage_loss,
            "wrd_acc": wrd_acc,
            "phn_acc": phn_acc,
            "hlg_acc": hlg_acc,
        }

        if stage == sb.Stage.VALID:
            print(phn_acc)

            self.scheduler.step(phn_acc)

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

    def init_optimizers(self):
        """Initialize optimizer, scheduler and add to checkpointer"""
        self.optimizer = self.opt_class(self.modules.parameters())
        self.optimizers_dict = {"opt_class": self.optimizer}
        self.scheduler = self.hparams.lr_annealing(self.optimizer)

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("optimizer", self.optimizer)
            self.checkpointer.add_recoverable("scheduler", self.scheduler)


def make_manifests(hparams):
    """Create the manifests from the data folder"""

    for lang in hparams["train_languages"]:
        data_root = pathlib.Path(hparams["data_folder"]) / lang
        wavs = list(data_root.glob("*.mp3"))
        ids = [wav.stem for wav in wavs]
        test_size = valid_size = int(len(ids) * hparams["test_portion"])
        subsets = {
            "test": set(ids[:test_size]),
            "valid": set(ids[test_size:test_size + valid_size]),
            "train": set(ids[test_size + valid_size:]),
        }
        for stage in ["train", "valid", "test"]:
            manifest_path = pathlib.Path(hparams[f"{stage}_{lang}_manifest"])
            if not manifest_path.exists():
                manifest_path.parent.mkdir(exist_ok=True, parents=True)
                make_json(manifest_path, subsets[stage], wavs, lang)

def make_json(filename, ids, wavs, lang):
    """Create one manifest in json form"""
    manifest = {}
    for wav in wavs:
        if wav.stem not in ids:
            continue

        grid = textgrid.TextGrid.fromFile(wav.with_suffix(".TextGrid"))
        manifest[wav.stem] = {
            "wav": str(wav),
            "lang": lang,
            "frame_count": torchaudio.info(wav).num_frames,
            "wrd_grid": convert_to_tuples(grid.getList("words")[0]),
            "phn_grid": convert_to_tuples(grid.getList("phones")[0]),
        }
    
    with open(filename, "w") as f:
        json.dump(manifest, f, indent=2)

def convert_to_tuples(grid):
    """Convert grid to tuple of 'wrd', 'start', 'end'. """
    return [(i.mark.replace("ʲ", ""), i.minTime, i.maxTime) for i in grid if i.mark]

def read_label_file(filename, col):
    """Read file with list of labels."""
    return pd.read_csv(filename)[col]

def csv2map(filename, key_col, val_col):
    df = pd.read_csv(filename).dropna(subset=[key_col, val_col])
    return {k: v for k, v in zip(df[key_col], df[val_col])}

def make_datasets(hparams):
    """Create data pipelines for all stages, and label encoders."""
    hparams["lang_encoder"] = sb.dataio.encoder.CategoricalEncoder()
    hparams["lang_encoder"].update_from_iterable(hparams["supported_languages"])
    hparams["wrd_encoder"] = sb.dataio.encoder.CategoricalEncoder()
    hparams["wrd_encoder"].add_label("<sil>")
    hparams["wrd_encoder"].add_unk()

    # Add wrd from both languages so we don't have to modify architecture
    # Some words may be spelled the same but we disambiguate with a language tag
    for lang, wrd_file in hparams["wrd_files"].items():
        wrd_list = read_label_file(wrd_file, "word")
        hparams["wrd_encoder"].update_from_iterable(wrd_list + "_" + lang)

    # Add lang to phonemes to disambiguate between languages
    hparams["phn_encoder"] = sb.dataio.encoder.CategoricalEncoder()
    hparams["phn_encoder"].add_label("sil")
    hparams["phn_encoder"].add_label("spn")
    hparams["hlg_encoder"] = sb.dataio.encoder.CategoricalEncoder()
    hparams["hlg_encoder"].add_label("sil")
    hparams["hlg_encoder"].add_label("spn")
    for lang, phn_file in hparams["phn_files"].items():
        phn2hlg = csv2map(phn_file, "ipa", "homolog")
        phn2hlg["spn"] = "spn"
        hparams[f"phn2hlg_{lang}"] = phn2hlg
        hparams["hlg_encoder"].update_from_iterable(phn2hlg.values())
        disambiguated_phns = [f"{phn}_{lang}" for phn in phn2hlg]
        hparams["phn_encoder"].update_from_iterable(disambiguated_phns)

    print("# of (language-dependent) words:", len(hparams["wrd_encoder"].ind2lab))
    print("# of (language-dependent) phonemes:", len(hparams["phn_encoder"].ind2lab))
    print("# of (language-independent) homologs:", len(hparams["hlg_encoder"].ind2lab))

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("signal")
    def audio_pipeline(wav):
        return sb.dataio.dataio.read_audio(wav)

    @sb.utils.data_pipeline.takes("lang", "wrd_grid", "phn_grid", "frame_count")
    @sb.utils.data_pipeline.provides("lang_enc", "wrd_targets", "phn_targets", "hlg_targets")
    def label_pipeline(lang, wrd_timings, phn_timings, frame_count):

        yield hparams["lang_encoder"].encode_label(lang)

        target_size = frame_count // hparams["downsample_factor"] + 1
        target_rate = hparams["fs"] // hparams["downsample_factor"]
        wrd_label_sequence = torch.zeros(target_size, dtype=torch.long)
        phn_label_sequence = torch.zeros(target_size, dtype=torch.long)
        hlg_label_sequence = torch.zeros(target_size, dtype=torch.long)

        # Iterate words to create frame-level targets at the specified rate
        for wrd, start, stop in wrd_timings:
            start, stop = int(start * target_rate), int(stop * target_rate)
            wrd_label_sequence[start:stop] = hparams["wrd_encoder"].encode_label_torch(wrd + "_" + lang)

        yield wrd_label_sequence

        # Iterate phonemees to create frame-level targets at the specified rate
        for phn, start, stop in phn_timings:
            start, stop = int(start * target_rate), int(stop * target_rate)
            phn_label_sequence[start:stop] = hparams["phn_encoder"].encode_label_torch(phn + "_" + lang)
            hlg = hparams[f"phn2hlg_{lang}"][phn]
            hlg_label_sequence[start:stop] = hparams["hlg_encoder"].encode_label_torch(hlg)

        yield phn_label_sequence
        yield hlg_label_sequence

    datasets = {}
    for stage in ["train", "valid", "test"]:
        datasets[stage] = sb.dataio.dataset.DynamicItemDataset.from_json(
            hparams[f"{stage}_fr_manifest"],
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "signal", "lang_enc", "wrd_targets", "phn_targets", "hlg_targets"],
        ).filtered_sorted(sort_key="frame_count")
    
    return datasets

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

    # Manifests will only be made once, datasets every time
    make_manifests(hparams)
    datasets = make_datasets(hparams)

    bilingual_brain = BilingualBrain(
        modules={"model": hparams["model"]},
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
        max_key="accuracy",
        test_loader_kwargs=hparams["dataloader_options"],
    )
