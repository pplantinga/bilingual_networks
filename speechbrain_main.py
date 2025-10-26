"""
Speechbrain recipe for training a multilingual phoneme/homolog/word recognitionmodel
for investigating the effects of language attrition from lack of exposure.

To run:

> python speechbrain_main.py experiments/commonvoice_fr_phonemes_words.yaml --data_folder data

Author:
 * Peter Plantinga
"""
import sys
import json
import tqdm
import torch
import joblib
import logging
import pathlib
import textgrid
import torchaudio
import numpy as np
import pandas as pd
import soundfile as sf
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml

logger = sb.utils.logger.get_logger("speechbrain_main.py")

import gc
import psutil
import os

process = psutil.Process(os.getpid())

class BilingualBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Computes forward pass from wavs to phonemes and wrd"""
        batch.to(self.device)
        signal, lens = batch.signal
        feats = self.hparams.compute_features(signal)
        #wrd_out, phn_out, hlg_out = self.modules.model(feats, batch.lang_enc)
        phn_out, wrd_out, hlg_out, _, _, _ = self.modules.model((feats.transpose(1, 2), batch.lang_enc), lengths=None)

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
            # Where targets are nonzero, compute accuracy, expects [batch, class, time]
            self.hparams.word_metric(wrd_out.transpose(1, 2).detach(), wrd_targets.detach())
            self.hparams.phone_metric(phn_out.transpose(1, 2).detach(), phn_targets.detach())
            self.hparams.homolog_metric(hlg_out.transpose(1, 2).detach(), hlg_targets.detach())

        return wrd_loss + phn_loss + hlg_loss

    def compute_loss(self, predictions, targets):
        """Compute cross-entropy loss, ignoring the "silence" and padding index: 0"""
        # Move time dimension to end for predictions
        predictions = predictions.transpose(1, 2)

        # Ignore silences and padding
        return torch.nn.functional.cross_entropy(input=predictions, target=targets, ignore_index=0)

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Compute metrics and save progress"""

        gc.collect()
        torch.cuda.empty_cache()
        print("Cuda Allocated:", torch.cuda.memory_allocated()/1e6)
        print("Cuda Reserved:", torch.cuda.memory_reserved()/1e6)
        tensors = [o for o in gc.get_objects() if torch.is_tensor(o)]
        print("num tensors:", len(tensors))
        memory_info = process.memory_info()
        memory_usage_bytes = memory_info.rss  # Resident Set Size
        print(f"Memory Usage: {memory_usage_bytes / (1024 * 1024):.2f} MB")


        if stage != sb.Stage.TRAIN:
            stats={
                "loss": stage_loss,
                "wrd_acc": round(self.hparams.word_metric.compute().item(), 3),
                "phn_acc": round(self.hparams.phone_metric.compute().item(), 3),
                "hlg_acc": round(self.hparams.homolog_metric.compute().item(), 3),
            }

        if stage == sb.Stage.VALID:
            self.scheduler.step(stats["phn_acc"])
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
        test_size = valid_size = int(len(wavs) * hparams["test_portion"])
        subsets = {
            "test": wavs[:test_size],
            "valid": wavs[test_size:test_size + valid_size],
            "train": wavs[test_size + valid_size:],
        }
        for stage in ["train", "valid", "test"]:
            manifest_path = pathlib.Path(hparams[f"{stage}_{lang}_manifest"])
            if not manifest_path.exists():
                manifest_path.parent.mkdir(exist_ok=True, parents=True)
                logger.info(f"Creating {stage} manifest:")
                make_json(manifest_path, subsets[stage], lang)

def make_json(filename, subset, lang):
    """Create one manifest in json form"""
    manifest = {
        key: item
        for key, item in tqdm.tqdm(
            joblib.Parallel(n_jobs=8, return_as='generator_unordered')(
                joblib.delayed(make_item)(wav, lang)
                for wav in subset
            ),
            disable=None,
            total=len(subset),
        )
    }
    with open(filename, "w", encoding="utf8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

def make_item(wav, lang):
    """Create a single item of the manifest, to be run in parallel"""
    item = {
        "wav": str(wav),
        "lang": lang,
        "frame_count": torchaudio.info(wav).num_frames,
    }

    return (wav.stem, item)

def convert_to_tuples(grid):
    """Convert grid to tuple of 'wrd', 'start', 'end'. """
    def clean(mark):
        return mark.replace("ʲ", "").replace("m^{me}", "me")
    return [(clean(i.mark), i.minTime, i.maxTime) for i in grid if i.mark]

def read_label_file(filename, col):
    """Read file with list of labels."""
    return pd.read_csv(filename)[col]

def csv2map(filename, key_col, val_col):
    """Create a mapping from one column of a csv to another"""
    df = pd.read_csv(filename).dropna(subset=[key_col, val_col])
    return {k: v for k, v in zip(df[key_col], df[val_col])}


def make_encoders(hparams):
    # Language is set to "unknown" with some chance
    hparams["lang_encoder"] = sb.dataio.encoder.CategoricalEncoder()
    hparams["lang_encoder"].expect_len(len(hparams["supported_languages"]) + 1)
    hparams["lang_encoder"].add_unk()
    hparams["lang_encoder"].update_from_iterable(hparams["supported_languages"])

    # Put unknown words at index 0 and ignore them
    hparams["wrd_encoder"] = sb.dataio.encoder.CategoricalEncoder()
    hparams["wrd_encoder"].expect_len(hparams["word_outputs"])
    hparams["wrd_encoder"].add_unk()

    # Add words from both languages so we don't have to modify architecture
    # Some words may be spelled the same but we disambiguate with a language tag
    for lang, wrd_file in hparams["wrd_files"].items():
        wrd_list = read_label_file(wrd_file, "word")
        hparams["wrd_encoder"].update_from_iterable(wrd_list + "_" + lang)

    # Index 0 is silence in all cases and can be safely ignored.
    # Montreal Forced Aligner uses "spn" as a sort of "unknown speech noise"
    hparams["phn_encoder"] = sb.dataio.encoder.CategoricalEncoder()
    hparams["phn_encoder"].expect_len(hparams["phone_outputs"])
    hparams["phn_encoder"].add_label("sil")
    hparams["phn_encoder"].add_label("spn")
    hparams["hlg_encoder"] = sb.dataio.encoder.CategoricalEncoder()
    hparams["hlg_encoder"].expect_len(hparams["homolog_outputs"])
    hparams["hlg_encoder"].add_label("sil")
    hparams["hlg_encoder"].add_label("spn")

    # Iterate language phone files to add all symbols to encoders
    for lang, phn_file in hparams["phn_files"].items():

        # Build map and convert phoneme labels to lang-independent homologs
        phn2hlg = csv2map(phn_file, "ipa", "homolog")
        phn2hlg["spn"] = "spn"
        hparams[f"phn2hlg_{lang}"] = phn2hlg
        hparams["hlg_encoder"].update_from_iterable(phn2hlg.values())

        # Similarly to words, add language tag to disambiguate between languages
        disambiguated_phns = [f"{phn}_{lang}" for phn in phn2hlg]
        hparams["phn_encoder"].update_from_iterable(disambiguated_phns)

    # The phone/word counts are crucial for setting up the architecture correctly
    logger.info(f"# of (language-dependent) words: {len(hparams['wrd_encoder'].ind2lab)}")
    logger.info(f"# of (language-dependent) phonemes: {len(hparams['phn_encoder'].ind2lab)}")
    logger.info(f"# of (language-independent) homologs: {len(hparams['hlg_encoder'].ind2lab)}")


def make_datasets(hparams):
    """Create data pipelines for all stages, and label encoders."""

    # Precompute stuff for downsampling and cropping
    target_rate = hparams["fs"] // hparams["downsample_factor"]
    max_crop_len = int(hparams["random_crop_len"] * target_rate) + 1
    df = hparams["downsample_factor"]

    @sb.utils.data_pipeline.takes("lang")
    @sb.utils.data_pipeline.provides("lang_enc")
    def lang_pipeline(lang):
        return hparams["lang_encoder"].encode_label(lang)

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("signal", "crop_start")
    def audio_pipeline(wav):

        # Resample audio to target rate
        with torch.no_grad():
            audio, sr = sf.read(wav)
            audio = audio[::3].astype(np.float32)

            # Select random crop of the audio
            max_start = max(1, len(audio) // df - max_crop_len)
            crop_start = np.random.randint(max_start)
            crop_end = crop_start + max_crop_len - 1
            signal = audio[crop_start * df:crop_end * df].copy()
            del audio

        return signal, crop_start

    @sb.utils.data_pipeline.takes("lang", "wav", "signal", "crop_start")
    @sb.utils.data_pipeline.provides("wrd_targets", "phn_targets", "hlg_targets")
    def label_pipeline(lang, wav, signal, crop_start):
        """Encode the inputs/targets"""

        def time2rate(time):
            return int(time * target_rate)

        # Create time-aligned target vectors based on alignment info in manifest
        grid_path = pathlib.Path(wav).with_suffix(".TextGrid")
        grid = textgrid.TextGrid.fromFile(grid_path)
        crop_len = len(signal) // df + 1
        wrd_label_sequence = np.zeros(crop_len, dtype=int)
        phn_label_sequence = np.zeros(crop_len, dtype=int)
        hlg_label_sequence = np.zeros(crop_len, dtype=int)

        # Iterate words to create frame-level targets at the specified rate
        for wrd, start, stop in convert_to_tuples(grid.getList("words")[0]):
            start_idx = max(time2rate(start) - crop_start, 0)
            stop_idx = min(time2rate(stop) - crop_start, crop_len)
            if stop_idx > 0 and start_idx < crop_len:
                encoded_wrd = hparams["wrd_encoder"].encode_label_torch(wrd + "_" + lang).item()
                wrd_label_sequence[start_idx:stop_idx] = encoded_wrd

        # Iterate phonemees to create frame-level targets at the specified rate
        for phn, start, stop in convert_to_tuples(grid.getList("phones")[0]):
            start_idx = max(time2rate(start) - crop_start, 0)
            stop_idx = min(time2rate(stop) - crop_start, crop_len)
            if stop_idx > 0 and start_idx < crop_len:
                encoded_phn = hparams["phn_encoder"].encode_label_torch(phn + "_" + lang).item()
                phn_label_sequence[start_idx:stop_idx] = encoded_phn
                hlg = hparams[f"phn2hlg_{lang}"][phn]
                hlg_label_sequence[start_idx:stop_idx] = hparams["hlg_encoder"].encode_label_torch(hlg)

        return wrd_label_sequence, phn_label_sequence, hlg_label_sequence

    datasets = {}
    for stage in ["train", "valid", "test"]:
        datasets[stage] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{stage}_fr_manifest"],
            dynamic_items=[lang_pipeline, audio_pipeline, label_pipeline],
            output_keys=["id", "signal", "lang_enc", "wrd_targets", "phn_targets", "hlg_targets"],
        )#.filtered_sorted(sort_key="frame_count")
    
    return datasets


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
    make_manifests(hparams)
    make_encoders(hparams)
    datasets = make_datasets(hparams)

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
        max_key="accuracy",
        test_loader_kwargs=hparams["dataloader_options"],
    )
