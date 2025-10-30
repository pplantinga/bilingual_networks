"""
Data loading machinery for SpeechBrain recipe. See speechbrain_main.py

Author:
 * Peter Plantinga
"""
import json
import tqdm
import torch
import joblib
import pathlib
import textgrid
import torchaudio
import numpy as np
import pandas as pd
import soundfile as sf
import speechbrain as sb

logger = sb.utils.logger.get_logger("speechbrain_data.py")


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

translate_table = str.maketrans("", "", "ʲʷʰː")
def clean(mark):
    """Convert to somewhat simplified phoneme set"""
    return mark.translate(translate_table).replace("m^{me}", "me").replace("d̪", "d")

def convert_to_tuples(grid):
    """Convert grid to tuple of 'wrd', 'start', 'end'. """
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


@torch.no_grad()
def load_audio_and_resample(wav, target_sr=16000):
    audio, sr = sf.read(wav)
    audio = audio.astype(np.float32)

    # Faster to just take every other sample or whatever
    if sr % target_sr == 0:
        audio = audio[::sr // target_sr]
    else:
        audio = torchaudio.transforms.Resample(
            orig_freq=sr,
            new_freq=target_sr,
            lowpass_filter_width=4,
        )(torch.tensor(audio)).numpy()

    return audio


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
    def train_audio_pipeline(wav):
        audio = load_audio_and_resample(wav, hparams["fs"])

        # Select random crop of the audio
        max_start = max(1, len(audio) // df - max_crop_len)
        crop_start = np.random.randint(max_start)
        crop_end = crop_start + max_crop_len - 1
        signal = audio[crop_start * df:crop_end * df].copy()
        del audio

        # Return starting time so labels can be aligned
        return signal, crop_start

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("signal", "crop_start")
    def test_audio_pipeline(wav):
        audio = load_audio_and_resample(wav, hparams["fs"])
        return audio, 0

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
        audio_pipeline = train_audio_pipeline if stage == "train" else test_audio_pipeline
        datasets[stage] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{stage}_en_manifest"],
            dynamic_items=[lang_pipeline, audio_pipeline, label_pipeline],
            output_keys=["id", "signal", "lang_enc", "wrd_targets", "phn_targets", "hlg_targets"],
        )#.filtered_sorted(sort_key="frame_count")
    
    return datasets
