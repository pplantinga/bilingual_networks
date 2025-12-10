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
from speechbrain.utils.data_pipeline import takes, provides
from speechbrain.dataio.sampler import ReproducibleWeightedRandomSampler

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
    info = torchaudio.info(wav)
    dur = round(info.num_frames / info.sample_rate, 2)
    item = {"path": f"{lang}/{wav.stem}", "lang": lang, "duration": dur}

    return (wav.stem, item)


translate_table = str.maketrans("", "", "ʲʷʰː\u0329\u032A")
def clean(mark):
    """Convert to somewhat simplified phoneme set"""
    return mark.translate(translate_table).replace("m^{me}", "me")


def convert_to_tuples(grid):
    """Convert grid to tuple of 'word', 'start', 'end'. """
    return [(clean(i.mark), i.minTime, i.maxTime) for i in grid if i.mark]


def mapping_from_csv(filename, key_col, val_col):
    """Creates mapping from one column to another"""
    df = pd.read_csv(filename, usecols=[key_col, val_col])
    return df.set_index(key_col)[val_col].to_dict()


class PhonemeEncoder(sb.dataio.encoder.CategoricalEncoder):
    """Special encoder to handle converting ipa to homologs"""
    def __init__(self, ipa2hlg, *args, **kwargs):
        self.ipa2hlg = ipa2hlg
        super().__init__(*args, **kwargs)

    def encode_label(self, label, allow_unk=True):
        """Convert ipa input to language-independent version before encoding"""
        if label in self.ipa2hlg:
            label = self.ipa2hlg[label]
        return super().encode_label(label, allow_unk)


def make_encoders(hparams):
    # Put unknown words at index 0 and ignore them
    hparams["word_encoder"] = sb.dataio.encoder.CategoricalEncoder()
    hparams["word_encoder"].expect_len(hparams["word_outputs"])
    hparams["word_encoder"].add_unk()

    # Add words from both languages so we don't have to modify architecture
    # Some words may be spelled the same but we disambiguate with a language tag
    for lang, word_file in hparams["word_files"].items():
        word_list = pd.read_csv(word_file)["word"]
        hparams["word_encoder"].update_from_iterable(word_list + "_" + lang)
    logger.info(f"# of (language-dependent) words: {len(hparams['word_encoder'].ind2lab)}")

    # Iterate language phone files to add all symbols to encoders
    if hparams["phone_feedback"]:
        ipa2hlg = {}
        for lang, phon_file in hparams["phon_files"].items():
            ipa2hlg.update(mapping_from_csv(phon_file, "ipa", "homolog"))

        # Index 0 is silence in all cases and can be safely ignored.
        # Montreal Forced Aligner uses "spn" as a sort of "unknown speech noise"
        # "spn" and "sil" are mapped to this unk label and ignored
        hparams["phon_encoder"] = PhonemeEncoder(ipa2hlg)
        hparams["phon_encoder"].expect_len(hparams["phone_outputs"])
        hparams["phon_encoder"].add_unk()
        hparams["phon_encoder"].update_from_iterable(sorted(set(ipa2hlg.values())))
        logger.info(f"# of (language-independent) phonemes: {len(hparams['phon_encoder'].ind2lab)}")


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

    def grid2array(grid, field, crop_start, crop_len, encoder, postfix):
        """Convert a list from an alignment grid to an array suitable
        for use as a target tensor."""
        array = np.zeros(crop_len, dtype=int)
        for name, start, stop in convert_to_tuples(grid.getList(field)[0]):
            start_idx = max(int(start * target_rate) - crop_start, 0)
            stop_idx = min(int(stop * target_rate) - crop_start, crop_len)
            if stop_idx > 0 and start_idx < crop_len:
                encoded = encoder.encode_label_torch(name + postfix).item()
                array[start_idx:stop_idx] = encoded

        return array

    @takes("path")
    @provides("wav_path", "grid_path")
    def path_pipeline(path):
        """Create full path to files"""
        base = hparams["data_folder"] / pathlib.Path(path)
        return base.with_suffix(".mp3"), base.with_suffix(".TextGrid")

    @takes("wav_path")
    @provides("signal", "crop_start", "crop_len")
    def train_audio_pipeline(wav_path):
        """Training pipeline returns only a chunk for training efficiency"""
        audio = load_audio_and_resample(wav_path, hparams["fs"])

        # Select random crop of the audio
        max_start = max(1, len(audio) // df - max_crop_len)
        crop_start = np.random.randint(max_start)
        crop_end = crop_start + max_crop_len - 1
        signal = audio[crop_start * df:crop_end * df].copy()
        del audio

        # Return starting time and len so labels can be aligned
        crop_len = len(signal) // df + 1
        return signal, crop_start, crop_len

    @takes("wav_path")
    @provides("signal", "crop_start", "crop_len")
    def test_audio_pipeline(wav_path):
        """Testing pipeline returns full audio starting from 0."""
        audio = load_audio_and_resample(wav_path, hparams["fs"])
        crop_len = len(audio) // df + 1
        return audio, 0, crop_len

    @takes("grid_path")
    @provides("grid")
    def grid_pipeline(grid_path):
        """Load grid from file."""
        return textgrid.TextGrid.fromFile(grid_path)

    @takes("grid", "crop_start", "crop_len")
    @provides("phon_targets")
    def phon_pipeline(grid, crop_start, crop_len):
        """Create time-aligned phoneme targets based on textgrid."""
        return grid2array(
            grid, "phones", crop_start, crop_len, hparams["phon_encoder"], postfix=""
        )

    @takes("lang", "grid", "crop_start", "crop_len")
    @provides("word_targets")
    def word_pipeline(lang, grid, crop_start, crop_len):
        """Create time-aligned language-disambiguated word targets based on textgrid."""
        return grid2array(
            grid, "words", crop_start, crop_len, hparams["word_encoder"], postfix=f"_{lang}"
        )

    # This bit of shenanigans creates a separate language mask pipeline for each train lang
    def create_mask_pipeline(lang):
        @takes("lang")
        @provides(f"{lang}_mask")
        def mask_pipeline(l):
            return lang == l
        return mask_pipeline

    # Glom together all the pipelines and output keys
    pipelines = [path_pipeline, grid_pipeline, word_pipeline]
    output_keys = ["id", "lang", "signal", "lang", "word_targets"]
    if hparams["phone_feedback"]:
        pipelines.append(phon_pipeline)
        output_keys.append("phon_targets")
    for lang in hparams["train_languages"]:
        pipelines.append(create_mask_pipeline(lang))
        output_keys.append(f"{lang}_mask")

    datasets = {}
    for stage in ["train", "valid", "test"]:
        # Load data manually so that we can concatenate before creating our dataset
        data = {}
        for lang in hparams["train_languages"]:
            data.update(json.load(open(hparams[f"{stage}_{lang}_manifest"])))

        # Create dataset from components defined above
        audio_pipeline = train_audio_pipeline if stage == "train" else test_audio_pipeline
        datasets[stage] = sb.dataio.dataset.DynamicItemDataset(
            data, pipelines + [audio_pipeline], output_keys
        )

        if stage in ["valid", "test"]:
            # Long utterances cause OOM on validation, limit max length
            datasets[stage] = datasets[stage].filtered_sorted(
                key_max_value={"duration": 10.0},
                #key_test={"lang": lambda x: x == "en"},
                sort_key="duration",
            )

    # Enable random sampling if we're doing multilingual training
    # Or if we just want to take a random subset of samples each epoch
    if len(hparams["train_languages"]) > 1 or "samples_per_epoch" in hparams:
        if len(hparams["train_languages"]) > 1:
            with datasets["train"].output_keys_as(["lang"]):
                weights = [hparams[f"{d['lang']}_weight"] for d in datasets["train"]]
        else:
            weights = [1] * len(datasets["train"])

        if "samples_per_epoch" in hparams:
            num_samples = hparams["samples_per_epoch"]
        else:
            num_samples = len(datasets["train"]) // len(hparams["train_languages"])

        # Disable shuffle cuz sampler manages this
        hparams["dataloader_options"]["shuffle"] = False
        hparams["dataloader_options"]["sampler"] = ReproducibleWeightedRandomSampler(
            weights=weights,
            num_samples=num_samples,
            replacement=num_samples >= len(datasets["train"]),
        )


    return datasets
