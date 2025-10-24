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
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.data_pipeline import takes, provides

import psutil
import os

# Get current process ID
process = psutil.Process(os.getpid())


class SimpleDataset(torch.utils.data.Dataset):
    """Minimal

    Arguments
    ---------
    json_path: str
        Path to JSON manifest.
    """
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.entries = list(json.load(f).values())

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        sample = dict(self.entries[idx])  # copy to avoid mutation leaks
        return torch.rand(40000)

class BilingualBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Computes forward pass from wavs to phonemes and wrd"""
        batch = batch.to(self.device)
        signal, lens = batch.signal
        feats = self.hparams.compute_features(signal)
        out, _ = self.modules.model(feats)
        return out

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss between predicted and actual wrd and phonemes."""

        # Get CPU and memory usage in bytes
        cpu_usage = process.cpu_percent(interval=0.1)
        memory_info = process.memory_info()
        memory_usage_bytes = memory_info.rss  # Resident Set Size

        #print(f"CPU Usage: {cpu_usage:.2f}%")
        print(f"Memory Usage: {memory_usage_bytes / (1024 * 1024):.2f} MB")

        targets = torch.randint(size=(predictions.size(0), 251), high=256, device=self.device)

        return torch.nn.functional.cross_entropy(predictions.transpose(1, 2), targets)

def make_datasets(hparams):

    @takes("wav")
    @provides("signal")
    def audio_pipeline(wav):
        return torch.rand(40000)

    datasets = {}
    for stage in ["train", "valid", "test"]:
        datasets[stage] = sb.dataio.dataset.DynamicItemDataset.from_json(
            hparams[f"{stage}_fr_manifest"],
            dynamic_items=[audio_pipeline],
            output_keys=["signal"],
        )
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

    datasets = make_datasets(hparams)

    # Create trainer
    bilingual_brain = BilingualBrain(
        modules={"model": hparams["model"]},
        opt_class=torch.optim.Adam,
        hparams=hparams,
        run_opts=run_opts,
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
    #bilingual_brain.evaluate(
    #    datasets["test"],
    #)
