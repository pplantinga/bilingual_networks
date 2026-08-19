#!/usr/bin/env python
# coding: utf-8

# Neural Traces via RSA -- Transformer Edition
import torch
import json
import hyperpyyaml
import os
import io
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.regression import SpearmanCorrCoef
from pathlib import Path
from speechbrain.dataio import audio_io
import torchaudio
import textgrid
import torch.nn.functional as F
from collections import defaultdict
import itertools
import polars as pl


class ZipCheckpointLoader:
    """
    Reads SpeechBrain checkpoints directly from an archive zip file on NFS
    without extracting files to local disk.
    """
    def __init__(self, zip_path):
        self.zip_path = Path(zip_path)
        if not self.zip_path.exists():
            raise FileNotFoundError(f"Zip file not found at: {self.zip_path}")

    def list_checkpoints(self):
        """Finds and orders all valid SpeechBrain CKPT directories in the zip file."""
        ckpts = []
        with zipfile.ZipFile(self.zip_path, 'r') as z:
            # Look for CKPT.yaml files inside the zip
            ckpt_yamls = [f for f in z.namelist() if f.endswith("CKPT.yaml")]
            
            for yaml_path in ckpt_yamls:
                # Parent folder name, e.g., 'CKPT+2026-04-29+06-13-06+00'
                ckpt_dir = str(Path(yaml_path).parent)
                
                # Parse metadata inside CKPT.yaml
                with z.open(yaml_path) as f:
                    yaml_content = f.read().decode('utf-8')
                    meta = hyperpyyaml.load_hyperpyyaml(yaml_content)
                
                ckpts.append({
                    "dir": ckpt_dir,
                    "unixtime": meta.get("unixtime", 0),
                    "meta": meta
                })

        # Sort chronologically as SpeechBrain's ckpt_recency does
        ckpts.sort(key=lambda x: x["unixtime"])
        return ckpts

    def load_checkpoint_into_hparams(self, hparams, ckpt_info):
        """
        Loads all matching `.ckpt` files inside the checkpoint folder into
        their corresponding keys in `hparams`.
        """
        ckpt_dir = ckpt_info["dir"]

        with zipfile.ZipFile(self.zip_path, 'r') as z:
            # Find all .ckpt files in this specific checkpoint folder
            ckpt_files = [f for f in z.namelist() if f.startswith(ckpt_dir) and f.endswith(".ckpt")]

            for ckpt_file in ckpt_files:
                # Extract key name e.g. "normalizer.ckpt" -> "normalizer", "model.ckpt" -> "model"
                file_stem = Path(ckpt_file).stem

                # Check for possible hparams key aliases
                # (e.g. file is normalizer.ckpt, but hparams key is "normalize" or "normalizer")
                target_key = None
                if file_stem in hparams:
                    target_key = file_stem
                elif file_stem == "normalizer" and "normalize" in hparams:
                    target_key = "normalize"

                if target_key and target_key in hparams:
                    with z.open(ckpt_file) as f:
                        buffer = io.BytesIO(f.read())
                        state_dict = torch.load(buffer, map_location='cpu')

                        # Load state into the SpeechBrain module/normalizer
                        if hasattr(hparams[target_key], "_load_statistics_dict"):
                            hparams[target_key]._load_statistics_dict(state_dict)
                        elif hasattr(hparams[target_key], 'load_state_dict'):
                            hparams[target_key].load_state_dict(state_dict)

        # Set normalizer to eval mode / freeze update mode so it uses loaded stats
        if "normalize" in hparams and hasattr(hparams["normalize"], "update"):
            hparams["normalize"].update = False

@torch.no_grad()
def make_rsm(representations):
    """Generates a Representational Similarity Matrix (RSM)"""
    representations_norm = torch.nn.functional.normalize(representations, p=2, dim=1)
    rsm = representations_norm @ representations_norm.T
    return rsm

def extract_tril(matrix):
    """Extract the bottom triangle (excluding the diagonal) into a vector"""
    indices = torch.tril_indices(matrix.size(0), matrix.size(1), offset=-1)
    tril_vector = matrix[indices[0], indices[1]]
    return tril_vector

def compute_spearman_rank(rsm1, rsm2):
    """Compute spearman rank based on EVERY PAIR of samples"""
    tril1 = extract_tril(rsm1)
    tril2 = extract_tril(rsm2)

    spearman_fn = SpearmanCorrCoef()
    spearman_val = spearman_fn(tril1, tril2)
    return spearman_val

def load_hparams_only(hparams_file, model_stem):
    """Loads the HyperPyYAML configuration structure without triggering recovery."""
    slurm_tmpdir = os.getenv("SLURM_TMPDIR", "/tmp")
    overrides = {
        "output_folder": os.path.join(slurm_tmpdir, "bilingual_networks/results", model_stem),
        "data_folder": os.path.join(slurm_tmpdir, "bilingual_networks/data/"),
        "Transformer": {"output_hidden_states": True},
    }
    with open(hparams_file) as f:
        hparams = hyperpyyaml.load_hyperpyyaml(f, overrides=overrides)

    return hparams

def split_name(name):
    parts = name.split("_")
    seed = 1 if "take" not in parts[-1] else name[-1]
    model = "2".join([p[:2] for p in parts if not "take" in p])
    return model, seed

def generate_all_layer_reps_from_model(hparams, test_data, device="cuda"):
    compute_feats = hparams["compute_features"].to(device)
    normalize = hparams["normalize"].to(device)
    cnn = hparams["CNN"].to(device)
    transformer = hparams["Transformer"].to(device)

    layer_reps = defaultdict(list)

    for utt_id, sample in test_data.items():
        audio_path = Path(hparams["data_folder"]) / Path(sample["path"]).with_suffix(".mp3")
        audio, sr = audio_io.load(audio_path, always_2d=True)
        audio_len = torch.ones((1,), device=device)

        with torch.no_grad():
            audio = torchaudio.functional.resample(audio.to(device), orig_freq=sr, new_freq=16000)
            features = normalize(compute_feats(audio), audio_len, epoch=15)
            enc_out, hidden_states = transformer.encode(cnn(features), audio_len)

            for layer_idx, hidden in enumerate(hidden_states):
                rep = hidden.squeeze(0).mean(dim=0)
                layer_reps[layer_idx].append(rep)

    return {k: torch.stack(v, dim=0) for k, v in layer_reps.items()}


def create_correlation_file(
    data_subset,
    hparams_file,
    ckpt_prefix,
    mono_results_list,
    bi_results_list,
    output_csv,
    nfs_base_path,
    layers,
    mono_ckpt_ids,
    bi_ckpt_ids,
):
    rsms = {}
    
    # Process Monolingual Checkpoints
    for stem in mono_results_list:
        model_stem = ckpt_prefix + stem
        zip_path = Path(nfs_base_path) / f"{model_stem}.zip"
        print(f"Generating RSMs directly from ZIP: {zip_path}")
        
        loader = ZipCheckpointLoader(zip_path)
        ckpts = loader.list_checkpoints()
        
        hparams = load_hparams_only(hparams_file, model_stem)

        if len(ckpts) <= max(mono_ckpt_ids):
            print(f"Warning: requested checkpoint IDs exceed available checkpoints in {stem}")

        for i, ckpt_info in enumerate(ckpts):
            if i not in mono_ckpt_ids:
                continue
            
            # Load weights into Transformer from memory
            loader.load_checkpoint_into_hparams(hparams, ckpt_info)
            
            reps = generate_all_layer_reps_from_model(hparams, data_subset)
            rsms.update({f"{stem}_ckpt{i}_layer{k}": make_rsm(reps[k]) for k in reps})

    # Calculate Monolingual Correlations
    results = []
    for a, b in itertools.combinations(mono_results_list, 2):
        if a.startswith("de") and b.startswith("de"):
            continue
        for k in range(layers):
            for i in set(mono_ckpt_ids):
                name_a = f"{a}_ckpt{i}_layer{k}"
                name_b = f"{b}_ckpt{i}_layer{k}"
                name = f"{a}_v_{b}_ckpt{i}_layer{k}"
                model_a, seed_a = split_name(a)
                model_b, seed_b = split_name(b)
                
                if name_a in rsms and name_b in rsms:
                    score = compute_spearman_rank(rsms[name_a], rsms[name_b]).cpu().detach().item()
                    results.append({
                        "epoch": i,
                        "layer": k,
                        "model_a": model_a,
                        "seed_a": seed_a,
                        "model_b": model_b,
                        "seed_b": seed_b,
                        "score": round(score, 4),
                    })

    # Process Bilingual Checkpoints
    for stem in bi_results_list:
        model_stem = ckpt_prefix + stem
        zip_path = Path(nfs_base_path) / f"{model_stem}.zip"
        print(f"Generating RSMs directly from ZIP: {zip_path}")
        
        loader = ZipCheckpointLoader(zip_path)
        ckpts = loader.list_checkpoints()
        
        hparams = load_hparams_only(hparams_file, model_stem)

        if len(ckpts) <= max(bi_ckpt_ids):
            print(f"Warning: requested checkpoint IDs exceed available checkpoints in {stem}")

        for i, ckpt_info in enumerate(ckpts):
            if i not in bi_ckpt_ids:
                continue
            
            loader.load_checkpoint_into_hparams(hparams, ckpt_info)
            
            reps = generate_all_layer_reps_from_model(hparams, data_subset)
            rsms.update({f"{stem}_ckpt{i}_layer{k}": make_rsm(reps[k]) for k in reps})

    # Calculate Bilingual vs Monolingual Correlations
    for k in range(layers):
        for i, j in zip(bi_ckpt_ids, mono_ckpt_ids):
            for target_a in bi_results_list:
                for target_b in mono_results_list:
                    name_a = f"{target_a}_ckpt{i}_layer{k}"
                    name_b = f"{target_b}_ckpt{j}_layer{k}"
                    model_a, seed_a = split_name(target_a)
                    model_b, seed_b = split_name(target_b)
                    
                    if name_a in rsms and name_b in rsms:
                        score = compute_spearman_rank(rsms[name_a], rsms[name_b]).cpu().detach().item()
                        results.append({
                            "epoch": i,
                            "layer": k,
                            "model_a": model_a,
                            "seed_a": seed_a,
                            "model_b": model_b,
                            "seed_b": seed_b,
                            "score": round(score, 4),
                        })

    df = pl.DataFrame(results)
    df.write_csv(output_csv)


if __name__ == "__main__":
    lang = "de"
    with open(f"../manifests/test_{lang}.json") as f:
        manifest = json.load(f)
    subset = dict(list(manifest.items())[:500])
    print(f"Size of {lang} subset:", len(subset))

    lang2 = "fr"
    lang3 = "en"
    output_file = f"correlation_scores_asr_medium_{lang}2{lang2}_same_steps_long.csv"

    # Set NFS path where zip files reside
    NFS_BASE_PATH = "/home/plantinp/scratch/bilingual_networks/transformer_warmup/"

    mono_results_list = ["de25_take4", "de25_take6", "fr25_take4", "fr25_take6"]
    bi_results_list = ["de5_fr25_take2", "de5_fr25_take3", "de5_fr25_take6", "de5_fr25_take7", "de5_fr25_take8"]
    bi_results_list += ["en5_fr25_take2", "en5_fr25_take3", "en5_fr25_take6", "en5_fr25_take7", "en5_fr25_take8", "en5_fr25_take9", "en5_fr25_take10"]


    create_correlation_file(
        data_subset=subset,
        hparams_file="../experiments/transformer.yaml",
        ckpt_prefix="conformer_medium_",
        mono_results_list=mono_results_list,
        bi_results_list=bi_results_list,
        output_csv=output_file,
        nfs_base_path=NFS_BASE_PATH,
        layers=13,
        mono_ckpt_ids=list(range(4, 25)) + [24] * 5,
        bi_ckpt_ids=list(range(4, 30)),
    )
    print(f"Created {output_file}")
