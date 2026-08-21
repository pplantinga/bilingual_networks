# Lost but Not Erased: Finding Traces of a Forgotten Language in Neural Speech Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-ee4c2c.svg)](https://pytorch.org/)
[![SpeechBrain](https://img.shields.io/badge/SpeechBrain-1.0.3-orange.svg)](https://speechbrain.github.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Previous behavioral and neuroimaging studies have shown that international adoptees can retain neural traces of their birth language for decades, even when adoption occurred as early as 12–18 months of age and conscious recollection is absent.

This repository investigates whether deep neural speech recognition models exhibit analogous linguistic retention. When an end-to-end Automatic Speech Recognition (ASR) system is pretrained on an initial language ($L_{pre}$) and then abruptly and completely switched to a second language ($L_{post}$), it experiences rapid behavioral attrition on $L_{pre}$. However, using **Representational Similarity Analysis (RSA)**, **Layer-wise Phoneme Probing**, and **Savings / Layer Splicing** paradigms, this project demonstrates that deep networks preserve localized phonetic and acoustic representations of their early language experience in the early-to-intermediate layers of the encoder.

---

## Key Experimental Concepts & Models

| Notation | Model Type | Description |
| :--- | :--- | :--- |
| **$L_{pre}$** | Birth Language | Initial language used during early pretraining (e.g., German / DE). |
| **$L_{post}$** | Adoption Language | Second language trained on after the switch (e.g., French / FR). |
| **$L_{ctrl}$** | Control Language | Disparate third language used for control baselines (e.g., English / EN). |
| **$M_{pre}$** | Monolingual Baseline | Model trained exclusively on $L_{pre}$. |
| **$M_{post}$** | Monolingual Baseline | Model trained exclusively on $L_{post}$. |
| **$M_a$** | Adoptee Model | Model trained on $L_{pre}$, then switched completely to $L_{post}$ ($L_{pre} \to L_{post}$). |
| **$M_{ctrl}$** | Control Model | Model trained on $L_{ctrl}$, then switched to $L_{post}$ ($L_{ctrl} \to L_{post}$). |

---

## Four Pillars of Evaluation

1. **Behavioral Language Attrition & Acquisition** ([`bilingual_transformer.py`](bilingual_transformer.py), [`figures/figure_2_language_attrition.py`](figures/figure_2_language_attrition.py)):
   - Evaluates next-token accuracy and Word Error Rate (WER) after language switching.
   - Shows that $M_a$ rapidly loses behavioral decodability on $L_{pre}$ (falling to chance) while matching monolingual $M_{post}$ performance on $L_{post}$.

2. **Layer-wise Phonemic Probing** ([`phoneme_probe_transformer.py`](phoneme_probe_transformer.py), [`figures/figure_3_phoneme_probe.py`](figures/figure_3_phoneme_probe.py)):
   - Attaches linear diagnostic probes to the CNN front-end and each of the 12 Transformer encoder layers.
   - Evaluates decodability across **Pre-phonemic** (Layers 1–4), **Phonemic** (Layers 5–8), and **Post-phonemic** (Layers 9–12) representations to locate where $L_{pre}$ phonemic representations survive.

3. **Representational Similarity Analysis (RSA)** ([`correlation_expts/generate_correlations_unzip.py`](correlation_expts/generate_correlations_unzip.py), [`figures/figure_4_correlation_scores.py`](figures/figure_4_correlation_scores.py)):
   - Computes pairwise cosine Representational Similarity Matrices (RSMs) across model checkpoints and layers.
   - Measures Spearman rank correlation over training trajectories, revealing that early encoder layers in $M_a$ retain significantly higher similarity to $M_{pre}$ than control models $M_{ctrl}$.

4. **Savings Paradigm & Layer Splicing** ([`train_transformer.py`](train_transformer.py), [`experiments/savings_partial_load.yaml`](experiments/savings_partial_load.yaml), [`figures/figure_5_savings.py`](figures/figure_5_savings.py)):
   - Tests whether early experience facilitates accelerated relearning of $L_{pre}$ (Savings effect).
   - Splices frozen encoder sub-blocks from $M_a$ into naive models via affine bridge adapters to demonstrate that early layers directly account for the relearning advantage.

---

## Repository Structure

```text
bilingual_networks/
├── data/
│   └── phonemes_{de,en,fr}.csv           # Language-specific phoneme definitions
├── experiments/                          # HyperPyYAML / SpeechBrain configuration files
│   ├── transformer.yaml                  # Monolingual / standard Conformer-Transformer ASR
│   ├── bilingual_transformer.yaml        # Language-switching & multi-decoder training
│   ├── phoneme_probe_transformer.yaml    # Hidden-layer phoneme probe training
│   ├── savings_partial_load.yaml         # Layer splicing & partial load with affine bridge
│   ├── discrete_audio_transformer.yaml   # Discrete audio token inputs
│   └── discrete_phoneme_transformer.yaml # Discrete phoneme token inputs
├── figures/                              # Scripts and data to generate publication figures
│   ├── figure_1_methods.svg              # Overview schematic of experimental methods
│   ├── figure_2_language_attrition.py    # Generates Fig 2: Attrition & acquisition curves
│   ├── figure_3_phoneme_probe.py         # Generates Fig 3: Layer-wise phoneme probing
│   ├── figure_4_correlation_scores.py    # Generates Fig 4: RSA correlation trajectories
│   ├── figure_5_savings.py               # Generates Fig 5: Savings and layer replacement
│   └── *.csv                             # Evaluation logs and summary data for figures
├── correlation_expts/
│   └── generate_correlations_unzip.py    # Computes RSMs and Spearman correlations from checkpoint archives
├── data_sb.py                            # Data pipelines, TextGrid parsing, and manifest generation
├── train_transformer.py                  # Main SpeechBrain training recipe for Transformer ASR
├── bilingual_transformer.py              # Multi-decoder training recipe for language switching
├── phoneme_probe_transformer.py          # Encoder probing recipe across hidden layers
├── discrete_audio_transformer.py         # Quantized discrete audio token ASR recipe
├── discrete_phoneme_transformer.py       # Discrete phoneme sequence ASR recipe
├── requirements.txt                      # Python dependencies
└── LICENSE                               # MIT License
```

---

## Installation

```bash
pip install -r requirements.txt
```
---

## Data Preparation

The experiments use the [Mozilla Common Voice](https://commonvoice.mozilla.org/) corpus for English (`en`), French (`fr`), and German (`de`).

1. Audio files (`.mp3` or `.wav`) should be placed in `data/<lang>/` (e.g. `data/de/`, `data/fr/`, `data/en/`).
2. Alignments are generated using the [Montreal Forced Aligner (MFA)](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) to produce `.TextGrid` files containing word and phone intervals alongside each audio file.
3. Manifests (`manifests/train_<lang>.json`, `manifests/valid_<lang>.json`, `manifests/test_<lang>.json`) and SentencePiece tokenizers are automatically built on first run by [`data_sb.py`](data_sb.py).

---

## Running Experiments

All recipes use [SpeechBrain](https://speechbrain.github.io/) with [HyperPyYAML](https://github.com/speechbrain/HyperPyYAML) configuration files.

### 1. Training Base Models ($M_{pre}$, $M_{post}$)

To train a standard Conformer/Transformer ASR model on a specific language (e.g., German):

```bash
python train_transformer.py experiments/transformer.yaml \
    --train_language="de" \
    --output_folder="results/conformer_de/seed_3420"
```

### 2. Adoptee Simulation ($M_a$: $L_{pre} \to L_{post}$)

To train the adoptee model, load a checkpoint pretrained on $L_{pre}$ (e.g., 5 epochs of German) and continue training on $L_{post}$ (French) with separate decoders:

```bash
python bilingual_transformer.py experiments/bilingual_transformer.yaml \
    --pretrained_path="results/conformer_de/seed_3420/save/CKPT+..." \
    --train_languages="['fr']" \
    --test_languages="['en', 'fr', 'de']" \
    --output_folder="results/adoptee_de2fr/seed_3420"
```

### 3. Layer-Wise Phoneme Probing

To probe the hidden representations of a trained model across all 12 encoder layers for phonetic decodability:

```bash
python phoneme_probe_transformer.py experiments/phoneme_probe_transformer.yaml \
    --pretrained_path="results/adoptee_de2fr/seed_3420/save/CKPT+..." \
    --train_language="fr" \
    --freeze_model=True \
    --output_folder="results/phoneme_probe_fr/seed_3420"
```

### 4. Representational Similarity Analysis (RSA)

To generate RSMs and Spearman correlation trajectories directly from checkpoint archives:

```bash
python correlation_expts/generate_correlations_unzip.py
```

### 5. Savings & Layer Splicing

To evaluate relearning speedup or splice specific encoder layers (e.g., layers 1–4) with affine bridge adapters:

```bash
python train_transformer.py experiments/savings_partial_load.yaml \
    --partial_load_path="results/adoptee_de2fr/seed_3420/save/CKPT+..." \
    --output_folder="results/savings_layers_1_4/seed_3420"
```

---

## Reproducing Figures

The `figures/` directory contains standalone plotting scripts and pre-extracted metric CSVs:

```bash
cd figures

# Figure 2: Language Attrition and Monolingual Matching
python figure_2_language_attrition.py

# Figure 3: Layer-wise Phonemic Decodability & Statistical Tests
python figure_3_phoneme_probe.py

# Figure 4: RSA Trajectories & Representational Convergence
python figure_4_correlation_scores.py

# Figure 5: Relearning Savings & Layer Replacement Speedups
python figure_5_savings.py
```
