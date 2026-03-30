# Unsupervised Cross-Lingual Machine Translation for Low-Resource Indian Languages

<p align="center">
  <b>Vedant Maheshwari* &nbsp;·&nbsp; Naksh Singh* &nbsp;·&nbsp; Premjith B.†</b><br>
  School of Artificial Intelligence, Amrita Vishwa Vidyapeetham, Coimbatore, India<br>
  <i>† Corresponding Author &nbsp;·&nbsp; * Equal Contribution</i>
</p>

<p align="center">
  <a href="https://huggingface.co/nakshhh/Machine_Translation_Models"><img src="https://img.shields.io/badge/🤗%20Models-HuggingFace-yellow" alt="Models"></a>
  <a href="https://huggingface.co/nakshhh/Data_Machine_Translation"><img src="https://img.shields.io/badge/🤗%20Dataset-HuggingFace-blue" alt="Dataset"></a>
  <img src="https://img.shields.io/badge/Python-3.9%2B-green" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-red" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-Apache%202.0-lightgrey" alt="License">
</p>

---

## Overview

Parallel corpora for most Indian languages are either unavailable or extremely scarce, making conventional supervised neural machine translation (NMT) infeasible. This project proposes a **two-phase semantic-pivot framework** for **unsupervised cross-lingual NMT** that operates with **zero human-annotated parallel data**.

The central idea is to cleanly decouple the problem into two stages:

1. **Validate and stabilise** a shared multilingual semantic space using frozen IndicBERT embeddings and a token-level denoising autoencoder (DAE) trained on controlled corruptions of monolingual text.
2. **Generate fluent translations** by training a shared contextual Transformer encoder and language-specific mT5 decoders on a synthetic parallel corpus produced via back-translation with Indic–Indic 1B.

The system covers **7 typologically diverse Indian languages** spanning 4 scripts and 2 language families (Indo-Aryan and Dravidian), achieving large improvements over zero-shot baselines without a single human-labelled parallel sentence.

---

## Table of Contents

- [Key Contributions](#key-contributions)
- [Architecture](#architecture)
- [Supported Languages](#supported-languages)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Pretrained Models](#pretrained-models)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Figures](#figures)
- [Limitations](#limitations)
- [Citation](#citation)

---

## Key Contributions

- A **two-phase semantic-pivot framework** that eliminates the need for parallel corpora in cross-lingual NMT.
- A **controlled multi-strategy corruption pipeline** (adjacent word-order swap + character/spacing noise via AugLy) producing `(corrupted, clean)` monolingual pairs for DAE training, with a safety filter that rejects over-fragmented outputs.
- A **token-level denoising autoencoder** with a combined reconstruction + invariance loss that projects IndicBERT embeddings onto a noise-invariant, language-agnostic semantic manifold.
- **Synthetic parallel corpus generation** via back-translation using [ai4bharat/indictrans2-indic-indic-1B](https://huggingface.co/ai4bharat/indictrans2-indic-indic-1B), covering all 42 directed language pairs (~4.3M sentence pairs total).
- A **shared contextual Transformer encoder** + **language-specific mT5-small decoders** architecture with a staged freezing strategy that prevents semantic drift.
- Evaluation on both a held-out synthetic test set and the **IN22-Gen benchmark**, with ablation confirming the necessity of the contextual encoder.

---

## Architecture

### Phase 1 — Semantic Representation Learning

```
Monolingual Text (7 languages, ~723K sentences total)
         │
         ▼
┌────────────────────────────────────────────────────┐
│                 Corruption Pipeline                │
│  Step 1: Randomly swap one pair of adjacent        │
│          tokens per sentence                        │
│  Step 2: Apply 1–2 AugLy augmenters at low         │
│          probability (char noise, word split/merge, │
│          case change, punctuation insertion)        │
│  Step 3: Reject if >35% of tokens are single-char  │
│  Output: (language, clean, corrupted) CSV          │
└────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────┐
│      Frozen IndicBERT  (ai4bharat/indic-bert)      │
│  Applied independently to clean and corrupted      │
│  H_clean, H_corrupt  ∈  ℝ^{T × 768}               │
└────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────┐
│          Token-Level Denoising Autoencoder         │
│                                                    │
│  Encoder f_θ:  768 → 512 (ReLU) → 256             │
│  Decoder g_φ:  256 → 512 (ReLU) → 768             │
│  Applied per-token independently (no seq. model)  │
│                                                    │
│  L_recon = (1/T) Σ ‖ĥ_t − h_t‖²                  │
│  L_inv   = (1/T) Σ ‖f_θ(h_t) − f_θ(h̃_t)‖²       │
│  L_DAE   = L_recon + 0.5 · L_inv                  │
└────────────────────────────────────────────────────┘
         │   (semantic manifold validated)
         ▼
┌────────────────────────────────────────────────────┐
│       Back-Translation via Indic–Indic 1B          │
│   Greedy decoding (beam=1), batch size 32          │
│   42 directed pairs → ~4.3M synthetic pairs        │
└────────────────────────────────────────────────────┘
```

### Phase 2 — Semantic-to-Text Generation

```
Source Sentence
      │
      ▼  [FROZEN throughout Phase 2]
 IndicBERT  →  H ∈ ℝ^{T × 768}
      │
      ▼  [FROZEN throughout Phase 2]
 DAE encoder f_θ  →  Z ∈ ℝ^{T × 256}
      │
      ▼  [TRAINED in Stage 1]
 Shared Contextual Transformer Encoder
 4 layers · 4 heads · d_model = 256
 C = TransformerEnc(Z)  ∈  ℝ^{T × 256}
      │
      ▼  [TRAINED in Stage 1]
 Projection Layer   W ∈ ℝ^{512 × 256}
 C̃ = WC + b  ∈  ℝ^{T × 512}
      │
      ▼  [TRAINED in Stage 2, one per target language]
 Language-Specific mT5-small Decoder
 (native mT5 encoder removed; C̃ fed via cross-attention)
 p(y^(j) | C̃) — autoregressive NLL with teacher forcing
      │
      ▼
 Target Language Text
```

**Training Strategy:**

| Stage | What trains | What is frozen |
|---|---|---|
| Stage 1 | Shared contextual encoder + projection layer | IndicBERT, DAE |
| Stage 2 (×7) | Language-specific decoder (one per language) | IndicBERT, DAE, shared encoder, projection |

The staged freezing strategy ensures the semantic manifold is fixed before generation begins, preventing catastrophic forgetting and semantic drift.

---

## Supported Languages

| Code | Language | Script | Family | Monolingual Sentences |
|---|---|---|---|---|
| `asm_Beng` | Assamese | Bengali | Indo-Aryan | 99,490 |
| `ben_Beng` | Bengali | Bengali | Indo-Aryan | 107,870 |
| `guj_Gujr` | Gujarati | Gujarati | Indo-Aryan | 105,694 |
| `mal_Mlym` | Malayalam | Malayalam | Dravidian | 98,031 |
| `mar_Deva` | Marathi | Devanagari | Indo-Aryan | 117,275 |
| `tam_Taml` | Tamil | Tamil | Dravidian | 97,767 |
| `ory_Orya` | Odia | Odia | Indo-Aryan | 97,402 |

All English alignments from BPCC are intentionally discarded — each language corpus is treated as a purely monolingual resource.

---

## Results

### Held-Out Synthetic Test Set — Proposed vs. Zero-Shot Baseline

| Source → Target | Baseline BLEU | **Proposed BLEU** | **ChrF++** | **IndicBERTScore** |
|---|---|---|---|---|
| Marathi → Gujarati | 0.08 | **25.92** | **51.46** | **0.95** |
| Assamese → Bengali | 0.03 | **20.96** | **52.91** | **0.95** |
| Odia → Assamese | 0.05 | **21.11** | **50.21** | **0.95** |
| Bengali → Assamese | 0.02 | **21.06** | **50.97** | **0.94** |
| Gujarati → Odia | 0.03 | **18.48** | **45.10** | **0.93** |
| Tamil → Malayalam | 0.00 | **11.32** | **41.61** | **0.95** |
| Malayalam → Odia | 0.05 | **8.52** | **34.54** | **0.90** |

The zero-shot mT5-small baseline achieves BLEU < 0.1 across all 42 directed pairs. The proposed framework achieves **BLEU 8–26** and **IndicBERTScore consistently above 0.87** across all pairs.

### IN22-Gen Benchmark (Out-of-Domain)

| Source | Proposed BLEU | IndicTrans2 BLEU |
|---|---|---|
| Marathi → Gujarati | 6.80 | 17.97 |
| Assamese → Bengali | 4.55 | 10.25 |
| Odia → Bengali | 3.22 | 9.46 |
| Tamil → Marathi | 4.24 | 13.45 |

Our unsupervised system trained on 4.3M synthetic pairs reaches **33–50% of IndicTrans2 performance** (which was trained on 230M human-annotated bitext pairs). The strong semantic preservation (IndicBERTScore ~0.88–0.91) across all out-of-domain pairs validates the robustness of the semantic pivot approach.

### Ablation — Without Contextual Encoder

Removing the shared contextual Transformer encoder and feeding DAE latents directly to the decoder causes:

| Metric | With Encoder | Without Encoder | Change |
|---|---|---|---|
| BLEU (asm→ben) | 20.96 | 0.05 | **−98%** |
| ChrF++ (asm→ben) | 52.91 | 9.29 | **−65%** |
| IndicBERTScore (asm→ben) | 0.95 | 0.73 | −23% |

This confirms that token-independent DAE representations are semantically aligned but cannot produce fluent sentences without sentence-level dependency modelling from the shared Transformer encoder.

---

## Repository Structure

```
MachineTranslation/
│
├── codefile/
│   ├── unsup_data.py                    # Loads monolingual BPCC TSVs and produces
│   │                                    # a unified (lang, text) CSV for all 7 languages
│   │
│   ├── Correct_corrupt.py               # Corruption pipeline: word-order swap +
│   │                                    # AugLy noise → (language, clean, corrupted) CSV
│   │
│   ├── analysis.py                      # Standalone corruption analysis script:
│   │                                    # stats, script detection, length vs corruption,
│   │                                    # saves all plots to figs/ (no training code)
│   │
│   ├── dae-nlp-15f39b.ipynb             # Phase 1: DAE training
│   │                                    # Encodes with frozen IndicBERT, trains DAE
│   │                                    # with L_recon + L_inv loss, saves dae_model.pt
│   │
│   ├── parallel-corpus-generator.ipynb  # Phase 1: Back-translation via Indic–Indic 1B
│   │                                    # Generates ~4.3M synthetic pairs across
│   │                                    # all 42 directed language pairs
│   │
│   ├── nlp-project-part-2.ipynb         # Phase 2: Full pipeline training
│   │                                    # Defines CustomMTPipeline; Stage 1 trains
│   │                                    # shared encoder; Stage 2 fine-tunes per-language
│   │                                    # mT5-small decoders
│   │
│   ├── fork-of-evaluation-script.ipynb  # Evaluation: BLEU / ChrF++ / IndicBERTScore
│   │                                    # for proposed model, zero-shot baseline,
│   │                                    # and IndicTrans2 SOTA comparison
│   │
│   └── ablation.ipynb                   # Ablation: removes contextual encoder,
│                                        # evaluates DAE → projector → decoder only
│
├── figs/                                # Corruption analysis figures (data inspection only)
│   ├── corruption_buckets_bar.png
│   ├── corruption_buckets_pie.png
│   ├── length_vs_corruption.png
│   ├── length_group_avg_corruption.png
│   ├── augmenters.png
│   ├── buckets_bar.png
│   └── buckets_pie.png
│
├── results/                             # Per-language evaluation CSVs
│   ├── final_avg_results_*.csv          # Proposed model scores by source language
│   ├── baseline_results_*.csv           # Zero-shot mT5-small baseline scores
│   ├── indictrans2_results_*.csv        # IndicTrans2 SOTA comparison scores
│   └── ablation_no_context_results_*.csv # Ablation scores (no contextual encoder)
│
└── README.md
```

> **Note on `figs/`:** All figures are outputs of `codefile/analysis.py`, a standalone data analysis script that characterises the corruption pipeline. They are not outputs of any model training stage — there is no model code associated with these plots.

---

## Dataset

**🤗 [nakshhh/Data_Machine_Translation](https://huggingface.co/nakshhh/Data_Machine_Translation)**

The dataset contains three components:

**1. Monolingual BPCC data** — Raw monolingual sentences for 7 Indian languages extracted from the [BPCC corpus](https://github.com/AI4Bharat/IndicTrans) (IndicCorp / IndicNLP Suite). All English-side alignments are discarded. Input to both the corruption pipeline and the back-translation step.

**2. Corrupted-clean pairs** — Produced by `Correct_corrupt.py`. Columns: `language`, `correct`, `corrupted`. Analysis of the corpus shows ~30% of sentences receive AugLy augmentation, median corruption of 0% (conservative by design), with the 90th percentile below 20% token modification. Pearson correlation between sentence length and corruption ratio is slightly negative, confirming no systematic length bias.

**3. Synthetic parallel corpus** — ~4.3M `(source, target)` sentence pairs covering all 42 directed language pairs, generated via greedy back-translation with Indic–Indic 1B. One CSV file per directed pair (`{src_lang}_to_{tgt_lang}.csv`). Used to train the shared encoder (Stage 1) and all language-specific decoders (Stage 2).

---

## Pretrained Models

**🤗 [nakshhh/Machine_Translation_Models](https://huggingface.co/nakshhh/Machine_Translation_Models)**

| File | Description | Size |
|---|---|---|
| `shared_context_encoder.pt` | Shared Transformer encoder (4 layers, 4 heads, d=256) | 21 MB |
| `shared_projector.pt` | Linear projection layer (256 → 512) | 528 KB |
| `decoder_asm_Beng.pt` | mT5-small decoder — Assamese | 1.13 GB |
| `decoder_ben_Beng.pt` | mT5-small decoder — Bengali | 1.13 GB |
| `decoder_guj_Gujr.pt` | mT5-small decoder — Gujarati | *(pending re-upload)* |
| `decoder_mal_Mlym.pt` | mT5-small decoder — Malayalam | 1.13 GB |
| `decoder_mar_Deva.pt` | mT5-small decoder — Marathi | *(pending re-upload)* |
| `decoder_ory_Orya.pt` | mT5-small decoder — Odia | 652 MB |
| `decoder_tam_Taml.pt` | mT5-small decoder — Tamil | 1.13 GB |

**Download all models:**

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="nakshhh/Machine_Translation_Models",
    local_dir="./models"
)
```

**Download individual files:**

```python
from huggingface_hub import hf_hub_download

# Shared encoder (needed for all translations)
hf_hub_download(
    repo_id="nakshhh/Machine_Translation_Models",
    filename="models/shared_context_encoder.pt",
    local_dir="./models"
)
hf_hub_download(
    repo_id="nakshhh/Machine_Translation_Models",
    filename="models/shared_projector.pt",
    local_dir="./models"
)

# Language-specific decoder (one per target language, e.g. Tamil)
hf_hub_download(
    repo_id="nakshhh/Machine_Translation_Models",
    filename="models/decoder_tam_Taml.pt",
    local_dir="./models"
)
```

---

## Installation

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0 with CUDA (experiments used NVIDIA Tesla P100, 16 GB VRAM)
- HuggingFace Transformers ≥ 4.38
- sentencepiece, sacrebleu, augly, datasets, huggingface_hub

### Setup

```bash
git clone https://github.com/nakshsingh18/MachineTranslation.git
cd MachineTranslation

# PyTorch with CUDA 11.8 (adjust for your environment)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Core dependencies
pip install transformers sentencepiece sacrebleu datasets \
            huggingface_hub tqdm pandas numpy matplotlib

# AugLy — required for the corruption pipeline
pip install augly

# IndicTransToolkit — required for back-translation and IndicTrans2 evaluation
git clone https://github.com/VarunGumma/IndicTransToolkit.git
cd IndicTransToolkit && pip install --editable ./ && cd ..
```

### HuggingFace Authentication

All notebooks load models from HuggingFace. Set your token before running:

```bash
export HF_TOKEN=your_hf_token_here
```

Or inside a notebook:
```python
from huggingface_hub import login
login(token="your_hf_token_here")
```

---

## How to Run

The full pipeline consists of **five sequential steps** followed by optional evaluation and ablation. All notebooks were developed and run on **Kaggle (NVIDIA Tesla P100, 16 GB VRAM, 30 GB RAM)**. Kaggle-specific paths (`/kaggle/input/`, `/kaggle/working/`) appear inside the notebooks and should be replaced with your local paths.

---

### Step 1 — Build Monolingual Dataset

**Script:** `codefile/unsup_data.py`

Loads raw BPCC `.tsv` files for each of the 7 languages and merges them into a single unified CSV with columns `lang` and `text`. The `tgt` column (Indian language side) is kept; the English `src` is discarded.

```bash
# Place BPCC .tsv files in the same directory, then:
python codefile/unsup_data.py
# Output: unsup_5lang_full.csv
```

Expected TSV columns: `src_lang`, `tgt_lang`, `src`, `tgt`. The BPCC data can be downloaded from [ai4bharat/indic-corpus](https://huggingface.co/datasets/ai4bharat/indic-corpus).

---

### Step 2 — Generate Corrupted-Clean Pairs

**Script:** `codefile/Correct_corrupt.py`

Applies a two-step corruption strategy to every sentence:

1. **Word-order swap** — randomly swap one adjacent token pair per sentence.
2. **AugLy noise** — apply 1 or 2 randomly selected augmenters (40% chance of 2, 60% of 1).

A safety filter rejects any output where > 35% of tokens are single characters (prevents character soup on Indic scripts).

```bash
python codefile/Correct_corrupt.py
# Input:  unsup_5lang_full.csv
# Output: pairs_lang_correct_corrupted_jumbled.csv  (columns: language, correct, corrupted)
```

**AugLy augmenters used:**

| Augmenter | Probability |
|---|---|
| `ReplaceSimilarChars` | 0.20 |
| `ReplaceSimilarUnicodeChars` | 0.20 |
| `SplitWords` | 0.08 |
| `MergeWords` | 0.08 |
| `ChangeCase` | 0.08 |
| `InsertWhitespaceChars` | 0.02 |
| `InsertPunctuationChars` | 0.05 |

---

### Step 3 — Train the Denoising Autoencoder

**Notebook:** `codefile/dae-nlp-15f39b.ipynb`

Loads `pairs_lang_correct_corrupted_jumbled.csv`, encodes both clean and corrupted sentences through **frozen IndicBERT**, and trains the DAE on the resulting embedding pairs.

**Model classes defined in the notebook:**

```python
class IndicBert(nn.Module):
    """Frozen wrapper around ai4bharat/indic-bert.
       Returns last_hidden_state: (B, T, 768). No gradients."""

class DAE(nn.Module):
    """Token-level denoising autoencoder.
       Encoder: Linear(768→512, ReLU) → Linear(512→256)
       Decoder: Linear(256→512, ReLU) → Linear(512→768)
       Applied per-token; no sequence-level interaction."""
```

**Training hyperparameters:**

| Parameter | Value |
|---|---|
| Embedding dim (IndicBERT output) | 768 |
| Latent dim (DAE bottleneck) | 256 |
| Batch size | 64 |
| Learning rate | 1e-3 (Adam) |
| Invariance weight α | 0.5 |
| Epochs | 10 |
| Max sequence length | 64 tokens |

**Loss:**
```
L_DAE   = L_recon + α · L_inv

L_recon = (1/T) Σ ‖ĥ_t − h_t‖²              MSE: reconstructed vs clean embeddings
L_inv   = (1/T) Σ ‖f_θ(h_t) − f_θ(h̃_t)‖²   Align latent reps of clean vs corrupted
```

**Output:** `dae_model.pt`

---

### Step 4 — Generate Synthetic Parallel Corpus

**Notebook:** `codefile/parallel-corpus-generator.ipynb`

Translates clean monolingual sentences across all 42 directed language pairs using `ai4bharat/indictrans2-indic-indic-1B`. Uses **greedy decoding** (`num_beams=1`, `do_sample=False`) for reproducibility.

**Configuration block (top of notebook):**

```python
MODEL_NAME  = "ai4bharat/indictrans2-indic-indic-1B"
BATCH_SIZE  = 32
MAX_LENGTH  = 128
LANGS       = ["mar_Deva", "ben_Beng", "guj_Gujr",
               "asm_Beng", "ory_Orya", "tam_Taml", "mal_Mlym"]
```

The notebook uses `IndicProcessor` for pre/post-processing and is resume-safe — it skips pairs whose output CSV already exists.

**Output:** one CSV per directed pair in `parallel_corpus/`:
```
mar_Deva_to_ben_Beng.csv     # columns: mar_Deva, ben_Beng
mar_Deva_to_guj_Gujr.csv
...  (42 files total, ~4.3M sentence pairs)
```

---

### Step 5 — Train Shared Encoder + Language-Specific Decoders

**Notebook:** `codefile/nlp-project-part-2.ipynb`

This notebook defines the full `CustomMTPipeline` and runs the two-stage training.

**Full model architecture:**

```python
class CustomMTPipeline(nn.Module):
    # A. IndicBERT backbone — FROZEN
    self.indic = AutoModel.from_pretrained("ai4bharat/indic-bert")

    # B. Pre-trained DAE — FROZEN, loaded from dae_model.pt
    self.dae = DAE(d_in=768, d_latent=256)

    # C. Shared contextual Transformer encoder — trained in Stage 1
    #    Input: DAE latent Z ∈ ℝ^{T×256}
    self.context_encoder = ContextualEncoder(d_model=256, nhead=4, num_layers=4)

    # D. Projection to mT5 embedding space — trained in Stage 1
    self.projector = nn.Linear(256, 512)

    # E. mT5-small decoder stack — trained in Stage 2 (one per language)
    #    Native mT5 encoder is deleted; C̃ is injected via cross-attention
    self.t5 = T5ForConditionalGeneration.from_pretrained("google/mt5-small")
    del self.t5.encoder
```

**Stage 1 — Shared Encoder Training:**

| Parameter | Value |
|---|---|
| Encoder layers / heads | 4 / 4 |
| Hidden dim | 256 |
| Batch size | 32 |
| Learning rate | 3e-4 (AdamW) |
| Epochs | 2 |
| Samples per language pair | 200,000 |
| Frozen | IndicBERT, DAE |

**Stage 2 — Language-Specific Decoder Fine-Tuning:**

| Parameter | Value |
|---|---|
| Decoder | mT5-small decoder stack |
| Layers / Heads / d_model | 8 / 6 / 512 |
| Dropout | 0.1 |
| Batch size | 32 |
| Learning rate | 3e-4 (AdamW) |
| Epochs | 3 per language |
| Decoding (inference) | Greedy (beam=1) |
| Frozen | IndicBERT, DAE, shared encoder, projector |
| Objective | Autoregressive NLL with teacher forcing |

Training was run language-by-language on Kaggle P100 GPUs with checkpoint-based resumption between sessions. The notebook contains `RESUME_FROM_CHECKPOINT` and `already_done()` helpers for this purpose.

**Output:** `shared_context_encoder.pt`, `shared_projector.pt`, `decoder_{lang_code}.pt` per language.

---

### Step 6 — Evaluate

**Notebook:** `codefile/fork-of-evaluation-script.ipynb`

Evaluates three systems on the same test set (held-out synthetic pairs and IN22-Gen benchmark):

1. **Proposed model** — Loads shared encoder + projector + target decoder, runs inference.
2. **Zero-shot baseline** — Runs `google/mt5-small` directly without any fine-tuning.
3. **IndicTrans2 SOTA** — Runs `ai4bharat/indictrans2-indic-indic-1B` for upper-bound comparison.

**Configuration per evaluation block (adjust per language):**

```python
TARGET_LANG      = "ta"        # 2-letter ISO code for the target language
DECODER_WEIGHTS  = "/path/to/models/decoder_tam_Taml.pt"
SHARED_ENC_PATH  = "/path/to/models/shared_context_encoder.pt"
SHARED_PROJ_PATH = "/path/to/models/shared_projector.pt"
BATCH_SIZE       = 16
```

**Language code mapping used in the notebook:**

| ISO-2 | IndicTrans2 Code |
|---|---|
| `as` | `asm_Beng` |
| `bn` | `ben_Beng` |
| `gu` | `guj_Gujr` |
| `ml` | `mal_Mlym` |
| `mr` | `mar_Deva` |
| `or` | `ory_Orya` |
| `ta` | `tam_Taml` |

**Metrics computed:**

| Metric | What it measures |
|---|---|
| **BLEU** (sacrebleu, 2-gram) | Lexical n-gram precision |
| **ChrF++** (sacrebleu, n=4) | Character-level F-score; robust to morphological variation |
| **IndicBERTScore** | Cosine similarity of IndicBERT token embeddings; semantic fidelity |

Results are saved to `results/` as per-language CSVs (columns: `Source Language`, `BLEU`, `ChrF++`, `IndicBERT`).

---

### Step 7 — Ablation (Optional)

**Notebook:** `codefile/ablation.ipynb`

Tests the pipeline **without the contextual encoder** — DAE latent representations `Z ∈ ℝ^{T×256}` are projected directly to the decoder embedding space (`256 → 512`) without any shared Transformer encoding. This isolates the contribution of sentence-level dependency modelling to the final translation quality.

---

### Corruption Data Analysis (Standalone, No Training)

**Script:** `codefile/analysis.py`

Characterises the `pairs_lang_correct_corrupted_jumbled.csv` output. Computes corruption intensity statistics, sentence-length correlations, token explosion rate, and script distribution per language. Generates all plots saved in `figs/`. **This script has no role in model training or evaluation.**

```bash
# Edit INPUT_FILE at the top of the script, then:
python codefile/analysis.py
```

---

## Figures

The `figs/` directory contains exploratory data analysis plots produced by `codefile/analysis.py` to validate the corruption pipeline design. They were not used in model training.

| File | Description |
|---|---|
| `corruption_buckets_bar.png` | Bar chart: distribution of corruption intensity across 5 buckets (0%, 0–5%, 5–10%, 10–20%, 20–25%) |
| `corruption_buckets_pie.png` | Pie chart of the same distribution |
| `length_vs_corruption.png` | Scatter plot: sentence length (tokens) vs. corruption ratio |
| `length_group_avg_corruption.png` | Bar chart: average corruption ratio by sentence-length group |
| `augmenters.png` | Breakdown of AugLy augmenter usage |
| `buckets_bar.png` / `buckets_pie.png` | Corruption distribution before capping ratios to [0, 1] |

---

## Limitations

- **Synthetic data bias:** All parallel training data is generated by Indic–Indic 1B, inheriting its errors and style. This contributes to the benchmark performance gap (BLEU 2–7 vs. 6–18 for IndicTrans2 on IN22-Gen).
- **Frozen IndicBERT:** Preserves pre-trained multilingual alignment but prevents task-specific adaptation.
- **Language coverage:** Only 7 of 22 scheduled Indian languages; decoder checkpoints for Gujarati and Marathi are currently pending re-upload.
- **Family divergence:** Dravidian languages (Tamil, Malayalam) achieve lower BLEU than Indo-Aryan languages, reflecting larger structural divergence.
- **Scale gap:** IndicTrans2 was trained on 230M bitext pairs; this system uses 4.3M synthetic pairs, explaining the 33–50% relative performance.

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{maheshwari-singh-2025-unsupervised,
  title     = {Unsupervised Cross-Lingual Machine Translation for Low-Resource Indian Languages},
  author    = {Maheshwari, Vedant and Singh, Naksh and B., Premjith},
  booktitle = {Proceedings of the Annual Meeting of the Association for Computational Linguistics},
  year      = {2025},
  address   = {Amrita Vishwa Vidyapeetham, Coimbatore, India}
}
```

---

## Acknowledgements

This work builds on:

- [IndicBERT / IndicNLP Suite](https://github.com/AI4Bharat/indic-nlp-library) — Kakwani et al., 2020
- [IndicTrans2 / Indic–Indic 1B](https://github.com/AI4Bharat/IndicTrans2) — Gala et al., 2023
- [IndicTransToolkit](https://github.com/VarunGumma/IndicTransToolkit) — pre/post-processing for IndicTrans2
- [AugLy](https://github.com/facebookresearch/AugLy) — Meta AI text augmentation
- [mT5](https://huggingface.co/google/mt5-small) — Xue et al., 2021

---

## License

Apache 2.0 — see [LICENSE](https://www.apache.org/licenses/LICENSE-2.0).
