<div align="center">

# TouchSeqNet

**Self-Supervised Touch Behavioral Authentication via Masked Sequence Modeling**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<p>
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-datasets">Datasets</a> •
  <a href="#-results">Results</a> •
  <a href="#-citation">Citation</a>
</p>

---

</div>

## Overview

TouchSeqNet is a two-stage framework for touch-based user authentication. It first learns general behavioral representations through **self-supervised masked sequence modeling**, then performs user identity verification via a **Siamese network** with contrastive learning.

<div align="center">

```
                         ┌─────────────────────────────────────────┐
                         │          Stage 1: Pretraining           │
                         │                                         │
                         │   Touch Sequence                        │
                         │        │                                │
                         │   [Input Projection]                    │
                         │        │                                │
                         │   ┌────┴────┐                           │
                         │   │  Mask   │                           │
                         │   ┌────┐ ┌────┐                        │
                         │   │Vis.│ │Mask│                         │
                         │   └──┬─┘ └──┬─┘                        │
                         │      │      │                           │
                         │  Encoder  Mom.Encoder                   │
                         │      │      │                           │
                         │  Regressor──┘  Tokenizer                │
                         │      │            │                     │
                         │  Align Loss + Reconstruct Loss          │
                         └─────────────────────────────────────────┘

                         ┌─────────────────────────────────────────┐
                         │          Stage 2: Finetuning            │
                         │                                         │
                         │   Seq 1            Seq 2                │
                         │     │                │                  │
                         │  [Input Proj.]    [Input Proj.]         │
                         │     │                │    (shared)      │
                         │  [Transformer]    [Transformer]         │
                         │     │                │                  │
                         │  [TCN + Attn]     [TCN + Attn]          │
                         │     │                │                  │
                         │  [AvgPool]        [AvgPool]             │
                         │     └──── Concat ────┘                  │
                         │            │                            │
                         │       [Classifier]                      │
                         │            │                            │
                         │     Same / Different                    │
                         │                                         │
                         │  Contrastive Loss + CrossEntropy Loss   │
                         └─────────────────────────────────────────┘
```

</div>

## Highlights

- **Self-supervised pretraining** — Masked sequence modeling with momentum encoder and Gumbel-Softmax tokenizer learns robust touch representations without labels
- **Transformer + TCN hybrid** — Transformer encoder captures global dependencies, TCN with channel attention captures local temporal patterns
- **Siamese verification** — Pairwise architecture with joint contrastive + classification loss for identity authentication
- **Multi-dataset support** — Unified pipeline for Touchalytics, BioIdent, and multi-finger gesture datasets

## Architecture

### Stage 1: Self-Supervised Pretraining

| Component | Description |
|:---|:---|
| **Input Projection** | 1D convolution maps raw multi-channel sequences into `d_model`-dim embeddings |
| **Positional Embedding** | Learnable position encodings for temporal order |
| **Transformer Encoder** | Multi-layer multi-head self-attention blocks |
| **Momentum Encoder** | EMA-updated encoder for stable target representations |
| **Tokenizer** | Gumbel-Softmax discrete tokenizer for reconstruction targets |
| **Regressor** | Cross-attention module predicts masked representations from visible ones |

**Pretraining Objective:** `L = α · Align(MSE) + β · Reconstruct(CE)`

### Stage 2: Supervised Finetuning

| Component | Description |
|:---|:---|
| **Encoder-TCN** | Pretrained Transformer + TCN with multi-head attention + channel attention |
| **Siamese Network** | Shared-weight dual-branch architecture for pairwise comparison |
| **Classifier** | MLP head on concatenated embeddings for binary classification |

**Finetuning Objective:** `L = m · ContrastiveLoss + n · CrossEntropyLoss`

## Datasets

<table>
<thead>
  <tr>
    <th>Dataset</th>
    <th>Features</th>
    <th>Channels</th>
    <th>Description</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td><b>Touchalytics</b></td>
    <td>time, x, y, pressure, finger_area</td>
    <td align="center">5</td>
    <td>Touch stroke sequences with action states</td>
  </tr>
  <tr>
    <td><b>BioIdent</b></td>
    <td>x, y, pressure, finger_area</td>
    <td align="center">4</td>
    <td>Multi-device touch behavior data</td>
  </tr>
  <tr>
    <td><b>Finger Gesture</b></td>
    <td>Multi-finger time series</td>
    <td align="center">25</td>
    <td>Five-finger gesture interaction data</td>
  </tr>
</tbody>
</table>

<details>
<summary><b>Data Preprocessing Pipeline</b></summary>
<br>

```
Raw CSV ──► Group by User ──► Extract Sequences ──► Normalize ──► Filter ──► Pair Generation ──► DataLoader
```

1. **Read & Group** — Load CSV, group by `user_id` (and `device_id`)
2. **Sequence Extraction** — Extract valid touch strokes via action state machine (`0→2→1`)
3. **Normalization** — Differencing for time/x/y + Z-Score for pressure/area
4. **Length Filtering** — Remove extreme-length sequences (top/bottom 20%)
5. **Pair Generation** — Positive pairs (same user) + negative pairs (different users), balanced 1:1
6. **Train/Val Split** — 60/40 stratified split with padding and masking

</details>

## Quick Start

### Installation

```bash
# Core dependencies
pip install torch scikit-learn pandas numpy matplotlib tqdm

# Optional: Mamba-based attention variant
pip install mamba-ssm
```

<details>
<summary><b>Full Requirements</b></summary>
<br>

| Package | Version | Required |
|:---|:---|:---:|
| Python | >= 3.8 | Yes |
| PyTorch | >= 1.12 | Yes |
| scikit-learn | latest | Yes |
| pandas | latest | Yes |
| numpy | latest | Yes |
| matplotlib | latest | Yes |
| tqdm | latest | Yes |
| mamba-ssm | latest | Optional |

</details>

### Prepare Data

Place dataset CSV files under the `data/` directory:

```
data/
├── touchalytics/data.csv
├── biodent/rawdata.csv
└── BehavePassDB/
```

### Train

```bash
python main.py
```

This will automatically run the full pipeline:
1. Load and preprocess the dataset
2. Self-supervised pretraining → saves `pretrain_model.pkl`
3. Supervised finetuning with per-epoch evaluation
4. Output training curves and metrics

### Custom Configuration

```bash
python main.py \
  --d_model 64 \
  --layers 8 \
  --attn_heads 4 \
  --num_epoch_pretrain 1 \
  --num_epoch 10 \
  --lr 0.001 \
  --train_batch_size 256 \
  --mask_ratio 0.5
```

<details>
<summary><b>All Hyperparameters</b></summary>
<br>

| Parameter | Default | Description |
|:---|:---:|:---|
| `d_model` | 64 | Transformer hidden dimension |
| `layers` | 8 | Number of Transformer encoder layers |
| `attn_heads` | 4 | Number of attention heads |
| `dropout` | 0.2 | Dropout rate |
| `wave_length` | 4 | 1D conv kernel & stride size |
| `vocab_size` | 192 | Tokenizer vocabulary size |
| `mask_ratio` | 0.5 | Masking ratio during pretraining |
| `momentum` | 0.99 | Momentum encoder EMA coefficient |
| `alpha` | 5.0 | Align loss weight (pretrain) |
| `beta` | 1.0 | Reconstruct loss weight (pretrain) |
| `m` | 10 | Contrastive loss weight (finetune) |
| `n` | 1 | CrossEntropy loss weight (finetune) |
| `num_epoch_pretrain` | 1 | Pretraining epochs |
| `num_epoch` | 10 | Finetuning epochs |
| `lr` | 0.001 | Learning rate |
| `weight_decay` | 0.01 | Weight decay (AdamW) |
| `train_batch_size` | 256 | Training batch size |
| `val_batch_size` | 256 | Validation batch size |

</details>

## Results

The model is evaluated on the following metrics per epoch:

| Metric | Description |
|:---|:---|
| **Accuracy** | Overall classification accuracy |
| **F1 Score** | Harmonic mean of precision and recall |
| **AUC** | Area Under the ROC Curve |
| **FAR** | False Acceptance Rate |
| **FRR** | False Rejection Rate |
| **EER** | Equal Error Rate (where FAR ≈ FRR) |

### Output Files

```
exp/model/<dataset>/
└── pretrain_model.pkl              # Pretrained model weights

out/figure/<dataset>/picture/
├── loss_curve.png                  # Training loss curve
├── accuracy_curve.png              # Accuracy curve (with best epoch marked)
├── f1 score_curve.png              # F1 score curve
├── far_curve.png                   # FAR curve
├── frr_curve.png                   # FRR curve
├── eer_curve.png                   # EER curve
├── metrics_overview.png            # Combined 2x2 metrics subplot
└── result.txt                      # Tabular metrics per epoch
```

## Project Structure

```
TouchSeqNet/
├── main.py                   # Entry point
├── args.py                   # Hyperparameter configuration
├── process.py                # Training & evaluation logic
├── loss.py                   # Align, Reconstruct, ContrastiveLoss
├── dataloader_help.py        # Dataset dispatcher
│
├── model/
│   ├── TouchSeqNet.py        # Encoder, Tokenizer, Regressor
│   ├── Encoder_TCN.py        # TCN, Encoder-TCN, SiameseClassifier
│   └── layers.py             # Attention, TransformerBlock, TemporalBlock, etc.
│
├── data_process/
│   ├── touchalytics.py       # Touchalytics processing
│   ├── biodent.py            # BioIdent processing
│   └── ffinger.py            # Finger gesture processing
│
├── utils/
│   ├── figure_help.py        # Visualization utilities
│   └── ffingers_help.py      # Finger data utilities
│
├── data/                     # Datasets (CSV)
├── paper/                    # Related paper
├── out/                      # Training outputs
└── picture/                  # Saved plots
```

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{touchseqnet,
  title   = {TouchSeqNet: Self-Supervised Touch Behavioral Authentication
             via Masked Sequence Modeling},
  year    = {2025},
}
```

## License

This project is released under the [MIT License](LICENSE).
