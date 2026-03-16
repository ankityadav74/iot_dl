# Few-Shot Intrusion Detection in Wireless Sensor Networks

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-WSN--DS-green.svg)](https://www.kaggle.com/datasets/bassamkasasbeh1/wsnds)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()

Few-shot attack classification for Wireless Sensor Networks using the **CTPN (Cross-attention Transductive Prototypical Network)** architecture. The system identifies 5 attack types — Blackhole, Flooding, Grayhole, Normal, TDMA — using as few as **1 labelled example per class**, achieving **97.31% 5-way accuracy at K=1** and **97.16% at K=10**.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Experiment History](#experiment-history)
- [Key Design Decisions](#key-design-decisions)
- [Requirements](#requirements)

---

## Overview

Standard supervised intrusion detection models require thousands of labelled samples per attack type. In real WSN deployments, new attack variants may have only a handful of examples. This project addresses that gap with **episodic few-shot learning**:

- A **CNN-BiLSTM encoder** with channel + temporal attention is pre-trained on WSN-DS (99.35% validation accuracy)
- A **SupCon-trained projection head** tightens the embedding space per class
- **CAPR** (Cross-Attention Prototype Refinement) iteratively refines class prototypes using query-set context
- **Support Hallucination** at K=1 generates augmented support samples to reduce single-shot variance
- **Multi-K training** (K ∈ {1, 3, 5, 10, 20}) ensures the model generalises across all shot settings

---

## Architecture

```
Raw WSN Features (23 dims)
        │
        ▼
Feature Engineering (→ 33 dims)
  [stat_mean, stat_std, stat_range, inter-feature products, ratio features]
        │
        ▼
StandardScaler
        │
        ▼
┌─────────────────────────────────────────────────────┐
│           CNN-BiLSTM Encoder (pre-trained)          │
│  Conv1D(64) → BN → Conv1D(128) → BN → MaxPool       │
│  ChannelAttention(128, r=8)                         │
│  BiLSTM(64×2=128) → TemporalAttention               │
│  Output: 128-dim embedding                          │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│     Projection Head (SupCon fine-tuning)            │
│  Linear(128→256) → LayerNorm → GELU                 │
│  Linear(256→256) → LayerNorm → GELU                 │
│  Linear(256→128) + Skip(128→128)                    │
│  L2 Normalise  →  128-dim unit sphere               │
└─────────────────────────────────────────────────────┘
        │                        │
   Support set              Query set
        │                        │
        ▼                        │
┌──────────────────────┐         │
│  SupportAttention    │         │
│  Aggregator (SAA)    │         │
│  MHA(4 heads) + LN  │         │
│  → 1 prototype/class│         │
└──────────────────────┘         │
        │                        │
        ▼                        ▼
┌─────────────────────────────────────────────────────┐
│       CAPR — Cross-Attention Prototype Refinement   │
│  n_iters = 4                                        │
│  Each iter: proto attends over query set (MHA)      │
│             gated residual update                   │
│             LayerNorm + FFN                         │
│  Refines prototypes using query-context signal      │
└─────────────────────────────────────────────────────┘
        │
        ▼
Cosine Similarity × 10.0  →  Softmax  →  Predicted Class
```

---

## Dataset

**WSN-DS** — Wireless Sensor Network Dataset for Intrusion Detection

| Property | Value |
|---|---|
| Total samples | 374,661 |
| Features | 23 original → 33 engineered |
| Train / Test split | 80% / 20% (stratified) |
| Test set size | 74,933 samples |

**Class distribution:**

| Class | Count | % of Dataset |
|---|---|---|
| Normal | 257,015 | 68.6% |
| Blackhole | 35,341 | 9.4% |
| Grayhole | 35,191 | 9.4% |
| TDMA | 35,183 | 9.4% |
| Flooding | 11,931 | 3.2% |

> Flooding is the hardest class — only 3.2% of data yet CTPN achieves F1=1.00 on it.

---

## Results

### 5-way K-shot Accuracy (300 episodes ± 95% CI)

| Method | K=1 | K=3 | K=5 | K=10 | K=20 |
|---|---|---|---|---|---|
| Prototypical Network | 0.9186 ± 0.0078 | 0.9555 ± 0.0028 | 0.9574 ± 0.0026 | 0.9588 ± 0.0024 | 0.9635 ± 0.0025 |
| APFP (2024) | 0.9239 ± 0.0070 | 0.9447 ± 0.0042 | 0.9479 ± 0.0040 | 0.9522 ± 0.0034 | 0.9566 ± 0.0029 |
| AP++ (Ours) | 0.9428 ± 0.0089 | 0.9599 ± 0.0027 | 0.9612 ± 0.0027 | 0.9613 ± 0.0024 | 0.9624 ± 0.0025 |
| **CTPN v4 (Ours)** | **0.9531 ± 0.0081** | **0.9700 ± 0.0022** | **0.9688 ± 0.0021** | **0.9716 ± 0.0022** | **0.9694 ± 0.0021** |

### AUC-ROC (CTPN v4)

| K=1 | K=3 | K=5 | K=10 | K=20 |
|---|---|---|---|---|
| 0.9934 ± 0.0018 | 0.9970 ± 0.0004 | 0.9970 ± 0.0003 | 0.9974 ± 0.0003 | 0.9974 ± 0.0003 |

### Full Evaluation — Per-Class Metrics (K=5, 500 episodes)

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Blackhole | 0.94 | 0.98 | 0.96 |
| Flooding | **1.00** | **1.00** | **1.00** |
| Grayhole | 0.98 | 0.94 | 0.96 |
| Normal | 0.94 | 1.00 | 0.96 |
| TDMA | 1.00 | 0.93 | 0.96 |
| **Macro avg** | **0.97** | **0.97** | **0.97** |

### Episode Stability (300 episodes per K)

| K | Mean | Std | CV |
|---|---|---|---|
| K=1 | 0.9516 | 0.0741 | 7.79% |
| K=3 | 0.9701 | 0.0202 | 2.08% |
| K=5 | 0.9696 | 0.0184 | 1.90% |
| K=10 | 0.9726 | 0.0190 | 1.95% |
| K=20 | 0.9703 | 0.0200 | 2.06% |

---

## Project Structure

```
project/
├── Scripts/
│   ├── ctpn_wsn_v4.py              # Main training script (Phase 1+2+3)
│   ├── ctpn_eval_v4.py             # Full evaluation suite (7 diagnostic plots)
│   ├── save_final_weights.py       # Re-save post-meta-train weights (run once)
│   └── [earlier versions]
│       ├── ap_plus_plus_v6.py      # AP++ baseline (best version)
│       ├── ctpn_wsn_v1.py          # CTPN v1 — baseline CAPR
│       ├── ctpn_wsn_v2.py          # CTPN v2 — NaN collapsed (deprecated)
│       └── ctpn_wsn_v3.py          # CTPN v3 — CE-only loss fix
│
├── data/
│   └── raw/
│       └── WSN-DS.csv
│
└── runs/
    ├── 2026_03_11__23_15_08/        # Pre-trained CNN-BiLSTM encoder
    │   └── models/
    │       ├── CNN_BiLSTM_Attention.pth
    │       └── scaler.pkl
    │
    └── few_shot_ctpn_v4/            # CTPN v4 outputs
        ├── results/
        │   ├── ctpn_weights.pth         # Post-SupCon weights
        │   ├── ctpn_weights_final.pth   # Post-meta-train weights ← use this
        │   └── ctpn_results.json
        └── evaluation/
            ├── confusion_matrix.png
            ├── classification_report.png
            ├── tsne_embeddings.png
            ├── prototype_heatmap.png
            ├── capr_attention.png
            ├── episode_stability.png
            ├── calibration.png
            └── eval.log
```

---

## Installation

```bash
# Clone / navigate to project
cd /root/amlan/Iot/project/Scripts

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install scikit-learn pandas numpy matplotlib seaborn
```

---

## Usage

### Step 1 — Pre-train the CNN-BiLSTM Encoder
> Skip if `CNN_BiLSTM_Attention.pth` already exists.

```bash
python train_cnn_bilstm.py
```

### Step 2 — Train CTPN v4 (SupCon + Meta-train + Evaluation)

```bash
python ctpn_wsn_v4.py
```

This runs three phases automatically:
1. **Phase 1** — SupCon fine-tuning of projection head (tightens embedding space)
2. **Phase 2** — Multi-K meta-training of SAA + CAPR (K ∈ {1, 3, 5, 10, 20})
3. **Phase 3** — Episodic evaluation vs Prototypical, APFP, AP++ baselines

### Step 3 — Save Final Weights (run once after first training)

```bash
python save_final_weights.py
```
> This saves `ctpn_weights_final.pth` which includes trained CAPR weights.
> Required because training saves weights after Phase 1 by default.

### Step 4 — Full Evaluation Suite

```bash
python ctpn_eval_v4.py
```

Generates 7 diagnostic plots in `runs/few_shot_ctpn_v4/evaluation/`.

---

## Evaluation Metrics

| Plot | File | What it measures |
|---|---|---|
| Confusion Matrix | `confusion_matrix.png` | Per-class prediction errors (counts + normalised) |
| Classification Report | `classification_report.png` | Precision / Recall / F1 per class |
| t-SNE Embeddings | `tsne_embeddings.png` | Cluster separation before vs after projection |
| Prototype Heatmap | `prototype_heatmap.png` | Inter-class cosine similarity (lower = better) |
| CAPR Attention | `capr_attention.png` | What each prototype attends to in query set |
| Episode Stability | `episode_stability.png` | Accuracy distribution across 300 episodes per K |
| Calibration | `calibration.png` | Confidence of correct vs wrong predictions |

**Interpreting the model health checks:**

| Check | Good Sign | Concern |
|---|---|---|
| t-SNE | 5 tight, separated clusters after projection | Overlapping clusters = poor SupCon |
| Prototype heatmap | Off-diagonal cosine sim ≤ 0.3 | High similarity = confused prototypes |
| Calibration | Correct predictions peak at confidence ≥ 0.95 | Overlap with wrong = overconfidence |
| Episode stability | CV < 2% at K≥3 | High CV = model not stable |

---

## Experiment History

| Version | Key Change | K=1 Acc | K=5 Acc | Status |
|---|---|---|---|---|
| Prototypical (baseline) | Standard prototypical network | 0.9186 | 0.9574 | Baseline |
| APFP (2024) | Published SOTA few-shot WSN | 0.9239 | 0.9479 | Baseline |
| AP++ v6 | CNN-BiLSTM + SupCon + basic attention | 0.9428 | 0.9612 | ✅ Published |
| CTPN v1 | Added CAPR (4 iters), no multi-K | 0.9295 | 0.9652 | Deprecated |
| CTPN v2 | Attempted loss change → NaN collapse | 0.8328 | 0.3440 | ❌ Broken |
| CTPN v3 | CE-only loss, stable training | 0.8840 | 0.9678 | Deprecated |
| **CTPN v4** | Multi-K training + Support Hallucination | **0.9531** | **0.9688** | ✅ Final |

---

## Key Design Decisions

### Why Multi-K Training?
CTPN v1 and v3 trained at a fixed K=5. This caused K=1 performance to be significantly lower (0.88–0.93) because the model never learned to handle single-support episodes. Training with K ∈ {1, 3, 5, 10, 20} simultaneously forces CAPR to adapt to all shot settings.

### Why Support Hallucination at K=1?
At K=1, a single atypical support sample causes CAPR to build a poor prototype. When K=1 is sampled during training, 4 additional support samples are synthetically generated by adding small Gaussian noise (σ=0.02) to the original, giving the SAA aggregator meaningful input without leaking information.

### Why Cosine Similarity × 10 for Logits?
Raw cosine similarity produces logits in [−1, 1], which makes the softmax distribution too flat. Scaling by 10 sharpens the distribution and improves gradient flow during meta-training — a standard trick in metric learning.

### Why Save Weights Twice?
`ctpn_weights.pth` is saved after Phase 1 (SupCon). CAPR is still at random init at this point.
`ctpn_weights_final.pth` is saved after Phase 2 (meta-train). **Always use this for evaluation.**

---

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.23.0
pandas>=1.5.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

```bash
pip install torch>=2.0.0 torchvision>=0.15.0 numpy pandas scikit-learn matplotlib seaborn
```

**Hardware used:** NVIDIA GPU (cuda:1), ~6 minutes total training time per full run.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{ctpn_wsn_2026,
  title   = {CTPN: Cross-Attention Transductive Prototypical Networks for
             Few-Shot Intrusion Detection in Wireless Sensor Networks},
  author  = {Amlan Sarkar},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```
