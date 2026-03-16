# Few-Shot Intrusion Detection in Wireless Sensor Networks

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-WSN--DS-green.svg)](https://www.kaggle.com/datasets/bassamkasasbeh1/wsnds)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()

Few-shot attack classification for Wireless Sensor Networks using two novel approaches built on a shared
CNN-BiLSTM-Attention encoder pre-trained on WSN-DS.
The final system — **CTPN v4** — identifies 5 attack types using as few as **1 labelled example per class**,
achieving **95.31% accuracy at K=1** and **97.16% at K=10** with AUC ≥ 0.993 across all shot settings.

---

## Table of Contents

- [Motivation](#motivation)
- [Shared Foundation — CNN-BiLSTM-Attention Encoder](#shared-foundation)
- [Phase 1 Supervised Baselines](#phase-1-supervised-baselines)
- [Approach 1 — AP++ Adaptive Prototype](#approach-1--ap-adaptive-prototype)
- [Approach 2 — CTPN Cross-Attention Transductive Prototypical Network](#approach-2--ctpn)
- [All Versions Timeline](#all-versions-timeline)
- [Results](#results)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation Suite](#evaluation-suite)
- [Key Design Decisions](#key-design-decisions)
- [Requirements](#requirements)

---

## Motivation

Standard supervised intrusion detection requires thousands of labelled samples per attack type.
In real WSN deployments, new attack variants may appear with only a handful of examples.
This project solves that with **episodic few-shot learning**: the model must classify a new attack
class from K labelled support examples (K = 1, 3, 5, 10, or 20) without retraining.

Two approaches were developed sequentially. Approach 1 (AP++) improved prototype construction using
hand-crafted heuristics. Approach 2 (CTPN) replaced heuristics with learnable neural attention
modules, achieving consistently better results across all K values.

---

## Shared Foundation

### CNN-BiLSTM-Attention Encoder (pre-trained)

Both approaches share the same frozen encoder, pre-trained in a fully supervised setting on WSN-DS.

```
Raw WSN Features (23 dims)
        │
Feature Engineering (→ 33 dims)
  [stat_mean, stat_std, stat_range, inter-feature products, ratio features]
        │
StandardScaler
        │
┌─────────────────────────────────────────────────────┐
│            CNN-BiLSTM-Attention Encoder             │
│                                                     │
│  Conv1D(64) → BatchNorm → ReLU                      │
│  Conv1D(128) → BatchNorm → ReLU → MaxPool           │
│  ChannelAttention (Squeeze-and-Excitation, r=8)     │
│    → re-weights 128 feature channels                │
│  BiLSTM(64×2 = 128 hidden)                          │
│  TemporalAttention                                  │
│    → soft attention over T timesteps                │
│    → weighted sum → 128-dim context vector          │
│                                                     │
│  Output: 128-dim embedding per sample               │
└─────────────────────────────────────────────────────┘
```

**Why this encoder?** In comparative supervised training across 6 architectures (LSTM, BiLSTM,
CNN1D, CNN-BiLSTM, CNN-BiLSTM-Attention, TransformerIDS), CNN-BiLSTM-Attention achieved the
highest validation accuracy (99.30%) while remaining compact (137K parameters, 538KB).
It was selected as the shared backbone for both few-shot approaches.

---

## Phase 1 Supervised Baselines

Before building the few-shot system, 6 deep learning architectures and 2 ML baselines
were trained and compared:

| Model | Test Acc | F1 | Parameters | Deployment Tier |
|---|---|---|---|---|
| XGBoost | 0.9968 | 0.9968 | — | Cloud |
| Random Forest | 0.9929 | 0.9930 | — | Cloud |
| **CNN-BiLSTM-Attention** | **0.9931** | **0.9931** | 137K | **Edge ← selected** |
| CNN1D | 0.9922 | 0.9922 | 54K | Edge |
| CNN-BiLSTM | 0.9900 | 0.9900 | 133K | Edge |
| LSTM | 0.9893 | 0.9893 | 135K | Gateway |
| BiLSTM | 0.9875 | 0.9876 | 336K | Cloud |
| TransformerIDS | 0.9691 | 0.9686 | 38K | Edge |

Three novel contributions were also implemented at this stage:
- **C1 — MC Dropout Uncertainty Quantification**: 30-sample MC inference, flags novel/zero-day attacks below confidence threshold (99.84% accuracy on high-confidence samples)
- **C2 — Concept Drift Detection**: Chunk-based sliding accuracy monitor with automatic RF retraining on drift (0.85% accuracy improvement over static model)
- **C3 — Energy-Complexity Trade-off Framework**: Scores each model on accuracy × speed × compactness for tier-based WSN deployment (Edge / Gateway / Cloud)

---

## Approach 1 — AP++ (Adaptive Prototype)

### What Is It?

AP++ is a **training-free few-shot method** — it uses the frozen encoder embeddings directly,
with no additional gradient steps. Given K support embeddings per class, AP++ builds an
**adaptive prototype** (a refined class centroid) and classifies each query by Euclidean
distance to the nearest prototype.

### What Standard Prototypical Networks Do (and Why It Fails)

The baseline prototypical network computes a class prototype as the plain **mean** of support
embeddings, then uses Euclidean distance to classify queries. APFP (2024) adds a fixed-temperature
cosine soft-attention weighting over support samples, but still suffers 5 systematic weaknesses:

| APFP 2024 Shortcoming | AP++ Solution |
|---|---|
| S1: Fixed cosine similarity only | **A1: Dual-metric blend** — cosine + euclidean (α=0.6) |
| S2: Fixed temperature τ=5 | **A2: Adaptive temperature** — τ = basetemp / intraclass_std |
| S3: No outlier removal | **A3: Outlier filter** — removes support samples >2σ from centroid |
| S4: One-shot prototype (no refinement) | **A4: EM iterative refinement** — up to 4 iterations |
| S5: Ignores class boundaries | **A5: Inter-class repulsion** — down-weights support samples near other class prototypes |

### How Each Component Works

**A1 — Dual-Metric Blending:**
Instead of cosine-only, AP++ computes both cosine similarity and negative Euclidean similarity
between each support sample and the query, normalises each to [0,1], then blends:
`combined = 0.6 × cosine + 0.4 × euclidean`
This captures both angular and magnitude information in the embedding space.

**A2 — Adaptive Temperature:**
The softmax temperature τ is set proportional to the intra-class embedding spread:
`τ = basetemp / intraclass_std`
When support samples are tightly clustered (low std), temperature is high (sharper attention).
When spread out, temperature is lower (more uniform weighting).

**A3 — Outlier-Robust Filtering:**
Before computing the prototype, support samples further than 2σ from the initial centroid
are removed. This prevents a single atypical support example from corrupting the prototype,
which is critical at K=1 where there is no redundancy.

**A4 — EM Iterative Refinement:**
After computing an initial prototype, the attention weights and prototype are alternately
re-estimated up to 4 times. Each iteration brings the prototype closer to the true
cluster centre in embedding space, similar to a few-step EM algorithm.

**A5 — Inter-Class Repulsion:**
Each support sample is weighted inversely proportional to its proximity to **other** class
prototypes: `repulsion = 1 − exp(−min_other_dist / √D)`
Support samples near class boundaries contribute less to the prototype, making it more
discriminative.

### AP++ Ablation Study (K=5)

| Configuration | Accuracy | Gain over APFP |
|---|---|---|
| APFP 2024 Baseline | 0.9479 | — |
| + Dual Metric (A1) | 0.9503 | +0.0024 |
| + Adaptive Temp (A2) | 0.9527 | +0.0048 |
| + Outlier Filter (A3) | 0.9541 | +0.0062 |
| + Inter-Class Repulsion (A4) | 0.9578 | +0.0099 |
| **Full AP++ (A1-A5)** | **0.9612** | **+0.0133** |

### Why AP++ Is Better Than APFP

AP++ gains +1.33 percentage points at K=5 and +1.89pp at K=1 over APFP by addressing
all five structural weaknesses simultaneously. Because it requires no training, it
generalises to new attack classes instantly at inference time.

### AP++ Limitation

AP++ is entirely heuristic — it has no learnable parameters and cannot adapt its prototype
construction strategy based on feedback. Each component was designed independently,
and their interactions are not optimised end-to-end. This motivated Approach 2.

---

## Approach 2 — CTPN

### What Is It?

CTPN (Cross-Attention Transductive Prototypical Network) replaces AP++'s hand-crafted heuristics
with **trained neural modules** that learn how to build and refine prototypes end-to-end.
Three new components are added on top of the shared encoder:

```
Support Set (K × D embeddings per class)
        │
┌───────────────────────────────────────┐
│  Projection Head (SupCon-trained)     │
│  Linear(128→256) → LayerNorm → GELU  │
│  Linear(256→256) → LayerNorm → GELU  │
│  Linear(256→128) + Skip(128→128)     │
│  L2 Normalise → 128-d unit sphere    │
└───────────────────────────────────────┘
        │                       │
   Support projected        Query projected
        │
┌───────────────────────────────────────┐
│  Support Attention Aggregator (SAA)  │
│  MultiheadAttention(4 heads) + LN    │
│  K support embeddings → mean pool    │
│  → 1 prototype per class             │
└───────────────────────────────────────┘
        │
  N prototypes (one per class)
        │             │
        └──────┬──────┘
               │
┌──────────────────────────────────────────────────┐
│  CAPR — Cross-Attention Prototype Refinement     │
│  n_iters = 4                                     │
│                                                  │
│  Each iteration:                                 │
│    prototype attends over ALL query embeddings   │
│    (MultiheadAttention, 4 heads)                 │
│    gated residual update                         │
│    → prototype[i] += gate × attn_output[i]      │
│    LayerNorm + FFN                               │
│                                                  │
│  Prototypes refine themselves using query-       │
│  context signal — the model learns which         │
│  queries are informative for each class          │
└──────────────────────────────────────────────────┘
        │
Cosine Similarity × 10 → Softmax → Predicted Class
```

### What AP++ Does vs What CTPN Does

| Aspect | AP++ | CTPN |
|---|---|---|
| Prototype building | Hand-crafted attention weights | Learned SAA (trained MHA) |
| Prototype refinement | Fixed EM iterations | Trained CAPR (gradient-optimised) |
| Query-context usage | None — prototype ignores query set | CAPR explicitly attends to queries |
| Distance metric | Euclidean + cosine blend | Cosine similarity (unit sphere) |
| Embedding space | Raw encoder 128-d | SupCon-trained 128-d unit sphere |
| Learnable parameters | 0 | ~200K (ProjectionHead + SAA + CAPR) |
| Training required | No | Yes (SupCon + meta-train) |
| K=1 accuracy | 0.9428 | **0.9531** |
| K=5 accuracy | 0.9612 | **0.9688** |

### Why CTPN Is Better Than AP++

**1. SupCon Projection tightens the embedding space.**
The raw encoder embeds samples for classification. SupCon fine-tuning pulls same-class samples
together on a unit sphere and pushes different classes apart, making prototypes more
representative and cosine distance more reliable.

**2. SAA replaces simple mean with learned aggregation.**
AP++ uses exponential soft-attention to aggregate K support samples.
SAA uses multi-head self-attention over the K embeddings, learning which support samples
are most representative for a given episode context.

**3. CAPR uses query information to refine prototypes.**
AP++ prototypes are built entirely from support samples — queries have no influence.
CAPR lets each prototype iteratively attend over the full query set, adjusting itself
toward the queries it needs to discriminate. This is the core transductive advantage:
the model sees the query distribution and adapts before classifying.

**4. End-to-end training integrates all components.**
AP++ components were designed independently. CTPN trains all three modules jointly
via meta-learning, optimising the entire prototype-building + refinement + classification
pipeline with a single loss function.

### Training Pipeline

CTPN training has two phases:

**Phase 1 — Supervised Contrastive Learning (SupCon):**
The projection head is fine-tuned using SupCon loss on the pre-extracted encoder embeddings.
This shapes the 128-d embedding space so that same-class samples cluster tightly
and different classes are well-separated on the unit sphere.

**Phase 2 — Multi-K Meta-Training:**
SAA and CAPR are trained episodically with K randomly sampled from {1, 3, 5, 10, 20}
in each episode. Standard cross-entropy loss over cosine similarity logits (×10 temperature).
600 episodes total, N-way=5, N-query=15 per class.

---

## All Versions Timeline

### AP++ Versions

| Version | Key Change | K=1 Acc | K=5 Acc |
|---|---|---|---|
| Prototypical Network | Plain mean prototype + Euclidean | 0.9186 | 0.9574 |
| APFP (2024) | Fixed cosine soft-attention | 0.9239 | 0.9479 |
| AP++ v1 | + Dual-metric blend | ~0.926 | ~0.950 |
| AP++ v2 | + Adaptive temperature | ~0.928 | ~0.953 |
| AP++ v3 | + Outlier filter | ~0.930 | ~0.955 |
| AP++ v4 | + Inter-class repulsion | ~0.934 | ~0.958 |
| **AP++ v6 (final)** | + EM iterative refinement (all 5 components) | **0.9428** | **0.9612** |

### CTPN Versions

| Version | Key Change | K=1 Acc | K=5 Acc | Status |
|---|---|---|---|---|
| CTPN v1 | CAPR added, fixed K=5 training | 0.9295 | 0.9652 | Deprecated |
| CTPN v2 | Modified loss → NaN collapse | 0.8328 | 0.3440 | ❌ Broken |
| CTPN v3 | CE-only loss, NaN fix | 0.8840 | 0.9678 | Deprecated |
| **CTPN v4** | Multi-K training + Support Hallucination | **0.9531** | **0.9688** | ✅ Final |

**Why v2 collapsed:** Changing the loss function caused NaN gradients within 5 episodes.
The NaN propagated through CAPR's attention softmax → all logits became NaN → random predictions.

**Why v3 underperformed at K=1:** Training exclusively at K=5 meant CAPR never learned to handle
single-support episodes. When evaluated at K=1, the SAA received only 1 embedding per class
(degenerate aggregation) and CAPR had never seen this distribution during training.

**What v4 fixed (two mechanisms):**

*Multi-K training:* Each meta-training episode randomly samples K from {1, 3, 5, 10, 20}.
This forces CAPR to learn robust refinement under all support-set sizes, eliminating
the K=1 generalisation gap.

*Support Hallucination at K=1:* When K=1 is sampled, 4 additional synthetic support samples
are generated by adding small Gaussian noise (σ=0.02) to the original support embedding.
This gives SAA 5 inputs instead of 1, avoiding degenerate prototype aggregation during
training without injecting false label information.

---

## Results

### 5-way K-shot Accuracy (300 episodes ± 95% CI)

| Method | K=1 | K=3 | K=5 | K=10 | K=20 |
|---|---|---|---|---|---|
| Prototypical Network | 0.9186 ± 0.0078 | 0.9555 ± 0.0028 | 0.9574 ± 0.0026 | 0.9588 ± 0.0024 | 0.9635 ± 0.0025 |
| APFP (2024) | 0.9239 ± 0.0070 | 0.9447 ± 0.0042 | 0.9479 ± 0.0040 | 0.9522 ± 0.0034 | 0.9566 ± 0.0029 |
| AP++ v6 (Ours) | 0.9428 ± 0.0089 | 0.9599 ± 0.0027 | 0.9612 ± 0.0027 | 0.9613 ± 0.0024 | 0.9624 ± 0.0025 |
| **CTPN v4 (Ours)** | **0.9531 ± 0.0081** | **0.9700 ± 0.0022** | **0.9688 ± 0.0021** | **0.9716 ± 0.0022** | **0.9694 ± 0.0021** |

### CTPN v4 — AUC-ROC Across K

| K=1 | K=3 | K=5 | K=10 | K=20 |
|---|---|---|---|---|
| 0.9934 ± 0.0018 | 0.9970 ± 0.0004 | 0.9970 ± 0.0003 | 0.9974 ± 0.0003 | 0.9974 ± 0.0003 |

### CTPN v4 — Full Evaluation Per-Class (K=5, 500 episodes)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Blackhole | 0.94 | 0.98 | 0.96 |
| Flooding | **1.00** | **1.00** | **1.00** |
| Grayhole | 0.98 | 0.94 | 0.96 |
| Normal | 0.94 | 1.00 | 0.96 |
| TDMA | 1.00 | 0.93 | 0.96 |
| **Macro avg** | **0.97** | **0.97** | **0.97** |

> Flooding achieves perfect F1 despite being the rarest class (3.2% of dataset, 662 test samples).

### Episode Stability (CTPN v4, 300 episodes per K)

| K | Mean | Std | CV |
|---|---|---|---|
| K=1 | 0.9516 | 0.0741 | 7.79% — expected single-shot variance |
| K=3 | 0.9701 | 0.0202 | 2.08% |
| K=5 | 0.9696 | 0.0184 | **1.90%** |
| K=10 | 0.9726 | 0.0190 | 1.95% |
| K=20 | 0.9703 | 0.0200 | 2.06% |

---

## Dataset

**WSN-DS** — Wireless Sensor Network Dataset for Intrusion Detection

| Property | Value |
|---|---|
| Total samples | 374,661 |
| Original features | 23 |
| After feature engineering | 33 |
| Train / Test split | 80% / 20% (stratified) |
| Test set size | 74,933 |

**Class distribution (post-SMOTE training set):**

| Class | Raw Count | Raw % | Post-SMOTE |
|---|---|---|---|
| Normal | 272,052 | 90.8% | 272,052 |
| Grayhole | 11,677 | 3.9% | 11,677 |
| Blackhole | 10,049 | 2.7% | 10,000 |
| TDMA | 5,310 | 1.8% | 10,000 |
| Flooding | 3,312 | 0.9% | 10,000 |

---

## Project Structure

```
project/
├── Scripts/
│   ├── [Phase 1 — Supervised Baselines]
│   │   └── wsn_pipeline.py                 # Full supervised training pipeline
│   │
│   ├── [Approach 1 — AP++]
│   │   ├── ap_plus_plus_v1.py              # Dual-metric only
│   │   ├── ap_plus_plus_v2.py              # + Adaptive temperature
│   │   ├── ap_plus_plus_v3.py              # + Outlier filter
│   │   ├── ap_plus_plus_v4.py              # + Repulsion
│   │   └── ap_plus_plus_v6.py              # Full AP++ (all 5 components) ← final
│   │
│   ├── [Approach 2 — CTPN]
│   │   ├── ctpn_wsn_v1.py                  # CAPR baseline
│   │   ├── ctpn_wsn_v2.py                  # NaN collapsed ❌
│   │   ├── ctpn_wsn_v3.py                  # CE-only fix
│   │   ├── ctpn_wsn_v4.py                  # Multi-K + Hallucination ← final
│   │   ├── ctpn_eval_v4.py                 # Full evaluation suite
│   │   └── save_final_weights.py           # Re-save post-meta-train weights
│
├── data/
│   └── raw/
│       └── WSN-DS.csv
│
└── runs/
    ├── 2026_03_11__23_15_08/               # Supervised training run
    │   └── models/
    │       ├── CNN_BiLSTM_Attention.pth    # Encoder weights (val_acc=0.9935)
    │       └── scaler.pkl
    │
    ├── few_shot_applusplus/                # AP++ outputs
    │   ├── plots/
    │   │   ├── applusplus_design.png       # APFP shortcomings → AP++ solutions
    │   │   ├── applusplus_comparison.png   # K-shot comparison plot
    │   │   ├── applusplus_ablation.png     # Component contribution
    │   │   └── applusplus_perclass.png     # Per-class at K=5
    │   └── results/
    │       └── applusplus.json
    │
    └── few_shot_ctpn_v4/                   # CTPN v4 outputs
        ├── results/
        │   ├── ctpn_weights.pth            # Post-SupCon weights (Phase 1 only)
        │   ├── ctpn_weights_final.pth      # Post-meta-train weights ← use for eval
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
cd /root/amlan/Iot/project/Scripts

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install scikit-learn pandas numpy matplotlib seaborn
```

---

## Usage

### Run Supervised Pre-training (skip if encoder exists)

```bash
python wsn_pipeline.py
# Trains 6 DL models + 2 ML baselines
# Saves CNN_BiLSTM_Attention.pth — used by both few-shot approaches
```

### Run AP++ (Approach 1 — no training required)

```bash
python ap_plus_plus_v6.py
# ~5 minutes | No GPU training needed
# Generates 4 plots + applusplus.json
```

### Run CTPN v4 (Approach 2 — full training)

```bash
python ctpn_wsn_v4.py
# ~6 minutes total
# Phase 1: SupCon fine-tuning
# Phase 2: Multi-K meta-training (600 episodes)
# Phase 3: Episodic evaluation vs all baselines
```

### Save Final Weights (run once after first CTPN training)

```bash
python save_final_weights.py
# Re-runs meta_train and saves ctpn_weights_final.pth
# Required because ctpn_wsn_v4.py saves weights after Phase 1 (before CAPR is trained)
```

### Run Full Evaluation Suite

```bash
python ctpn_eval_v4.py
# Generates 7 diagnostic plots in runs/few_shot_ctpn_v4/evaluation/
```

---

## Evaluation Suite

| Plot | File | What it measures |
|---|---|---|
| Confusion Matrix | `confusion_matrix.png` | Per-class prediction errors (raw counts + normalised) |
| Classification Report | `classification_report.png` | Precision / Recall / F1 per class (bar chart) |
| t-SNE Embeddings | `tsne_embeddings.png` | Cluster separation before vs after SupCon projection |
| Prototype Heatmap | `prototype_heatmap.png` | Inter-class cosine similarity — off-diagonal should be ≤ 0.3 |
| CAPR Attention | `capr_attention.png` | Which queries each prototype attends to (last iteration) |
| Episode Stability | `episode_stability.png` | Accuracy box plot + K=1 vs K=5 histogram |
| Calibration | `calibration.png` | Confidence of correct vs wrong predictions |

**Model health checklist:**

| Signal | Healthy | Concern |
|---|---|---|
| t-SNE after projection | 5 tight, separated clusters | Overlapping blobs = SupCon failed |
| Prototype heatmap off-diagonal | ≤ 0.3 | >0.5 = prototypes not discriminative |
| Calibration (right panel) | Correct predictions peak at confidence ≥ 0.95 | Overlap with wrong = overconfident |
| Episode stability K≥3 | CV < 2.5% | High CV = unstable training |
| CAPR attention | Each prototype attends mostly to same-class queries | Uniform attention = CAPR not learning |

---

## Key Design Decisions

### Why Two Approaches?

AP++ demonstrates that **systematic heuristic improvements** can significantly outperform APFP 2024
without any additional training. This makes it deployment-friendly and interpretable.
CTPN then shows that **replacing heuristics with learned modules** gives consistent additional gains,
at the cost of a one-time training step.

### Why SupCon Before Meta-Training?

Meta-training optimises CAPR to refine prototypes. If the embedding space is poor (raw encoder),
CAPR wastes capacity learning to correct bad embeddings rather than learning good prototype dynamics.
SupCon first aligns the embedding space so CAPR can focus entirely on prototype refinement.

### Why Save Weights After Meta-Train?

`ctpn_weights.pth` is saved after Phase 1 (SupCon only). CAPR weights are random at that point.
`ctpn_weights_final.pth` is saved after Phase 2 (meta-train). **Always use this for evaluation.**
Loading the wrong file produces ~19% accuracy (near-random for 5-class problem).

### Why Cosine × 10 for Logits?

On the unit sphere (SupCon output), raw cosine similarity lies in [−1, 1].
Passing these directly to softmax produces very flat distributions (low confidence).
Scaling by 10 sharpens the logit distribution and stabilises meta-training gradients.
This is standard practice in metric learning (ProtoNet, CLIP, etc.).

### Why K=1 Has Higher Variance?

At K=1, the single support sample fully determines the prototype. If it happens to be an
atypical example of its class, the prototype is poor and the episode fails. Support Hallucination
reduces this during training, but at inference K=1 still uses only the real support sample.
The result: high mean accuracy (0.9516) but higher episode-to-episode variance (CV=7.79%),
with a bimodal distribution — most episodes succeed near 0.97, a small tail fails near 0.60.
This is the fundamental challenge of 1-shot learning and is consistent with published literature.

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
pip install torch>=2.0.0 numpy pandas scikit-learn matplotlib seaborn
```

**Hardware:** NVIDIA H100 PCIe (cuda:1), PyTorch 2.1.0+cu128

---

## Citation

```bibtex
@misc{ctpn_wsn_2026,
  title   = {AP++ and CTPN: Heuristic and Neural Few-Shot Approaches for
             Intrusion Detection in Wireless Sensor Networks},
  author  = {Amlan Sarkar},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```
