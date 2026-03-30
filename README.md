# 🛡️ WSN-DS Intrusion Detection System
### Deep Learning + Few-Shot Meta-Learning for Multi-Attack Classification in Wireless Sensor Networks

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10-red?logo=pytorch)
![CUDA](https://img.shields.io/badge/CUDA-12.8-green?logo=nvidia)
![Accuracy](https://img.shields.io/badge/Best%20Accuracy-99.68%25-brightgreen)
![Few-Shot](https://img.shields.io/badge/CTPN%205--shot-96.88%25-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Overview

This project presents a **comprehensive deep learning + few-shot meta-learning pipeline** for intrusion detection in Wireless Sensor Networks (WSN) using the **WSN-DS dataset**. The system goes far beyond standard classification — it addresses six critical dimensions for real-world deployment:

| Dimension | Method | Result |
|---|---|---|
| 🎯 **Accuracy** | 8 models benchmarked | Up to **99.68%** |
| 🔍 **Trustworthiness** | MC Dropout Uncertainty | 99.84% on high-confidence samples |
| 🔄 **Adaptability** | Concept Drift Detection | +0.85% improvement on drifted data |
| ⚡ **Deployability** | Energy-Complexity Framework | Edge / Gateway / Cloud tiering |
| 🧠 **Explainability** | SHAP + LIME + Attention Viz | Per-class local & global explanations |
| 🛡️ **Robustness** | FGSM Adversarial Testing | Vulnerability boundary quantified |
| 🔬 **Few-Shot Generalisation** | AP++ + CTPN (novel) | 96.88% accuracy at 5-shot |

---

## 📊 Dataset — WSN-DS

### What is WSN-DS?

WSN-DS (Wireless Sensor Network Dataset) is a purpose-built dataset for intrusion detection in IoT/WSN environments. It simulates a **LEACH (Low Energy Adaptive Clustering Hierarchy)** routing protocol network and captures real WSN attack behaviour — unlike generic datasets like KDD99 or NSL-KDD which were designed for wired networks.

| Property | Value |
|---|---|
| Total records | **374,661** |
| Raw features | **18** |
| Engineered features | **33** |
| Classes | **5** |
| Missing values | **0** |
| Train samples | 299,728 |
| Test samples | 74,933 |

### Class Distribution (Before Any Processing)

```
Class         Count      Percentage   Bar
─────────────────────────────────────────────────────────
Normal        340,066    90.77%   ████████████████████████████████████
Grayhole       14,596     3.90%   ██
Blackhole      10,049     2.68%   █▌
TDMA            6,638     1.77%   █
Flooding        3,312     0.88%   ▌
─────────────────────────────────────────────────────────
```

> **Problem:** A naive model that always predicts "Normal" achieves 90.77% accuracy —
> but detects **zero attacks**. This is why class imbalance must be addressed before training.

### Raw Features (18 columns)

| # | Feature | Description |
|---|---|---|
| 1 | id | Node identifier |
| 2 | time | Timestamp |
| 3 | is_CH | Is cluster head (0/1) |
| 4 | who_CH | Which node is cluster head |
| 5 | dist_CH | Distance to cluster head |
| 6 | dist_BS | Distance to base station |
| 7 | ADV_S | Advertisement packets sent |
| 8 | ADV_R | Advertisement packets received |
| 9 | JOIN_S | Join requests sent |
| 10 | JOIN_R | Join requests received |
| 11 | SCH_S | Schedule packets sent |
| 12 | SCH_R | Schedule packets received |
| 13 | Rank | Node rank |
| 14 | DATA_S | Data packets sent |
| 15 | DATA_R | Data packets received |
| 16 | DATA_Sent_To_BS | Data sent to base station |
| 17 | dist_CH_to_BS | Distance from CH to BS |
| 18 | Attack type | **Target label** |

### Attack Types Explained

| Attack | What it does | Key affected features |
|---|---|---|
| **Normal** | Legitimate WSN traffic | All features in normal range |
| **Blackhole** | Malicious node drops ALL packets it receives | DATA_R high, DATA_S → 0 |
| **Grayhole** | Selectively drops packets (harder to detect than Blackhole) | DATA_R > DATA_S (selective) |
| **Flooding** | Sends excessive JOIN/ADV packets to exhaust node energy | JOIN_S, ADV_S very high |
| **TDMA** | Disrupts time-slot scheduling to cause collisions | SCH_S, SCH_R anomalous |

---

## 🔧 Preprocessing Pipeline

The raw data goes through **5 sequential steps** before model training:

```
RAW DATA (374,661 × 18)
        │
        ▼  Step 1
┌───────────────────┐
│  Data Cleaning    │  Check missing values → 0 found, no rows dropped
│  & Encoding       │  LabelEncode target: Blackhole=0, Flooding=1,
│                   │                      Grayhole=2, Normal=3, TDMA=4
└────────┬──────────┘
         │
         ▼  Step 2
┌───────────────────┐
│  Train/Test Split │  80% train (299,728) / 20% test (74,933)
│  (Stratified)     │  stratify=y → class proportions preserved in both splits
└────────┬──────────┘
         │
         ▼  Step 3
┌───────────────────┐
│  Feature          │  18 → 33 features (computed on train, applied to test)
│  Engineering      │  +7 statistical, +3 interaction, +5 ratio features
└────────┬──────────┘
         │
         ▼  Step 4
┌───────────────────┐
│  SMOTE Balancing  │  Minority class upsampling (train set ONLY)
└────────┬──────────┘
         │
         ▼  Step 5
┌───────────────────┐
│  StandardScaler   │  fit() on train only, transform() both train & test
│  Normalisation    │  mean=0, std=1 per feature
└────────┬──────────┘
         │
         ▼
  READY FOR TRAINING
  Train: 313,729 × 33
  Test:   74,933 × 33
```

### Step 1 — Data Cleaning & Label Encoding

- Verified zero missing values across all 374,661 rows
- Applied `LabelEncoder` to convert string class names to integers:

```
Blackhole → 0 | Flooding → 1 | Grayhole → 2 | Normal → 3 | TDMA → 4
```

> PyTorch's `CrossEntropyLoss` expects integer class indices directly — it applies
> softmax + log internally. One-hot encoding would be redundant and wasteful.

### Step 2 — Stratified Train/Test Split (80/20)

| Class | Full Dataset | Train Set | Test Set |
|---|---|---|---|
| Normal | 90.77% | 90.77% | 90.77% |
| Grayhole | 3.90% | 3.90% | 3.90% |
| Blackhole | 2.68% | 2.68% | 2.68% |
| TDMA | 1.77% | 1.77% | 1.77% |
| Flooding | 0.88% | 0.88% | 0.88% |

> ⚠️ **Critical:** SMOTE is applied ONLY to the **train set AFTER splitting**.
> Applying SMOTE before splitting causes **data leakage** — synthetic samples derived
> from test data contaminate training, artificially inflating evaluation scores.

### Step 3 — Feature Engineering (18 → 33 features)

#### Statistical Features (+7)

| New Feature | Formula | Why useful |
|---|---|---|
| `stat_mean` | mean of all features per row | Global traffic activity level |
| `stat_std` | std of all features per row | Traffic variability — attacks cause unusual variance |
| `stat_max` | max feature value per row | Detects extreme outlier behaviour |
| `stat_min` | min feature value per row | Detects dropped-to-zero features (Blackhole: DATA_S→0) |
| `stat_range` | max − min per row | Overall feature spread per record |
| `stat_skew` | skewness per row | Flooding has highly skewed packet distributions |
| `stat_kurt` | kurtosis per row | Detects heavy-tailed / impulsive attack patterns |

#### Interaction Features (+3)

| New Feature | Formula | Why useful |
|---|---|---|
| `inter_A_x_B` | top_var_feat_1 × top_var_feat_2 | Joint effect of two most variable features |
| `inter_B_x_C` | top_var_feat_2 × top_var_feat_3 | Cross-feature anomaly signal |
| `inter_C_x_D` | top_var_feat_3 × top_var_feat_4 | Non-linear feature combinations |

> Top 4 features by **variance** are selected — these vary the most between Normal and attack traffic.

#### Ratio Features (+5)

| New Feature | Formula | Why useful |
|---|---|---|
| `ratio_feat_i` | feat_i / (row_sum + 1e-10) | Relative contribution of each feature — e.g. DATA_R >> DATA_S in Blackhole |

### Step 4 — Manual SMOTE Balancing

#### Before SMOTE (raw train split):
```
Class       Samples    % of train
────────────────────────────────────────────────────
Normal      272,052    90.77%   ← completely dominates
Grayhole     11,677     3.90%
Blackhole     8,039     2.68%
TDMA          5,310     1.77%
Flooding      2,650     0.88%   ← only 2,650 real samples!
```

#### After SMOTE (balanced train set):
```
Class       Samples    Change
──────────────────────────────────────────────────────
Normal      272,052    unchanged (already dominant)
Grayhole     11,677    unchanged (above threshold)
Blackhole    10,000    +1,961 synthetic samples added
TDMA         10,000    +4,690 synthetic samples added
Flooding     10,000    +7,350 synthetic samples added
──────────────────────────────────────────────────────
Total        313,729   +14,001 synthetic samples total
```

#### How SMOTE Generates Synthetic Samples:

```
For each minority sample x_i:
  1. Find k=5 nearest neighbours within the same class
  2. Randomly pick one neighbour x_nn
  3. Synthesise a new point along the line between them:
       x_new = x_i + λ × (x_nn - x_i)    where λ ∈ [0, 1] random

  x_i  ●────────────●  x_nn
           ◆  ◆  ◆       ← new synthetic points placed between real ones
```

> Minority classes are capped at 10,000 (not matched to Normal's 272,052) to avoid
> introducing excessive noise from purely synthetic data. This gives enough
> boundary coverage without degrading distribution fidelity.

### Step 5 — StandardScaler Normalisation

```python
scaler = StandardScaler()
scaler.fit(X_train_balanced)       # learn mean & std from TRAIN ONLY
X_train_sc = scaler.transform(X_train_balanced)
X_test_sc  = scaler.transform(X_test)    # same transform applied to test
```

Every feature → mean = 0, std = 1. Scaler saved to `models/scaler.pkl` for inference on new data.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           WSN-DS Dataset                                │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
              ┌─────────────────▼──────────────────┐
              │     Preprocessing Pipeline          │
              │  Cleaning → SMOTE → Scale → Split   │
              └─────────────────┬──────────────────┘
                                │
         ┌──────────────────────┼──────────────────────┐
         │                      │                      │
┌────────▼──────────┐  ┌────────▼───────────┐  ┌──────▼───────────┐
│  Deep Learning    │  │  CNN-BiLSTM-Attn   │  │  ML Baselines    │
│  LSTM             │  │  ⭐ MAIN ENCODER   │  │  Random Forest   │
│  BiLSTM           │  │  (trained & saved) │  │  XGBoost         │
│  CNN_1D           │  └────────┬───────────┘  └──────────────────┘
│  CNN_BiLSTM       │           │
│  TransformerIDS   │    ┌──────▼──────────────────────────────────┐
└───────────────────┘    │     Frozen Encoder (128-D embeddings)   │
                         └──────┬──────────────────────────────────┘
                                │
         ┌──────────────────────┼──────────────────────┐
         │                      │                      │
┌────────▼──────────┐  ┌────────▼───────────┐  ┌──────▼───────────┐
│  Few-Shot Methods │  │  AP++ (Novel C1)   │  │  CTPN (Novel C2) │
│  Prototypical Net │  │  Adaptive Proto++  │  │  Contrastive     │
│  Matching Net     │  │  7 enhancements    │  │  Transformer     │
│  Relation Net     │  │  over APFP 2024    │  │  Prototype Net   │
│  Siamese Net      │  └────────────────────┘  └──────────────────┘
│  Inductive Trans. │
└───────────────────┘
         │
┌────────┴──────────────────────────────────────────────────────────┐
│                    System Contributions                            │
│  C1 MC-Dropout Uncertainty  |  C2 Concept Drift Detection         │
│  C3 Energy-Complexity Tiers |  C4 LIME Explainability             │
│  C5 FGSM Adversarial Test   |  C6 Attention Visualization         │
└───────────────────────────────────────────────────────────────────┘
```

---

## ⭐ Core Encoder — CNN-BiLSTM with Dual Attention

```
Input (B, F, 1)
      │
      ▼
┌─────────────────────┐
│   Conv1D Block 1    │  F → 64 filters, kernel=3, BN + ReLU + MaxPool
│   Conv1D Block 2    │  64 → 128 filters, kernel=3, BN + ReLU + MaxPool
└──────────┬──────────┘
           │
      ┌────▼──────────────────┐
      │  Channel Attention     │  Squeeze-and-Excitation (reduction=8)
      │  (Feature Reweighting) │  FC(128→16→128), Sigmoid gate
      └────┬──────────────────┘
           │
      ┌────▼──────────────────┐
      │  Bidirectional LSTM   │  128 hidden (64×2 directions), batch-first
      │  (Temporal Modelling) │  outputs h_t for all timesteps T
      └────┬──────────────────┘
           │
      ┌────▼──────────────────────────────────────────────┐
      │  Temporal Attention                                │
      │  score_t = tanh(W · h_t)                          │
      │  α_t     = softmax(score_t)                       │
      │  context = Σ α_t × h_t   →  128-D embedding      │
      └────┬──────────────────────────────────────────────┘
           │
      ┌────▼──────────────────┐
      │  Classifier Head       │  FC(128→64) → Dropout(0.3) → FC(64→5)
      │  [stripped for FSL]    │  Removed when used as few-shot encoder
      └───────────────────────┘
```

**Parameters:** 137,622 total (lightweight enough for edge deployment)

**Design rationale:**
- **Channel Attention** — not all 33 features are equally relevant per class (routing features dominate Blackhole; packet rate features dominate Flooding)
- **Temporal Attention** — avoids recency bias by learning *which timesteps* carry the attack signature, rather than relying purely on the last LSTM state
- **Dual attention** — channel attention answers *what*, temporal attention answers *when*

---

## 🤖 Model Performance Comparison

### Deep Learning Models (Full Supervised Training)

| Model | Parameters | Test Acc | F1 | Train Time | Tier |
|---|---|---|---|---|---|
| LSTM | 135,429 | 98.93% | 0.9893 | 99s | 🟡 Gateway |
| BiLSTM | 336,133 | 98.75% | 0.9876 | 103s | 🔴 Cloud |
| CNN_1D | 54,469 | 99.22% | 0.9922 | 102s | 🟢 Edge |
| CNN_BiLSTM | 133,253 | 99.00% | 0.9900 | 119s | 🟢 Edge |
| **CNN_BiLSTM_Attention ⭐** | **137,622** | **99.31%** | **0.9931** | **133s** | **🟢 Edge** |
| TransformerIDS | 38,085 | 96.91% | 0.9686 | 109s | 🟢 Edge |

### ML Baselines

| Model | Test Acc | F1 | Train Time | Size | Inference |
|---|---|---|---|---|---|
| Random Forest | 99.29% | 0.9930 | 6.9s | 21.5 MB | 113µs |
| **XGBoost** | **99.68%** | **0.9968** | **714s** | **1.35 MB** | **89µs** |

### Per-Class F1 — CNN-BiLSTMAttention

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Normal | 1.00 | 1.00 | 1.00 |
| Grayhole | 0.95 | 0.97 | 0.96 |
| Blackhole | 0.96 | 0.95 | 0.95 |
| TDMA | 0.94 | 0.98 | 0.96 |
| Flooding | 0.96 | 0.94 | 0.95 |

---

## 🔬 Few-Shot Learning

All few-shot methods operate on **N-way K-shot episodic evaluation** (500 episodes, K ∈ {1, 3, 5, 10, 20}) using the **frozen CNN-BiLSTMAttention encoder** as the feature extractor. This simulates real deployment where new attack types may have very few labelled examples.

### Classical Baselines (6 methods)

| Method | 1-shot | 3-shot | 5-shot | 10-shot | 20-shot |
|---|---|---|---|---|---|
| Prototypical | 0.9233 | 0.9555 | 0.9568 | 0.9624 | 0.9642 |
| Matching Network | 0.9192 | 0.9501 | 0.9534 | 0.9562 | 0.9610 |
| Relation Network | 0.9108 | 0.9430 | 0.9511 | 0.9580 | 0.9605 |
| Siamese | 0.9014 | 0.9350 | 0.9440 | 0.9521 | 0.9580 |
| Inductive Transfer | 0.9252 | 0.9533 | 0.9572 | 0.9638 | 0.9656 |
| APFP 2024 (baseline) | 0.9249 | 0.9404 | 0.9464 | 0.9534 | 0.9598 |

---

### 🆕 Novel Method 1 — AP++ (Adaptive Prototype++)

AP++ is our improved meta-learner over the APFP 2024 baseline. It wraps 7 enhancements around standard prototype construction:

| # | Enhancement | Description |
|---|---|---|
| 1 | **Dual-metric similarity** | Cosine + Euclidean blended (α=0.6 cosine) |
| 2 | **Adaptive temperature** | τ = clip(τ_base / intra_std, τ_base×0.5, τ_base×3) |
| 3 | **Outlier filtering** | Remove support points >2σ from class centroid |
| 4 | **EM prototype refinement** | 4-iteration expectation-maximisation on soft assignments |
| 5 | **Inter-class repulsion** | Pushes prototypes away from nearest non-target centroid |
| 6 | **Mahalanobis normalisation** | Feature whitening before distance computation |
| 7 | **Episode LDA projection** | Linear Discriminant Analysis projection per episode |

**AP++ results vs baselines:**

| K | Prototypical | APFP 2024 | **AP++ (Ours)** | Δ vs APFP |
|---|---|---|---|---|
| 1 | 0.9233 | 0.9249 | **0.9330** | +0.81% |
| 3 | 0.9555 | 0.9404 | **0.9600** | +1.96% |
| 5 | 0.9568 | 0.9464 | **0.9623** | +1.59% |
| 10 | 0.9624 | 0.9534 | **0.9630** | +0.96% |
| 20 | 0.9642 | 0.9598 | **0.9625** | +0.27% |

---

### 🆕 Novel Method 2 — CTPN (Contrastive Transformer Prototype Network)

CTPN stacks three purpose-built modules on top of the frozen encoder:

```
Frozen Encoder (128-D)
       │
       ▼
┌──────────────────────────────────┐
│  Projection Head (3-layer MLP)   │
│  128 → 256 → 128 + skip → L2    │  Contrastive embedding space
└──────────┬───────────────────────┘
           │
      ┌────▼────────────────────────┐
      │  Support Attention          │
      │  Aggregator (SAA)           │  Multi-head self-attention over K
      │  K support embeddings       │  support samples → weighted prototype
      └────┬────────────────────────┘
           │
      ┌────▼────────────────────────┐
      │  CAPR — Cross-Attention     │
      │  Prototype Refinement       │  4 heads, 4 iterations; query
      │  (iterative)                │  attends to refined prototype space
      └────┬────────────────────────┘
           │
      Cosine similarity → episode classification
```

**Training regime:**
1. SupCon loss fine-tuning (60 epochs, batch=512) on the full WSN-DS train set
2. Multi-K meta-training over K ∈ {1,3,5,10,20} (600 episodes per K)
3. Saved to `runs/fewshot-ctpn-v4/results/ctpn_weights_final.pth`

**CTPN results (full benchmark):**

| K | Prototypical | APFP 2024 | AP++ | **CTPN (Ours)** | AUC |
|---|---|---|---|---|---|
| 1 | 0.9186 | 0.9239 | 0.9428 | **0.9531** | 0.9934 |
| 3 | 0.9555 | 0.9447 | 0.9599 | **0.9700** | 0.9970 |
| 5 | 0.9574 | 0.9479 | 0.9612 | **0.9688** | 0.9970 |
| 10 | 0.9588 | 0.9522 | 0.9613 | **0.9716** | 0.9974 |
| 20 | 0.9635 | 0.9566 | 0.9624 | **0.9694** | 0.9974 |

> CTPN outperforms all baselines at every K value, with the largest gains at low-K (1-shot: +3.45% over APFP, +1.03% over AP++).

---

## 🔭 Novel Contributions C1–C6

### C1 — MC Dropout Uncertainty Quantification

Detects zero-day and novel attacks that fall outside all known training classes.

```python
model.train()   # keep dropout ACTIVE during inference
preds = [model(x) for _ in range(30)]   # 30 stochastic forward passes
entropy    = -Σ p_i × log(p_i)
confidence = 1 - (entropy / log(n_classes))
# Flag as novel attack if confidence < 0.70
```

| Metric | Value |
|---|---|
| Overall accuracy | 99.35% |
| High-confidence sample accuracy | **99.84%** |
| Novel attack flags issued | 161 / 10,000 (1.6%) |

---

### C2 — Chunk-Based Concept Drift Detection

Monitors IDS accuracy over streaming data windows; triggers adaptive retraining when attack patterns evolve.

```
Stream → [Chunk 1][Chunk 2]...[Chunk 20]
          monitor accuracy per chunk
          if Δacc > 3% vs baseline → DRIFT DETECTED → retrain adaptive model
```

| Metric | Value |
|---|---|
| Drift events detected | 1 (at chunk 16) |
| Accuracy at drift: before → after | 99.1% → 96.0% |
| Static avg accuracy (no adaptation) | 97.31% |
| Adaptive avg accuracy (with retraining) | **98.16%** |
| Improvement | **+0.85%** |

---

### C3 — Energy-Complexity Trade-Off Framework

A quantitative framework to assign each model to a deployment tier:

```
Energy Score = (Accuracy × F1) / (log(1+inf_time) × log(1+size) × log(1+FLOPs))
```

| Score Range | Tier | Target Hardware |
|---|---|---|
| > 0.50 | 🟢 Edge | Sensor nodes, microcontrollers |
| 0.20 – 0.50 | 🟡 Gateway | Raspberry Pi, edge servers |
| < 0.20 | 🔴 Cloud | Central servers |

| Model | Acc | Inference | Size | Score | Tier |
|---|---|---|---|---|---|
| CNN_BiLSTM_Attention | 99.31% | 10.44s (batched) | ~538KB | 0.51 | 🟢 Edge |
| LSTM | 98.93% | 6.12s | ~523KB | 0.44 | 🟡 Gateway |
| XGBoost | 99.68% | 89µs | 1.35MB | 0.011 | 🔴 Cloud |
| Random Forest | 99.29% | 113µs | 21.5MB | 0.004 | 🔴 Cloud |

---

### C4 — LIME Local Explainability

Per-sample local explanations — answers *why THIS specific packet was classified as Blackhole* rather than providing averaged global importance only.

> Unlike SHAP (global average across dataset), LIME perturbs the individual
> input instance and fits a local linear surrogate. This is essential for
> operational alert triage where analysts must justify each flagged packet.

---

### C5 — FGSM Adversarial Robustness Testing

```
x_adv = x + ε × sign(∇ₓ Loss(x, y_true))
```

Tests model resistance against crafted evasion attacks at ε ∈ {0.0, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3}. Accuracy degradation curves identify the model's robustness boundary — the point at which an attacker can reliably evade detection.

---

### C6 — Attention Weight Visualisation

Extracts and plots the temporal attention weights α_t per attack class, revealing which feature timesteps the model focuses on. This provides mechanistic interpretability — proving the model learned genuine attack signatures rather than spurious statistical shortcuts.

---

## 🚀 How to Run

### Prerequisites
```bash
pip install torch numpy pandas scikit-learn xgboost shap lime matplotlib seaborn
```

### Step 1: Main Pipeline (all 8 models + C1/C2/C3)
```bash
nohup python project/Scripts/wsn_dl_research.py > runs/console.log 2>&1 &
tail -f runs/console.log
```

### Step 2: Novel Contributions C4/C5/C6
```bash
# Run AFTER main pipeline completes — requires saved model checkpoint
python project/Scripts/wsn_novel_c4c5c6.py
```

### Step 3: Few-Shot Benchmarks
```bash
python project/Scripts/wsn_few_shot_all.py        # all 6 baseline methods
python project/Scripts/wsn_ap_pp_v4.py            # AP++ novel method
```

### Step 4: CTPN Training + Evaluation
```bash
python project/Scripts/ctpn_wsn_v4.py             # train CTPN
python project/Scripts/ctpn_eval_v4.py             # evaluate & generate all plots
```

### Monitor Running Jobs
```bash
tail -f runs/console.log
ps aux | grep wsn | grep -v grep && echo "⏳ Running" || echo "✅ Done"
```

---

## 📁 Repository Structure

```
Iot/
├── project/
│   ├── Scripts/
│   │   ├── wsn_dl_research.py          # Main DL pipeline + C1/C2/C3
│   │   ├── wsn_research.py             # Earlier baseline (ADWIN, SHAP)
│   │   ├── wsn_few_shot.py             # Prototypical episodic eval
│   │   ├── wsn_few_shot_all.py         # 6-method few-shot benchmark
│   │   ├── wsn_ap_pp.py                # AP++ v3 novel method
│   │   ├── wsn_ap_pp_v4.py             # AP++ v4 novel method (latest)
│   │   ├── wsn_novel_c4c5c6.py         # C4 LIME + C5 FGSM + C6 Attn
│   │   ├── ctpn_wsn.py                 # CTPN training
│   │   ├── ctpn_wsn_v4.py              # CTPN v4 training (latest)
│   │   └── ctpn_eval_v4.py             # CTPN full evaluation suite
│   └── data/
│       └── raw/
│           └── WSN-DS.csv
│
├── runs/
│   └── 2026_03_11__22_50_37/
│       ├── plots/
│       │   ├── class_distribution.png
│       │   ├── training_history.png
│       │   ├── model_comparison.png
│       │   ├── roc_curves.png
│       │   ├── confusion_matrix_*.png       (8 matrices)
│       │   ├── C1_uncertainty.png
│       │   ├── C2_concept_drift.png
│       │   ├── C3_energy_tradeoff.png
│       │   ├── C4_lime_explanations.png
│       │   ├── C5_adversarial_robustness.png
│       │   └── C6_attention_visualization.png
│       ├── models/
│       │   ├── CNN_BiLSTM_Attention.pth
│       │   ├── LSTM.pth
│       │   ├── BiLSTM.pth
│       │   └── scaler.pkl
│       └── results/
│           ├── final_summary.json
│           ├── C1_uncertainty.json
│           ├── C2_drift.json
│           ├── C3_energy.csv
│           ├── C4_lime.json
│           ├── C5_adversarial.json
│           └── C6_attention.json
│
└── fewshot/
    └── fewshot-ctpn-v4/
        ├── evaluation/
        │   ├── confusion_matrix.png
        │   ├── classification_report.png
        │   ├── tsne_embeddings.png
        │   ├── prototype_heatmap.png
        │   ├── capr_attention.png
        │   ├── episode_stability.png
        │   └── calibration.png
        └── results/
            └── ctpn_weights_final.pth
```

---

## 📖 References

1. Almomani, I. et al. — *WSN-DS: A Dataset for Intrusion Detection Systems in WSN* (2016)
2. Hochreiter & Schmidhuber — *Long Short-Term Memory*, Neural Computation (1997)
3. Hu et al. — *Squeeze-and-Excitation Networks*, CVPR (2018)
4. Bahdanau et al. — *Neural Machine Translation by Jointly Learning to Align and Translate*, ICLR (2015)
5. Vinyals et al. — *Matching Networks for One Shot Learning*, NeurIPS (2016)
6. Snell et al. — *Prototypical Networks for Few-Shot Learning*, NeurIPS (2017)
7. Sung et al. — *Learning to Compare: Relation Network for Few-Shot Learning*, CVPR (2018)
8. Koch et al. — *Siamese Neural Networks for One-shot Image Recognition*, ICML (2015)
9. Khosla et al. — *Supervised Contrastive Learning*, NeurIPS (2020)
10. Gal & Ghahramani — *Dropout as a Bayesian Approximation*, ICML (2016)
11. Goodfellow et al. — *Explaining and Harnessing Adversarial Examples*, ICLR (2015)
12. Ribeiro et al. — *"Why Should I Trust You?": LIME*, KDD (2016)
13. Lundberg & Lee — *A Unified Approach to Interpreting Model Predictions (SHAP)*, NeurIPS (2017)
14. Chawla et al. — *SMOTE: Synthetic Minority Over-sampling Technique*, JAIR (2002)

---

## 👥 Authors

**Amlan Sarkar** — B.Tech, Computer Science & Engineering

**Ankit** — B.Tech, Computer Science & Engineering

*Deep Learning · IoT Security · Few-Shot Learning · Explainable AI*

---

*© 2026 — WSN-DS IDS Research Project · MIT License*
