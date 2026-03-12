# 🛡️ WSN-DS Intrusion Detection System
### Deep Learning-Based Multi-Attack Classification for Wireless Sensor Networks

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10-red?logo=pytorch)
![CUDA](https://img.shields.io/badge/CUDA-12.8-green?logo=nvidia)
![Accuracy](https://img.shields.io/badge/Best%20Accuracy-99.68%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Overview

This project presents a **comprehensive deep learning pipeline** for intrusion detection in Wireless Sensor Networks (WSN) using the **WSN-DS dataset**. Beyond classification accuracy, the system addresses five critical dimensions for real-world deployment:

| Dimension | Method | Result |
|---|---|---|
| 🎯 **Accuracy** | 8 models benchmarked | Up to **99.68%** |
| 🔍 **Trustworthiness** | MC Dropout Uncertainty | 99.84% on high-confidence samples |
| 🔄 **Adaptability** | Concept Drift Detection | +0.85% improvement on drifted data |
| ⚡ **Deployability** | Energy-Complexity Framework | Edge / Gateway / Cloud tiering |
| 🧠 **Explainability** | SHAP + LIME + Attention Viz | Per-class local & global explanations |
| 🛡️ **Robustness** | FGSM Adversarial Testing | Vulnerability boundary quantified |

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
│                   │  Before → After shown below
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

---

### Step 1 — Data Cleaning & Label Encoding

**What was done:**
- Verified zero missing values across all 374,661 rows
- Selected only numeric columns
- Applied `LabelEncoder` to convert string class names to integers:

```
Blackhole → 0 | Flooding → 1 | Grayhole → 2 | Normal → 3 | TDMA → 4
```

**Why label encoding (not one-hot):**
> PyTorch's `CrossEntropyLoss` expects integer class indices directly — it applies
> softmax + log internally. One-hot encoding would be redundant and wasteful.

---

### Step 2 — Stratified Train/Test Split (80/20)

**What was done:**
- Split: 299,728 train / 74,933 test
- `stratify=y` ensures class proportions are identical in both splits

| Class | Full Dataset | Train Set | Test Set |
|---|---|---|---|
| Normal | 90.77% | 90.77% | 90.77% |
| Grayhole | 3.90% | 3.90% | 3.90% |
| Blackhole | 2.68% | 2.68% | 2.68% |
| TDMA | 1.77% | 1.77% | 1.77% |
| Flooding | 0.88% | 0.88% | 0.88% |

> ⚠️ **Critical:** SMOTE is applied ONLY to the **train set AFTER splitting**.
> Applying SMOTE before splitting = **data leakage** — synthetic samples derived
> from test data contaminate training → artificially inflated evaluation scores.

---

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
| `inter_A_x_B` | top_var_feat_1 × top_var_feat_2 | Joint effect of the two most variable features |
| `inter_B_x_C` | top_var_feat_2 × top_var_feat_3 | Cross-feature anomaly signal |
| `inter_C_x_D` | top_var_feat_3 × top_var_feat_4 | Non-linear feature combinations |

> Top 4 features by **variance** are selected — these vary the most between Normal and attack traffic and therefore carry the most discriminative signal.

#### Ratio Features (+5)

| New Feature | Formula | Why useful |
|---|---|---|
| `ratio_feat_i` | feat_i / (row_sum + 1e-10) | Relative contribution of each feature — attack traffic has abnormal ratios (e.g. DATA_R >> DATA_S in Blackhole) |

**Why do feature engineering if neural networks learn features automatically?**
> Neural networks can learn features — but need data and time. Statistical features
> encode **domain knowledge**: Flooding has distinctly different row skewness than
> Normal. Pre-computing this:
> - Speeds up convergence
> - Helps minority classes with few real samples
> - Acts as an inductive bias, guiding the model toward known discriminative patterns

---

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

**Why not random oversampling (duplication)?**
> Duplication → model memorises exact copies → fails on slight variations.
> SMOTE generates **new points between real ones** → model learns the actual
> decision boundary shape of each minority class.

**Why not class weights?**
> Class weights change the loss penalty but the model still sees 272,052 Normal
> samples vs 2,650 Flooding samples. It thoroughly memorises Normal but barely
> learns Flooding's boundary. SMOTE physically provides more Flooding examples.

**Why cap at 10,000 (not match Normal's 272,052)?**
> Fully equalising Flooding would need ~270,000 synthetic samples — 100× more
> synthetics than real ones. The synthetic distribution would drift far from
> reality and introduce noise. Capping at 10,000 improves minority learning
> without flooding the training set with artificial data.

---

### Step 5 — StandardScaler Normalisation

```python
scaler = StandardScaler()
scaler.fit(X_train_balanced)       # learn mean & std from TRAIN ONLY
X_train_sc = scaler.transform(X_train_balanced)
X_test_sc  = scaler.transform(X_test)    # same transform applied to test
```

**Result:** Every feature → mean = 0, std = 1

**Why normalise:**

| Feature | Raw Range | Problem Without Scaling |
|---|---|---|
| `dist_BS` | 0 – 200 m | Large values dominate gradients |
| `DATA_S` | 0 – 10,000 | Overwhelms binary features |
| `is_CH` | 0 or 1 | Gets ignored by optimiser |

Without normalisation, the model learns "high DATA_S = attack" because of scale,
ignoring small-range but equally important binary/distance features entirely.

**Why fit on train only?**
> Fitting on the full dataset uses test set mean and std during training —
> **data leakage**. In production, new packets are scaled using training statistics.
> We simulate this exactly: `fit()` on train, `transform()` on both.

> 💾 Scaler saved to `models/scaler.pkl` for reproducible inference on new data.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        WSN-DS Dataset                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │   Feature Engineering   │  18 → 33 features
              │  (stats + interactions) │  skew, kurtosis, ratios
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │     Manual SMOTE        │  Minority class synthesis
              │  (max 10k per class)    │  k=5 nearest neighbours
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │    StandardScaler       │  mean=0, std=1 per feature
              │    Normalisation        │  fit on train only
              └────────────┬────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
┌────────▼───────┐ ┌───────▼──────┐ ┌───────▼──────┐
│  Deep Learning │ │  Deep Learn. │ │  ML Baseline │
│  LSTM          │ │  CNN_1D      │ │  RandomForest│
│  BiLSTM        │ │  CNN_BiLSTM  │ │  XGBoost     │
│  Transformer   │ │  CNN_BiLSTM  │ └──────────────┘
│                │ │  +Attention⭐│
└────────────────┘ └───────┬──────┘
                           │
         ┌─────────────────┼──────────────────────┐
         │                 │                       │
┌────────▼───────┐ ┌───────▼──────┐ ┌─────────────▼──────────┐
│  C1: MC        │ │  C2: Concept │ │  C3: Energy-Complexity  │
│  Dropout       │ │  Drift       │ │  Trade-off Framework    │
│  Uncertainty   │ │  Detection   │ │  (Edge/Gateway/Cloud)   │
└────────────────┘ └──────────────┘ └────────────────────────┘
         │                 │
┌────────▼───────┐ ┌───────▼──────┐ ┌──────────────┐
│  C4: LIME      │ │  C5: FGSM    │ │  C6: Attn.   │
│  Local XAI     │ │  Adversarial │ │  Visualization│
└────────────────┘ └──────────────┘ └──────────────┘
```

---

## 🤖 Models

### Deep Learning Models

| Model | Parameters | Test Acc | F1 | Train Time | Deployment |
|---|---|---|---|---|---|
| LSTM | 135,429 | 98.93% | 0.9893 | 99s | 🟡 Gateway |
| BiLSTM | 336,133 | 98.75% | 0.9876 | 103s | 🔴 Cloud |
| CNN_1D | 54,469 | 99.22% | 0.9922 | 102s | 🟢 Edge |
| CNN_BiLSTM | 133,253 | 99.00% | 0.9900 | 119s | 🟢 Edge |
| **CNN_BiLSTM_Attention** ⭐ | **137,622** | **99.31%** | **0.9931** | **133s** | **🟢 Edge** |
| Transformer_IDS | 38,085 | 96.91% | 0.9686 | 109s | 🟢 Edge |

### ML Baselines

| Model | Test Acc | F1 | Train Time | Size | Inference |
|---|---|---|---|---|---|
| Random Forest | 99.29% | 0.9930 | 6.9s | 21.5 MB | 113µs |
| **XGBoost** | **99.68%** | **0.9968** | **714s** | **1.35 MB** | **89µs** |

---

## ⭐ Novel Architecture — CNN-BiLSTM with Dual Attention

```
Input (B, F, 1)
      │
      ▼
┌─────────────────────┐
│   Conv1D Block      │  64 → 128 filters, BatchNorm, MaxPool
│   (Local Patterns)  │  Learns which feature groups co-activate
└──────────┬──────────┘
           │
      ┌────▼──────────────────┐
      │  Channel Attention     │  Squeeze-and-Excitation
      │  (Feature Reweighting) │  Which feature channels matter most?
      └────┬──────────────────┘
           │
      ┌────▼──────────────────┐
      │  Bidirectional LSTM   │  Forward + backward temporal context
      │  (Temporal Modelling)  │  64 units × 2 directions = 128 hidden
      └────┬──────────────────┘
           │
      ┌────▼──────────────────┐
      │  Temporal Attention   │  score_t = tanh(W · h_t)
      │  (Focus Mechanism)    │  context = Σ softmax(score_t) × h_t
      └────┬──────────────────┘
           │
      ┌────▼──────────────────┐
      │  Classifier Head      │  FC(128→64) → Dropout → FC(64→5)
      └───────────────────────┘
```

**Key design decisions:**
- **Channel Attention** — not all features matter equally: routing features dominate Blackhole detection; packet rate features dominate Flooding
- **Temporal Attention over last hidden state** — avoids recency bias; model learns *which time steps* carry the attack signature
- **Dual attention** — channel attention handles *what*, temporal attention handles *when*

---

## 🔬 Novel Contributions

### C1 — MC Dropout Uncertainty Quantification
Detects zero-day / novel attacks the model has never seen.

```python
model.train()   # keep dropout ON during inference
preds = [model(x) for _ in range(30)]   # 30 stochastic forward passes
entropy    = -Σ p_i × log(p_i)
confidence = 1 - (entropy / log(n_classes))
# Flag as novel if confidence < 0.70
```

| Metric | Value |
|---|---|
| Overall accuracy | 99.35% |
| High-confidence accuracy | **99.84%** |
| Novel attacks flagged | 161 (1.6%) |

---

### C2 — Chunk-Based Concept Drift Detection
Keeps IDS accurate as attack patterns evolve over time.

```
Stream → [Chunk 1][Chunk 2]...[Chunk 20]
          monitor accuracy per chunk
          if drop > 3% vs baseline → DRIFT → retrain adaptive model
```

| Metric | Value |
|---|---|
| Drift events detected | 1 |
| Static avg accuracy | 97.31% |
| Adaptive avg accuracy | **98.16%** |
| Improvement | **+0.85%** |

---

### C3 — Energy-Complexity Trade-off Framework

```
Energy Score = (Accuracy × F1) / (log(1+inf_time) × log(1+size) × log(1+FLOPs))
```

| Score | Tier | Suitable For |
|---|---|---|
| > 0.50 | 🟢 Edge | Sensor nodes, microcontrollers |
| 0.20 – 0.50 | 🟡 Gateway | Raspberry Pi, edge servers |
| < 0.20 | 🔴 Cloud | Central servers |

| Model | Score | Tier |
|---|---|---|
| CNN_BiLSTM_Attention | 0.51 | 🟢 Edge |
| LSTM | 0.44 | 🟡 Gateway |
| XGBoost | 0.011 | 🔴 Cloud |
| Random Forest | 0.004 | 🔴 Cloud |

---

### C4 — LIME Local Explainability
Per-sample local explanations — answers *why THIS specific packet was flagged* as an attack.

> Unlike SHAP (global average), LIME perturbs the individual input and fits a
> local linear surrogate. Essential for operational alert investigation.

---

### C5 — FGSM Adversarial Robustness

```
x_adversarial = x + ε × sign(∇ₓ Loss(x, y_true))
```

Tests model resistance against crafted evasion attacks. Quantifies accuracy
degradation at ε = 0.0 → 0.3, establishing the robustness boundary.

---

### C6 — Attention Weight Visualization
Reveals which temporal positions the model focuses on per attack class —
proving the model learned attack-specific signatures, not statistical shortcuts.

---

## 🚀 How to Run

### Prerequisites
```bash
pip install torch numpy pandas scikit-learn xgboost shap lime matplotlib seaborn
```

### Main Pipeline (all 8 models + C1/C2/C3)
```bash
nohup python project/Scripts/wsn_dl_research.py > runs/console.log 2>&1 &
tail -f runs/console.log
```

### Novel Contributions C4 / C5 / C6
```bash
# Run AFTER main pipeline completes
python project/Scripts/wsn_novel_c4c5c6.py
```

### Monitor Progress
```bash
tail -f runs/console.log
ps aux | grep wsn | grep -v grep && echo "⏳ Running" || echo "✅ Done"
```

---

## 📁 Output Structure

```
runs/
└── 2026_03_11__22_50_37/
    ├── plots/
    │   ├── class_distribution.png
    │   ├── training_history.png
    │   ├── model_comparison.png
    │   ├── roc_curves.png
    │   ├── confusion_matrix_*.png        (8 matrices)
    │   ├── C1_uncertainty.png
    │   ├── C2_concept_drift.png
    │   ├── C3_energy_tradeoff.png
    │   ├── C4_lime_explanations.png
    │   ├── C5_adversarial_robustness.png
    │   ├── C6_attention_visualization.png
    │   └── C6_attention_comparison.png
    ├── models/
    │   ├── CNN_BiLSTM_Attention.pth
    │   ├── LSTM.pth  /  BiLSTM.pth  /  ...
    │   └── scaler.pkl
    └── results/
        ├── final_summary.json
        ├── C1_uncertainty.json
        ├── C2_drift.json
        ├── C3_energy.csv
        ├── C4_lime.json
        ├── C5_adversarial.json
        └── C6_attention.json
```

---


## 📂 Project Structure

```
Iot/
├── project/
│   ├── Scripts/
│   │   ├── wsn_dl_research.py        # Main pipeline
│   │   └── wsn_novel_c4c5c6.py      # C4 LIME + C5 FGSM + C6 Attention
│   └── data/
│       └── raw/
│           └── WSN-DS.csv
├── runs/                             # All outputs auto-saved here
├── WSN_IDS_Defense_Guide.md         # Complete professor Q&A guide
└── README.md
```

---

## 📖 References

1. Almomani, I. et al. — *WSN-DS: A Dataset for Intrusion Detection Systems in WSN* (2016)
2. Hochreiter & Schmidhuber — *Long Short-Term Memory*, Neural Computation (1997)
3. Hu et al. — *Squeeze-and-Excitation Networks*, CVPR (2018)
4. Bahdanau et al. — *Neural Machine Translation by Jointly Learning to Align and Translate*, ICLR (2015)
5. Gal & Ghahramani — *Dropout as a Bayesian Approximation*, ICML (2016)
6. Goodfellow et al. — *Explaining and Harnessing Adversarial Examples*, ICLR (2015)
7. Ribeiro et al. — *"Why Should I Trust You?": LIME*, KDD (2016)
8. Lundberg & Lee — *A Unified Approach to Interpreting Model Predictions (SHAP)*, NeurIPS (2017)
9. Chawla et al. — *SMOTE: Synthetic Minority Over-sampling Technique*, JAIR (2002)

---

## 👥 Authors

**Amlan Sarkar**
B.Tech — Computer Science & Engineering


**Ankit**
B.Tech — Computer Science & Engineering


*Deep Learning · IoT Security · Explainable AI*

---
*© 2026 — WSN-DS IDS Research Project*
