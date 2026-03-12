# 🛡️ WSN-DS Intrusion Detection System
### Deep Learning-Based Multi-Attack Classification for Wireless Sensor Networks

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.10-red?logo=pytorch)
![CUDA](https://img.shields.io/badge/CUDA-12.8-green?logo=nvidia)
![Accuracy](https://img.shields.io/badge/Best%20Accuracy-99.68%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

##  Overview

This project presents a **comprehensive deep learning pipeline** for intrusion detection in Wireless Sensor Networks (WSN) using the **WSN-DS dataset**. Beyond classification accuracy, the system addresses five critical dimensions for real-world deployment:

| Dimension | Method | Result |
|---|---|---|
|  **Accuracy** | 8 models benchmarked | Up to **99.68%** |
|  **Trustworthiness** | MC Dropout Uncertainty | 99.84% on high-confidence samples |
|  **Adaptability** | Concept Drift Detection | +0.85% improvement on drifted data |
|  **Deployability** | Energy-Complexity Framework | Edge / Gateway / Cloud tiering |
|  **Explainability** | SHAP + LIME + Attention Viz | Per-class local & global explanations |
|  **Robustness** | FGSM Adversarial Testing | Vulnerability boundary quantified |

---

##  Dataset — WSN-DS

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
| **Normal** | Legitimate WSN traffic | All features normal range |
| **Blackhole** | Malicious node drops ALL packets it receives | DATA_R high, DATA_S→0 |
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
│  Feature          │  18 → 33 features (only on train, applied to test)
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
│  StandardScaler   │  fit on train, transform both train & test
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
- Selected only numeric columns (dropped non-numeric if any)
- Applied `LabelEncoder` to convert string class names to integers:

```
Blackhole → 0
Flooding  → 1
Grayhole  → 2
Normal    → 3
TDMA      → 4
```

**Why label encoding (not one-hot):**
> Neural networks with `CrossEntropyLoss` expect integer class indices, not one-hot vectors. One-hot encoding would require softmax output to match a 5-dimensional target — CrossEntropyLoss handles this internally and more efficiently.

---

### Step 2 — Stratified Train/Test Split (80/20)

**What was done:**
- Split dataset: 299,728 train / 74,933 test
- `stratify=y` ensures proportions match original distribution

**Before split → After split (class proportions preserved):**

| Class | Full Dataset | Train Set | Test Set |
|---|---|---|---|
| Normal | 90.77% | 90.77% | 90.77% |
| Grayhole | 3.90% | 3.90% | 3.90% |
| Blackhole | 2.68% | 2.68% | 2.68% |
| TDMA | 1.77% | 1.77% | 1.77% |
| Flooding | 0.88% | 0.88% | 0.88% |

**Why stratification:**
> Without stratification, a random split might place all 3,312 Flooding samples in train and none in test — making it impossible to evaluate Flooding detection. With stratification, each split is a faithful mini-copy of the original dataset.

> ⚠️ **Important:** SMOTE is applied ONLY to the train set AFTER splitting.
> Applying SMOTE before splitting would cause **data leakage** — synthetic samples derived from test data would contaminate training, making evaluation results artificially inflated.

---

### Step 3 — Feature Engineering (18 → 33 features)

**What was done:**
Created 15 additional features from the original 18:

#### Statistical Features (+7)
| New Feature | Formula | Why useful |
|---|---|---|
| `stat_mean` | mean of all 18 features per row | Global traffic level indicator |
| `stat_std` | std of all 18 features per row | Traffic variability — attacks cause unusual variance |
| `stat_max` | max feature value per row | Detects extreme outlier behaviour |
| `stat_min` | min feature value per row | Detects dropped-to-zero features (Blackhole) |
| `stat_range` | max − min per row | Overall feature spread |
| `stat_skew` | skewness per row | Flooding has highly skewed packet distributions |
| `stat_kurt` | kurtosis per row | Detects heavy-tailed attack distributions |

#### Interaction Features (+3)
| New Feature | Formula | Why useful |
|---|---|---|
| `inter_A_x_B` | top_var_feat_1 × top_var_feat_2 | Captures joint effect of two most variable features |
| `inter_B_x_C` | top_var_feat_2 × top_var_feat_3 | Cross-feature anomaly detection |
| `inter_C_x_D` | top_var_feat_3 × top_var_feat_4 | Non-linear feature combinations |

> Top 4 features by variance are selected to compute interactions — these are the features that change the most between Normal and attack traffic.

#### Ratio Features (+5)
| New Feature | Formula | Why useful |
|---|---|---|
| `ratio_feat_i` | feat_i / (row_sum + 1e-10) | Normalises each feature relative to total row magnitude — attack traffic has abnormal ratios |

**Why feature engineering if neural networks auto-learn features?**
> Neural networks can learn features — but need data and time. Statistical features encode **domain knowledge**: a Flooding attack has distinctly different row-level skewness than Normal traffic. Pre-computing these:
> - Improves convergence speed
> - Helps minority classes with few samples
> - Acts as an inductive bias — tells the model these relationships matter

---

### Step 4 — SMOTE Balancing

#### Before SMOTE (raw train split):
```
Class       Samples    % of train
──────────────────────────────────
Normal      272,052    90.77%   ← completely dominates
Grayhole     11,677     3.90%
Blackhole     8,039     2.68%
TDMA          5,310     1.77%
Flooding      2,650     0.88%   ← only 2,650 samples!
```

#### After SMOTE (balanced train set):
```
Class       Samples    Change
──────────────────────────────
Normal      272,052    unchanged (already large)
Grayhole     11,677    unchanged (already at threshold)
Blackhole    10,000    +1,961 synthetic samples
TDMA         10,000    +4,690 synthetic samples
Flooding     10,000    +7,350 synthetic samples
──────────────────────────────
Total        313,729   (+14,001 synthetic samples added)
```

#### How SMOTE Works:

```
For each minority sample x_i:
  1. Find k=5 nearest neighbours in same class
  2. Pick one neighbour x_nn at random
  3. Generate synthetic point:
     x_synthetic = x_i + λ × (x_nn - x_i)   where λ ∈ [0,1] random
  4. Repeat until target count reached
```

**Visually:**
```
  x_i  ●────────────●  x_nn
           ◆  ◆  ◆         ← synthetic points placed between real ones
```

**Why not just duplicate (random oversampling)?**
> Duplication copies exact samples → model memorises them → fails on any variation.
> SMOTE generates **new points between real ones** → model learns the boundary shape.

**Why not class weights instead?**
> Class weights change the loss penalty but the model still sees 272,052 Normal
> samples vs 2,650 Flooding samples. It memorises Normal thoroughly but barely
> learns Flooding's decision boundary. SMOTE physically gives the model more
> Flooding examples to learn from.

**Why cap at 10,000 (not equal to Normal's 272,052)?**
> Fully equalising would require 270,000 synthetic Flooding samples — 100× more
> synthetics than real ones. The synthetic distribution would drift far from reality.
> Capping at 10,000 adds enough to improve minority learning without overwhelming
> the training set with artificial data.

---

### Step 5 — StandardScaler Normalisation

**What was done:**
```python
scaler = StandardScaler()
scaler.fit(X_train)           # learn mean and std from TRAIN only
X_train_sc = scaler.transform(X_train)
X_test_sc  = scaler.transform(X_test)   # apply same transform to test
```

**Result:** Every feature has mean=0, std=1 across the training set.

**Why normalise:**
> Feature values span very different ranges:
> - `dist_BS` (distance): 0 – 200 metres
> - `DATA_S` (packet count): 0 – 10,000 packets
> - `is_CH` (binary): 0 or 1
>
> Without normalisation, gradient updates are dominated by large-magnitude features.
> The model would learn "high DATA_S = attack" simply because it has large values,
> ignoring the binary and small-range features entirely.

**Why fit on train only (not full dataset)?**
> Fitting the scaler on the full dataset would use test set statistics (mean, std)
> during training — this is **data leakage**. The test set must be completely unseen.
> In deployment, new data is scaled using the training set's mean and std — so we
> simulate that exact scenario.

**Scaler is saved** to `models/scaler.pkl` so it can be reloaded for inference on new data without recomputing.

---

##  System Architecture

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
│                │ │  Robustness  │ └──────────────┘
└────────────────┘ └──────────────┘
```

---

##  Models

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
│   (Local Patterns)  │  Learns local feature co-occurrences
└──────────┬──────────┘
           │
      ┌────▼──────────────────┐
      │  Channel Attention     │  Squeeze-and-Excitation
      │  (Feature Reweighting) │  Which channels matter most?
      └────┬──────────────────┘
           │
      ┌────▼──────────────────┐
      │  Bidirectional LSTM   │  Forward + backward context
      │  (Temporal Modelling)  │  64 units × 2 directions = 128
      └────┬──────────────────┘
           │
      ┌────▼──────────────────┐
      │  Temporal Attention   │  score_t = tanh(W·h_t)
      │  (Focus Mechanism)    │  context = Σ(softmax(score) × h)
      └────┬──────────────────┘
           │
      ┌────▼──────────────────┐
      │  Classifier           │  FC(128→64) → FC(64→5)
      └───────────────────────┘
```

---

## 🔬 Novel Contributions

### C1 — MC Dropout Uncertainty Quantification

| Metric | Value |
|---|---|
| Overall accuracy | 99.35% |
| High-confidence accuracy | **99.84%** |
| Novel attacks flagged | 161 (1.6%) |

### C2 — Concept Drift Detection

| Metric | Value |
|---|---|
| Drift events detected | 1 |
| Static avg accuracy | 97.31% |
| Adaptive avg accuracy | **98.16%** |
| Improvement | **+0.85%** |

### C3 — Energy-Complexity Trade-off

| Model | Score | Tier |
|---|---|---|
| CNN_BiLSTM_Attention | 0.51 | 🟢 Edge |
| LSTM | 0.44 | 🟡 Gateway |
| XGBoost | 0.011 | 🔴 Cloud |
| Random Forest | 0.004 | 🔴 Cloud |

### C4 — LIME Local Explainability
Per-sample local explanations for each attack class — answers *why THIS packet was flagged*.

### C5 — FGSM Adversarial Robustness
```
x_adversarial = x + ε × sign(∇ₓ Loss(x, y_true))
```
Quantifies accuracy degradation at ε = 0.0 → 0.3.

### C6 — Attention Weight Visualization
Reveals which temporal positions the model focuses on per attack type.

---

##  How to Run

### Prerequisites
```bash
pip install torch numpy pandas scikit-learn xgboost shap lime matplotlib seaborn
```

### Main Pipeline
```bash
nohup python project/Scripts/wsn_dl_research.py > runs/console.log 2>&1 &
tail -f runs/console.log
```

### Novel Contributions C4 / C5 / C6
```bash
python project/Scripts/wsn_novel_c4c5c6.py
```

### Monitor Progress
```bash
# Live log
tail -f runs/console.log

# Check if still running
ps aux | grep wsn | grep -v grep && echo "⏳ Running" || echo " Done"
```

---

##  Output Structure

```
runs/
└── 2026_03_11__22_50_37/
    ├── plots/
    │   ├── class_distribution.png
    │   ├── training_history.png
    │   ├── model_comparison.png
    │   ├── roc_curves.png
    │   ├── confusion_matrix_*.png     (8 matrices)
    │   ├── C1_uncertainty.png
    │   ├── C2_concept_drift.png
    │   ├── C3_energy_tradeoff.png
    │   ├── C4_lime_explanations.png
    │   ├── C5_adversarial_robustness.png
    │   ├── C6_attention_visualization.png
    │   └── C6_attention_comparison.png
    ├── models/
    │   ├── CNN_BiLSTM_Attention.pth
    │   ├── LSTM.pth
    │   ├── scaler.pkl
    │   └── ...
    └── results/
        ├── final_summary.json
        ├── C1_uncertainty.json
        ├── C2_drift.json
        ├── C3_energy.csv
        ├── C4_lime.json
        ├── C5_adversarial.json
        └── C6_attention.json
```



##  Project Structure

```
Iot/
├── project/
│   ├── Scripts/
│   │   ├── wsn_dl_research.py          # Main pipeline
│   │   └── wsn_novel_c4c5c6.py        # C4 LIME + C5 FGSM + C6 Attention
│   └── data/
│       └── raw/
│           └── WSN-DS.csv
├── runs/                               # All outputs auto-saved here
├── WSN_IDS_Defense_Guide.md           # Complete Q&A defense guide
└── README.md
```

---

##  References

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

## Author

**Amlan Sarkar**  **Ankit**
B.Tech — Computer Science & Engineering  
*Deep Learning · IoT Security · Explainable AI*

---
*© 2026 — WSN-DS IDS Research Project*
