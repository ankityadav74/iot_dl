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

## 📊 Dataset — WSN-DS

| Property | Value |
|---|---|
| Total records | 374,661 |
| Features (raw) | 18 |
| Features (engineered) | 33 |
| Classes | 5 |
| Train / Test split | 80% / 20% |

**Class Distribution:**

```
Normal    ████████████████████████████████████  90.77%  (340,066)
Grayhole  ██                                      3.90%  ( 14,596)
Blackhole █                                       2.68%  ( 10,049)
TDMA      ▌                                       1.77%  (  6,638)
Flooding  ▍                                       0.88%  (  3,312)
```

> **Why WSN-DS?** Unlike KDD99 or NSL-KDD, WSN-DS captures WSN-specific attacks
> (Grayhole, TDMA, Flooding) based on real LEACH routing protocol behaviour —
> making results directly applicable to IoT security deployments.

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
         │                 │                       │
┌────────▼───────┐ ┌───────▼──────┐
│  C4: LIME      │ │  C5: FGSM    │
│  Local XAI     │ │  Adversarial │
│                │ │  Robustness  │
└────────────────┘ └───────┬──────┘
                   ┌───────▼──────┐
                   │  C6: Attn.   │
                   │  Visualization│
                   └──────────────┘
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

The primary proposed architecture combines **local feature extraction**, **sequential modelling**, and **dual attention** in a single end-to-end pipeline:

```
Input (B, F, 1)
      │
      ▼
┌─────────────────────┐
│   Conv1D Block      │  64 → 128 filters, BatchNorm, MaxPool
│   (Local Patterns)  │  Learns attack signature fragments
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

**Key design decisions:**
- **Channel Attention:** Not all features matter equally per attack — routing features dominate Blackhole, packet rate features dominate Flooding
- **Temporal Attention over last-hidden-state:** Avoids recency bias; the model learns *which time steps contain the attack signature* rather than relying only on the last step
- **Dual attention:** Channel attention handles *what*, temporal attention handles *when*

---

## 🔬 Novel Contributions

### C1 — MC Dropout Uncertainty Quantification

Detects **zero-day / novel attacks** that the model has never seen during training.

```python
# During inference: keep dropout ON, run 30 forward passes
model.train()   # enables dropout
predictions = [model(x) for _ in range(30)]
entropy     = -Σ p_i × log(p_i)
confidence  = 1 - (entropy / log(n_classes))
# Flag as novel if confidence < 0.70
```

| Metric | Value |
|---|---|
| Overall accuracy | 99.35% |
| High-confidence accuracy | **99.84%** |
| Novel attacks flagged | 161 (1.6%) |

---

### C2 — Chunk-Based Concept Drift Detection

Keeps the IDS accurate as **attack patterns evolve over time**.

```
Stream → [Chunk 1][Chunk 2]...[Chunk 20]
          monitor accuracy per chunk
          if drop > 3% → DRIFT DETECTED → retrain
```

| Metric | Value |
|---|---|
| Drift events detected | 1 |
| Static model avg accuracy | 97.31% |
| Adaptive model avg accuracy | 98.16% |
| **Improvement** | **+0.85%** |

---

### C3 — Energy-Complexity Trade-off Framework

Assigns each model to a **deployment tier** based on accuracy vs resource cost:

```
Energy Score = (Accuracy × F1) / (log(1+inf_time) × log(1+size) × log(1+FLOPs))
```

| Score | Tier | Suitable For |
|---|---|---|
| > 0.50 | 🟢 Edge | Sensor nodes, microcontrollers |
| 0.20–0.50 | 🟡 Gateway | Raspberry Pi, edge servers |
| < 0.20 | 🔴 Cloud | Central servers, data centres |

**CNN_BiLSTM_Attention** is recommended for edge deployment:
- 9.9µs inference (9× faster than XGBoost at 89µs)
- 538KB model size (40× smaller than Random Forest at 21MB)

---

### C4 — LIME Local Explainability

Explains **why a specific packet was classified** as a particular attack.

> Unlike SHAP (global average importance), LIME perturbs individual inputs locally
> and fits a linear surrogate — telling a network admin exactly which features
> triggered THIS specific alert.

---

### C5 — Adversarial Robustness (FGSM)

Tests model resistance to **crafted evasion attacks**:

```
x_adversarial = x + ε × sign(∇ₓ Loss(x, y_true))
```

Quantifies the accuracy drop at increasing perturbation levels (ε = 0 to 0.3), establishing the model's robustness boundary and motivating future adversarial training.

---

### C6 — Attention Weight Visualization

Reveals **what the model focuses on per attack type** — proving it has learned attack-specific signatures rather than statistical shortcuts.

---

##  How to Run

### Prerequisites
```bash
pip install torch numpy pandas scikit-learn xgboost shap lime matplotlib seaborn
```

### Main Pipeline
```bash
# Full training pipeline (all 8 models + C1/C2/C3)
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
# Live log
tail -f runs/console.log

# Check if done
ps aux | grep wsn | grep -v grep && echo "⏳ Running" || echo " Done"

# Watch output files appear
watch -n 5 'ls -lh runs/$(ls runs/ -t | head -1)/plots/'
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
    ├── results/
    │   ├── final_summary.json
    │   ├── C1_uncertainty.json
    │   ├── C2_drift.json
    │   ├── C3_energy.csv
    │   ├── C4_lime.json
    │   ├── C5_adversarial.json
    │   └── C6_attention.json
    └── logs/
        └── training.log
```

---


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
└── README.md                          # This file
```

---

## 📖 References

1. Almomani, I. et al. — *WSN-DS: A Dataset for Intrusion Detection Systems in WSN* (2016)
2. Hochreiter & Schmidhuber — *Long Short-Term Memory*, Neural Computation (1997)
3. Hu et al. — *Squeeze-and-Excitation Networks*, CVPR (2018)
4. Bahdanau et al. — *Neural Machine Translation by Jointly Learning to Align and Translate*, ICLR (2015)
5. Gal & Ghahramani — *Dropout as a Bayesian Approximation*, ICML (2016)
6. Goodfellow et al. — *Explaining and Harnessing Adversarial Examples*, ICLR (2015)
7. Ribeiro et al. — *"Why Should I Trust You?": Explaining the Predictions of Any Classifier*, KDD (2016)
8. Lundberg & Lee — *A Unified Approach to Interpreting Model Predictions (SHAP)*, NeurIPS (2017)

---

##  Author

**Amlan Sarkar**  
**Ankit**

B.Tech — Computer Science & Engineering  
*Deep Learning · IoT Security · Explainable AI*

---

*© 2026 — WSN-DS IDS Research Project*
