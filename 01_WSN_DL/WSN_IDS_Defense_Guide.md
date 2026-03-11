# WSN Intrusion Detection — Complete Defense Guide
> **Project:** WSN-DS Deep Learning IDS | **Framework:** PyTorch  
> **Models:** LSTM · BiLSTM · CNN1D · CNN-BiLSTM · CNN-BiLSTM+Attention · Transformer  
> **Novel Contributions:** C1 MC Dropout · C2 Concept Drift · C3 Energy Framework · C4 LIME · C5 FGSM · C6 Attention Viz

---

## Table of Contents
1. [Dataset — WSN-DS](#1-dataset--wsn-ds)
2. [Feature Engineering](#2-feature-engineering-18--33-features)
3. [Class Imbalance — Manual SMOTE](#3-class-imbalance--manual-smote)
4. [Model Architectures](#4-model-architectures)
   - [LSTM](#lstm)
   - [BiLSTM](#bilstm)
   - [CNN_1D](#cnn_1d)
   - [CNN_BiLSTM](#cnn_bilstm)
   - [CNN_BiLSTM_Attention ⭐](#cnn_bilstm_attention-)
   - [Transformer_IDS](#transformer_ids)
5. [Training Setup](#5-training-setup)
6. [ML Baselines](#6-ml-baselines)
7. [Novel C1 — MC Dropout Uncertainty](#7-novel-c1--mc-dropout-uncertainty)
8. [Novel C2 — Concept Drift Detection](#8-novel-c2--concept-drift-detection)
9. [Novel C3 — Energy-Complexity Trade-off](#9-novel-c3--energy-complexity-trade-off)
10. [Novel C4 — LIME Explainability](#10-novel-c4--lime-explainability)
11. [Novel C5 — Adversarial Robustness (FGSM)](#11-novel-c5--adversarial-robustness-fgsm)
12. [Novel C6 — Attention Visualization](#12-novel-c6--attention-visualization)
13. [Big Picture — Overall Novelty](#13-big-picture--overall-novelty)

---

## Results Summary

| Model | Test Acc | F1 Score | Train Time | Deployment Tier |
|---|---|---|---|---|
| **XGBoost** 🥇 | **99.68%** | 0.9968 | 714s | 🔴 Cloud |
| **CNN_BiLSTM_Attention** 🥈 | **99.31%** | 0.9931 | 133s | 🟢 Edge |
| **Random Forest** 🥉 | **99.29%** | 0.9930 | 6.9s | 🔴 Cloud |
| CNN_1D | 99.22% | 0.9922 | 102s | 🟢 Edge |
| CNN_BiLSTM | 99.00% | 0.9900 | 119s | 🟢 Edge |
| LSTM | 98.93% | 0.9893 | 99s | 🟡 Gateway |
| BiLSTM | 98.75% | 0.9876 | 103s | 🔴 Cloud |
| Transformer_IDS | 96.91% | 0.9686 | 109s | 🟢 Edge |

---

## 1. Dataset — WSN-DS

**What it is:**  
A wireless sensor network dataset with 374,661 network traffic records across 5 classes:
- Normal (90.77%)
- Grayhole (3.90%)
- Blackhole (2.68%)
- TDMA (1.77%)
- Flooding (0.88%)

**Why this dataset:**
> WSN-DS is specifically designed for IoT/WSN environments. Unlike generic network datasets like KDD99 or NSL-KDD, WSN-DS reflects real WSN routing protocol behaviour (LEACH protocol), making results directly applicable to IoT security.

**Q: "Why not use a more popular dataset like NSL-KDD?"**
> NSL-KDD is from 1999 — it doesn't reflect modern IoT/WSN attack patterns. WSN-DS captures WSN-specific attacks like Grayhole (selective packet dropping) and TDMA scheduling attacks which are unique to sensor networks.

---

## 2. Feature Engineering (18 → 33 features)

**What was done:**  
Created 15 extra features from original 18:

| Category | Features Created |
|---|---|
| Statistical | mean, std, max, min, range, skew, kurtosis |
| Interaction | Products of top-4 high-variance feature pairs |
| Ratio | Feature / total row sum (5 features) |

**Why:**
> Raw features alone may not capture non-linear relationships. Statistical aggregates give the model a global view of the traffic pattern per sample. Interaction terms expose correlations between high-variance features. This is standard practice — it improved model convergence speed.

**Q: "Isn't this just making up data?"**
> No. These are deterministic mathematical transforms of existing features — no new information is invented. Skewness and kurtosis of a traffic record describe its distribution shape, which a neural network would eventually learn anyway but much slower.

---

## 3. Class Imbalance — Manual SMOTE

**The problem:**  
Normal traffic = 90.77% of data. A naive model predicting "Normal" for everything achieves 90% accuracy — useless as an IDS.

**What SMOTE does:**  
Synthetic Minority Over-sampling Technique — for each minority class sample, finds K nearest neighbours and generates a synthetic point **along the line between them**.

```
synthetic = sample_i + λ × (neighbour - sample_i)   where λ ∈ [0, 1]
```

**Why manual SMOTE (not sklearn's):**
> Implemented manually for full control over the oversampling cap (max 10,000 per class) and to avoid library dependency issues on the server.

**Q: "Why not just use class weights?"**
> Class weights penalise misclassification of minority classes during training but don't help the model *learn the decision boundary* of minority classes — it still sees far fewer examples. SMOTE physically adds training samples near the boundary, which teaches the model the actual shape of each class.

**Q: "Can SMOTE cause overfitting?"**
> Yes, if synthetic samples are placed outside the true distribution. Mitigated by capping at 10,000 samples (not equal to Normal's 272,052) and using k=5 nearest neighbours which keeps synthetic points close to real ones.

---

## 4. Model Architectures

### LSTM

**What it is:**  
Long Short-Term Memory — a recurrent network with three gates:
- **Forget gate:** What to discard from cell state
- **Input gate:** What new information to store
- **Output gate:** What to output from cell state

**Why in IDS:**
> Network traffic has temporal dependencies — a Blackhole attack unfolds over multiple packets in time. LSTM explicitly models this sequence. The forget gate discards irrelevant past traffic, and the input gate retains anomalous patterns.

**Q: "What problem does LSTM solve that a regular RNN can't?"**
> Vanishing gradient problem. In regular RNNs, gradients shrink exponentially during backpropagation through time, so the model forgets long-range dependencies. LSTM's gating mechanism maintains a constant error carousel (cell state) that preserves gradients across long sequences.

---

### BiLSTM

**What it is:**  
Two LSTMs — one reads the sequence forward, one reads it backward. Their outputs are concatenated at each time step.

**Why:**
> A Grayhole attack may have indicators both *before* (suspicious routing changes) and *after* (drop in packet delivery) the actual attack event. BiLSTM sees both directions simultaneously, giving a richer context vector.

**Q: "Does BiLSTM make sense for real-time detection?"**
> For offline forensic analysis yes — you have the full packet history. For real-time, a forward LSTM is more appropriate. This is why we also benchmarked single LSTM, and our energy framework (C3) correctly tagged BiLSTM as 'Cloud' tier due to its 2× parameter count.

---

### CNN_1D

**What it is:**  
1D Convolutional Neural Network — sliding filters across feature dimensions to detect local patterns.

Architecture:
```
Conv1D(1→64) → BN → Conv1D(64→128) → BN → MaxPool → Dropout
→ Conv1D(128→64) → GlobalAvgPool → FC(64) → FC(n_classes)
```

**Why:**
> CNNs detect local feature correlations regardless of position. A flooding attack signature (sudden spike in packet rate combined with low TTL) is a local pattern in the feature vector. CNNs detect it wherever it appears.

**Q: "Features aren't a time series — why use CNN?"**
> True — our features are tabular, not sequential. But Conv1D treats the feature vector as a 1D signal and learns which *combinations of adjacent features* are discriminative. It's essentially learning feature interaction patterns automatically, similar to what we did manually in feature engineering but learned end-to-end.

---

### CNN_BiLSTM

**What it is:**  
CNN layers first extract local feature patterns → output fed to BiLSTM for sequential modelling.

**Architecture logic:**
> CNN acts as a feature extractor — it compresses the 33 features into high-level abstract representations. BiLSTM then models temporal dependencies *between* these abstract representations. This is a hierarchical approach: local patterns first, then global temporal context.

---

### CNN_BiLSTM_Attention ⭐

> **Best DL model — 99.31% accuracy | Edge deployable**

**What it is:**  
CNN + BiLSTM + **dual attention mechanisms:**

#### Channel Attention (Squeeze-and-Excitation)
```
x (B,T,C) → GlobalAvgPool → FC → ReLU → FC → Sigmoid → scale x
```
Learns which feature **channels** matter more for each sample.

#### Temporal Attention
```
score_t = tanh(W · h_t)
weight_t = softmax(score_t)
context = Σ (weight_t × h_t)
```
Learns which **time steps** to focus on.

**Why this is the novel architecture:**
> Standard CNN-BiLSTM treats all channels and all time steps equally. In WSN attacks, not all features are equally important — for Flooding, packet rate dominates; for Blackhole, routing table features dominate. Channel attention reweights feature channels dynamically. Temporal attention then finds the *most informative moment* in the sequence rather than just taking the last hidden state.

**Q: "Why is attention better than just taking the last hidden state?"**
> The last hidden state in BiLSTM can suffer from recency bias — it weights recent time steps more. Attention considers ALL time steps and learns which ones contain the attack signature, regardless of when they occur in the sequence.

**Q: "What is the attention mechanism mathematically?"**
> For temporal attention: compute score for each time step t as `score_t = tanh(W·h_t)`, apply softmax to get weights summing to 1, then compute weighted sum of hidden states: `context = Σ(weight_t × h_t)`. This context vector goes into the classifier — it's a learned weighted summary of the entire sequence.

---

### Transformer_IDS

**What it is:**  
Multi-head self-attention — every feature attends to every other feature simultaneously.

**Why included despite lower accuracy (96.91%):**
> Transformers are state-of-the-art in NLP and increasingly in time series. Including it provides a comprehensive benchmark. Its lower accuracy is explained by dataset size — Transformers typically need larger datasets to outperform recurrent models, and 313k samples isn't sufficient for quadratic attention complexity to pay off.

---

## 5. Training Setup

### Adam Optimizer
**Why Adam over SGD:**
> Adam adapts the learning rate per-parameter using first and second moment estimates. For a multi-class IDS problem with sparse gradients (attack classes are rare), Adam converges much faster and more reliably than vanilla SGD.

### ReduceLROnPlateau
**What:** Reduces learning rate by 0.5× if validation loss doesn't improve for 5 epochs.

**Why:**
> When the model plateaus, the current learning rate is too large to find a finer minimum. Reducing it allows the optimizer to take smaller steps and escape the plateau. Visible in BiLSTM's log — LR dropped from 0.001 to 0.0005 at epoch 26, immediately improving performance.

### Batch Normalisation
**Why:**
> Normalises layer activations to zero mean and unit variance per mini-batch. Prevents internal covariate shift — the phenomenon where the distribution of layer inputs changes during training, forcing each layer to constantly re-adapt. Makes training stable and allows higher learning rates.

### Dropout (p=0.3)
**Why:**
> Randomly sets 30% of neurons to zero during training. Forces the network to learn redundant representations — no single neuron becomes the sole detector of any pattern. Primary regularisation against overfitting. Disabled during inference (neurons contribute with scaled weights).

### Early Stopping (patience=10)
**Why:**
> Stops training when validation accuracy stops improving for 10 consecutive epochs and saves the best checkpoint. Prevents overfitting after the optimal point. LSTM's best model was at epoch 26 — without early stopping it would have continued degrading.

### Gradient Clipping (max norm=1.0)
**Why:**
> Caps the gradient norm at 1.0 before each parameter update. Prevents exploding gradients — a common problem in RNNs where gradients grow exponentially through time steps, causing NaN losses or erratic weight updates.

---

## 6. ML Baselines

### Random Forest (99.29%)
**Why include it:**
> Random Forest is a strong baseline for tabular data. If deep learning models don't beat it, they aren't justified. CNN_BiLSTM_Attention (99.31%) marginally beats RF while being edge-deployable — RF requires 21MB and 113µs inference vs 538KB and 9.9µs for our attention model.

### XGBoost (99.68% — highest overall)
**Why XGBoost outperforms everything:**
> XGBoost is an ensemble of gradient-boosted decision trees with L1/L2 regularisation. It dominates tabular data benchmarks because it captures complex feature interactions through tree splits without requiring a specific input format. Its superiority confirms the WSN-DS dataset has strong tabular structure.

**Q: "If XGBoost is best, why use deep learning at all?"**
> Three reasons:
> 1. **Inference speed:** XGBoost needs 89µs per inference. Our CNN_BiLSTM needs only 9.9µs — 9× faster on constrained IoT devices.
> 2. **Scalability:** XGBoost cannot process raw sequential packet streams without manual feature engineering. DL models can adapt to raw inputs.
> 3. **Explainability & uncertainty:** XGBoost cannot provide MC Dropout uncertainty estimates (C1) or temporal attention analysis (C6).

---

## 7. Novel C1 — MC Dropout Uncertainty

**What:**  
Run the model 30 times on the same sample with dropout **enabled** during inference. The variance across predictions = uncertainty.

**The math:**
```
Entropy H = -Σ p_i × log(p_i)
Confidence = 1 - (H / H_max)    where H_max = log(n_classes)
Novel flag = True if Confidence < 0.70
```

**Results:**
- Overall accuracy: 99.35%
- High-confidence accuracy: **99.84%**
- Novel flagged: 161 samples (1.6%)

**Why this matters:**
> A standard classifier always gives a confident prediction even for inputs it has never seen. A novel Blackhole variant will be classified as 'Normal' with 95% confidence — no alarm raised. MC Dropout detects genuine uncertainty and flags inputs for human review.

**Q: "Why does dropout during inference simulate uncertainty?"**
> It approximates Bayesian inference — dropout training is mathematically equivalent to a variational Bayes approximation of a Gaussian process. Each dropout mask creates a different sub-network. Running 30 forward passes = sampling 30 different neural networks from the approximate posterior distribution. High variance = genuine model uncertainty.

---

## 8. Novel C2 — Concept Drift Detection

**What:**  
Stream test data in 20 chunks. Monitor accuracy chunk by chunk. If accuracy drops >3% vs baseline → drift detected → retrain adaptive model on recent data.

```
Drift condition: mean(last 3 chunks) < mean(first 3 chunks) - 0.03
```

**Results:**
- Drift events detected: 1
- Static avg accuracy: 97.31%
- Adaptive avg accuracy: 98.16%
- **Improvement: +0.85%**

**Why this is critical for WSN:**
> Network attack patterns evolve. An IDS trained in January will see new Blackhole variants by March. Static models degrade silently — accuracy drops without any alarm. Chunk-based detection catches drift and triggers retraining automatically.

**Q: "Your drift is artificially injected — is this valid?"**
> Yes, this is standard practice in concept drift research (ADWIN, DDM methods all use synthetic drift injection). We inject Gaussian noise with increasing magnitude on the latter half of the test stream to simulate distribution shift. The important result is whether the detection mechanism works correctly — it does.

---

## 9. Novel C3 — Energy-Complexity Trade-off

**The formula:**
```
Energy Score = (Accuracy × F1) / (log(1 + inf_time) × log(1 + model_size) × log(1 + FLOPs))
```

**Deployment tiers:**
| Score | Tier |
|---|---|
| > 0.5 | 🟢 Edge (sensor node) |
| 0.2 – 0.5 | 🟡 Gateway |
| < 0.2 | 🔴 Cloud only |

**Results:**
| Model | Score | Tier |
|---|---|---|
| Transformer_IDS | 1.92 | 🟢 Edge |
| CNN_1D | 1.06 | 🟢 Edge |
| CNN_BiLSTM | 0.57 | 🟢 Edge |
| **CNN_BiLSTM_Attention** | **0.51** | **🟢 Edge** |
| LSTM | 0.44 | 🟡 Gateway |
| BiLSTM | 0.17 | 🔴 Cloud |
| XGBoost | 0.011 | 🔴 Cloud |
| Random Forest | 0.004 | 🔴 Cloud |

**Q: "This formula seems arbitrary — why these weights?"**
> Log-scaling mirrors the logarithmic relationship between resource cost and user experience — doubling model size doesn't double deployment difficulty, but 10× size becomes a hard constraint. Equal weighting of inference time and model size reflects that both matter equally in constrained IoT. We acknowledge this as one valid formulation — Pareto-front analysis is a complementary alternative.

---

## 10. Novel C4 — LIME Explainability

**What:**  
For a single test sample, LIME perturbs input features, observes how the model's prediction changes, and fits a simple linear model to approximate the local decision boundary.

```
Explanation = argmin [ L(f, g, π_x) + Ω(g) ]
where f = original model, g = linear surrogate, π_x = locality kernel
```

**Why LIME in addition to SHAP:**
> SHAP gives the average contribution of each feature *across the entire dataset* — global explainability. LIME explains *why this specific packet was classified as Blackhole* — local explainability. For an IDS, local explanation is crucial: a network admin needs to know which feature of THIS alert triggered the alarm, not the average feature importance.

**Q: "LIME is an approximation — is it trustworthy?"**
> LIME's fidelity depends on the local linearity assumption. For smooth decision boundaries it's highly accurate. We used 500 perturbations per explanation to ensure the linear approximation is stable. LIME is a standard accepted technique in XAI — used in Google's What-If Tool and adopted in medical/legal AI applications.

---

## 11. Novel C5 — Adversarial Robustness (FGSM)

**What FGSM does:**
```
x_adversarial = x + ε × sign(∇ₓ Loss(x, y_true))
```
Compute the gradient of loss with respect to the *input*, then add a tiny perturbation in the direction that maximally increases the loss — minimal change to maximally fool the model.

**Why test this:**
> A sophisticated attacker can craft network packets nearly identical to normal traffic but that cross the model's decision boundary (evasion attack). Results at ε=0.1 reveal the model's vulnerability boundary and motivate future adversarial training.

**Q: "FGSM is a white-box attack — the attacker needs model weights. Is this realistic?"**
> FGSM is a white-box threat model establishing the *upper bound* on adversarial vulnerability. In practice, black-box attacks (using a substitute model) are more realistic but weaker. If the model is robust to white-box FGSM, it's robust to all weaker attacks. We use it as a stress test — not a claim of real-world robustness.

**Q: "How would you defend against this?"**
> Adversarial training — include FGSM-perturbed samples in the training set. This is a direct extension of this work and is noted as future work.

---

## 12. Novel C6 — Attention Visualization

**What:**  
Extract temporal attention weights from CNN_BiLSTM_Attention for each test sample. Average by attack class. Visualise which temporal positions the model focuses on per attack type.

**Why this is a contribution:**
> Black-box models in security are a liability — when the model makes a mistake, you have no explanation. Attention visualisation shows that different attack types trigger different focus patterns, validating that the model has learned attack-specific signatures and not just statistical correlations.

**Q: "Attention weights don't equal feature importance — are you overclaiming?"**
> This is a valid criticism (Jain & Wallace, 2019 showed attention ≠ explanation in all cases). We treat attention visualisation as a *diagnostic tool*, not ground-truth explanation. It shows *where* the model looks, not necessarily *why*. For rigorous feature attribution we use SHAP and LIME. Attention viz is complementary — combined they give a complete interpretability picture.

---

## 13. Big Picture — Overall Novelty

> **Q: "Why is this research novel overall?"**

Previous work on WSN IDS focuses only on classification accuracy. This work contributes **five dimensions beyond accuracy:**

| Dimension | Contribution | Method |
|---|---|---|
| **Trustworthiness** | Flags unknown attacks instead of silently misclassifying | MC Dropout (C1) |
| **Adaptability** | Keeps system accurate as attacks evolve | Concept Drift (C2) |
| **Deployability** | Matches models to hardware tiers | Energy Framework (C3) |
| **Explainability** | Makes the system auditable | LIME + SHAP + Attention (C4, C6) |
| **Robustness** | Quantifies vulnerability to evasion attacks | FGSM (C5) |

> Together these five dimensions address the gap between academic IDS benchmarks and real-world deployable WSN security systems.

---

*Last updated: March 2026*
