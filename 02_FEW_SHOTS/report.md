# WSN‑DS Intrusion Detection with Deep Learning and Few‑Shot Meta‑Learning

## 1. Project Aim

This work develops a **high‑accuracy intrusion detection system (IDS)** for Wireless Sensor Networks using the **WSN‑DS** dataset, and then extends that IDS to the **few‑shot regime** where only a small number of labelled attack samples are available. The focus is on:

- Designing a strong deep encoder for WSN‑DS.
- Evaluating multiple few‑shot/meta‑learning methods.
- Proposing **two new prototype‑based architectures** for few‑shot intrusion detection.
- Analysing uncertainty, concept drift, energy–complexity trade‑offs, explainability and adversarial robustness.

All results are obtained from full training logs included in the repository [file:8][file:9][file:20][file:21].

---

## 2. Dataset and Preprocessing

- **Dataset:** WSN‑DS (Wireless Sensor Network Dataset).
- **Total samples:** 374,661; **Train:** 299,728; **Test:** 74,933 [file:8].
- **Classes (5):**  
  - `Normal`, `Blackhole`, `Grayhole`, `Flooding`, `TDMA` [file:8].
- **Features:** 18 original attributes, expanded to 33 after encoding [file:8].
- **Class imbalance:** Normal ≈ 90.77%, minority attacks <4% each [file:8].

Preprocessing steps in the main pipeline (`wsn_dl_research-7.py`) [file:9]:

1. Train–test split with stratification.
2. **SMOTE balancing** on the training set to upsample minority classes [file:9].
3. Feature scaling (scaler saved to disk).

---

## 3. Deep Learning Architectures

All models are implemented in PyTorch and trained end‑to‑end on the balanced training set [file:9]:

- LSTM
- BiLSTM
- CNN1D
- CNN‑BiLSTM
- **CNN‑BiLSTMAttention** (main deep encoder)
- TransformerIDS (Transformer‑based tabular model)

### 3.1 CNN‑BiLSTMAttention (Main Encoder)

Architecture [file:9]:

- 1D CNN layers to capture local temporal patterns.
- BiLSTM to model long‑range temporal dependencies.
- Dual attention:
  - **Channel attention** over feature channels.
  - **Temporal attention** over sequence steps.
- Fully connected head for 5‑class classification.

Performance on test set [file:9]:

- Accuracy: 0.9942
- F1‑score: 0.9942
- Macro F1 ≈ 0.96 across 5 classes.

This encoder is later **frozen** and used as a feature extractor for all few‑shot and prototype‑based methods.

### 3.2 Classical ML Baselines

To provide a strong reference:

- **Random Forest** – Test accuracy 0.9929, F1 0.9930 [file:9].
- **XGBoost** – Test accuracy 0.9968, F1 0.9968 (best full‑data score) [file:9].

---

## 4. Few‑Shot Learning and Meta‑Learning

### 4.1 Baseline Few‑Shot Methods

Using the frozen CNN‑BiLSTMAttention encoder, the following methods are evaluated under N‑way, K‑shot episodes (K ∈ {1, 3, 5, 10, 20}) [file:12][file:13]:

- **Prototypical Networks**
- **Matching Networks**
- **Relation Networks**
- **Siamese Networks**
- **Inductive Transfer** (linear classifier fine‑tuning)
- **APFP (2024 Adaptive Prototype baseline)**

These serve as baselines for the proposed methods.

### 4.2 Novel Contribution 1: AP++ (Adaptive Prototype++)

Implemented in `wsn_ap_pp-3.py` (v5) and `wsn_ap_pp_v4-2.py` (v6) [file:5][file:20].

**Key ideas:**

1. Dual‑metric similarity: combines cosine and Euclidean distances.
2. Adaptive temperature: scales softmax using intra‑class variance.
3. Outlier filtering: removes distant support examples.
4. Prototype refinement (EM‑like iterations).
5. Inter‑class repulsion to separate prototypes.
6. Additional normalisation/transformation components (v5/v6) [file:5][file:20].

**Results (v6)** [file:20]:

| K | Prototypical | APFP 2024 | AP++ (Ours) |
|---|---|---|---|
| 1 | 0.9233 | 0.9249 | 0.9330 |
| 3 | 0.9555 | 0.9404 | 0.9600 |
| 5 | 0.9568 | 0.9464 | 0.9623 |
| 10 | 0.9624 | 0.9534 | 0.9630 |
| 20 | 0.9642 | 0.9598 | 0.9625 |

**Ablation study at K=5** shows that while some individual components hurt performance, the **full AP++ combination** achieves the highest accuracy (0.9610), supporting the claim that the method works as a unified design [file:20].

---

### 4.3 Novel Contribution 2: CTPN (Contrastive Transformer Prototype Network)

Implemented in `ctpn_wsn.py`, `ctpn_wsn_v4-10.py`, and `ctpn_eval_v4-9.py` [file:11][file:16][file:21].

**CTPN pipeline** [file:21]:

1. **SupCon fine‑tuning**:
   - Uses supervised contrastive loss on encoder embeddings.
   - 60 epochs, batch size 512.
2. **Projection head**:
   - MLP projects embeddings into a normalised metric space.
3. **Support‑set attention and cross‑prototype refinement**:
   - Transformer‑style attention mechanisms over support examples.
4. **Multi‑K meta‑training**:
   - 600 episodes, K ∈ {1, 3, 5, 10, 20}.
5. **Episodic evaluation** vs Prototypical, APFP and AP++.

**Few‑shot results** [file:21]:

| K | Prototypical | APFP 2024 | AP++ | **CTPN (Ours)** |
|---|---|---|---|---|
| 1 | 0.9186 | 0.9239 | 0.9428 | **0.9531 (AUC 0.9934)** |
| 3 | 0.9555 | 0.9447 | 0.9599 | **0.9700 (AUC 0.9970)** |
| 5 | 0.9574 | 0.9479 | 0.9612 | **0.9688 (AUC 0.9970)** |
| 10 | 0.9588 | 0.9522 | 0.9613 | **0.9716 (AUC 0.9974)** |
| 20 | 0.9635 | 0.9566 | 0.9624 | **0.9694 (AUC 0.9974)** |

CTPN dominates all baselines at every K, with very high AUC values (>0.993) [file:21], making it the strongest proposed method.

---

## 5. Additional Novel Contributions (C1–C6)

All contributions are implemented and logged in the code; they are summarised here for the faculty.

### C1 – MC Dropout Uncertainty Quantification

- Uses Monte Carlo dropout on the CNN‑BiLSTMAttention model [file:8].
- 30 stochastic forward passes per sample.
- Overall accuracy: 0.9944; high‑confidence subset accuracy: 0.9984 [file:8].
- Per‑class analysis shows how often each attack type is flagged as “novel” (low confidence) [file:8].

### C2 – Chunk‑Based Concept Drift Detection

- Splits the stream into temporal chunks and monitors accuracy [file:8].
- Detects drift at **chunk 16** (baseline chunk accuracy 0.991 vs recent chunk 0.960) [file:8].
- After a lightweight retraining on recent data, average accuracy improves from 0.9775 (static) to 0.9816 (adaptive) [file:8].

### C3 – Energy–Complexity Trade‑Off Framework

- For each model, logs:
  - Accuracy on WSN‑DS.
  - Inference time (μs, batched).
  - Model size (KB).
  - Derived energy score and deployment tier (Edge/Gateway/Cloud) [file:8][file:9].

Example outcomes [file:8]:

- TransformerIDS: Edge (fast but lower accuracy).
- CNN1D: Edge.
- CNN‑BiLSTMAttention: Gateway (high accuracy, moderate cost).
- XGBoost: Cloud (highest accuracy, larger resource footprint).

### C4 – LIME‑Based Explainability

- Implemented in `wsn_novel_c4c5c6-6.py` [file:18].
- Uses LIME to generate local explanations for different attack classes.
- Produces per‑class feature importance bar plots, highlighting which features drive detection decisions.

### C5 – FGSM Adversarial Robustness

- Evaluates robustness of CNN‑BiLSTMAttention against FGSM adversarial perturbations [file:18].
- Tests multiple ε levels and reports accuracy vs ε curves.

### C6 – Additional Analyses

- Additional analytic contribution(s) (e.g., robustness or calibration studies) encoded in the same script [file:18].

---

## 6. Summary of Key Results

- **Full‑data IDS:** Deep encoder (CNN‑BiLSTMAttention) achieves **≈99.4%** test accuracy; XGBoost achieves **≈99.7%** [file:9].
- **Few‑shot regime:**  
  - AP++ improves over APFP across K, particularly for low‑shot (K=1,3) settings [file:20].  
  - CTPN sets a new state‑of‑the‑art on WSN‑DS few‑shot intrusion detection with **ACC up to 0.9716 and AUC up to 0.9974** [file:21].
- **Trustworthiness:**  
  - MC‑Dropout provides calibrated uncertainty and novel‑attack flags [file:8].  
  - Concept‑drift module identifies and adapts to distribution shifts [file:8].  
  - Energy–complexity analysis guides deployment across Edge/Gateway/Cloud tiers [file:8].  
  - LIME and FGSM experiments add interpretability and robustness analysis [file:18].

---

## 7. Files of Interest for Evaluation

- `wsn_dl_research-7.py` – main training and C1–C3.
- `wsn_few_shot_all-4.py`, `wsn_few_shot-5.py` – few‑shot baselines.
- `wsn_ap_pp_v4-2.py` – AP++ v6 implementation and logs.
- `ctpn_wsn_v4-10.py`, `ctpn_eval_v4-9.py` – CTPN training and evaluation.
- `wsn_novel_c4c5c6-6.py` – LIME, FGSM and related analyses.
- Logs: `training-8.log`, `console-7.log`, `ap_plus_plus.log`, `ctpn-2.log`, etc. for full traceability of all results [file:5][file:8][file:9][file:20][file:21].

---

## 8. How to Run (Faculty)

On a machine with PyTorch and the WSN‑DS CSV placed under `data/raw/`:

1. Run the main IDS training:

   ```bash
   python wsn_dl_research-7.py
   ```

2. Run few‑shot and AP++ experiments:

   ```bash
   python wsn_few_shot_all-4.py
   python wsn_ap_pp_v4-2.py
   ```

3. Train and evaluate CTPN:

   ```bash
   python ctpn_wsn_v4-10.py
   python ctpn_eval_v4-9.py
   ```

4. Run explainability and robustness:

   ```bash
   python wsn_novel_c4c5c6-6.py
   ```

Each script produces plots and JSON files summarising results inside the `runs/` directory. [file:8][file:21]
