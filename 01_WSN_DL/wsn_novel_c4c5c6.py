# ============================================================================
# WSN-DS IDS — NOVEL CONTRIBUTIONS C4, C5, C6
# C4: LIME Local Explainability
# C5: Adversarial Robustness (FGSM)
# C6: Attention Weight Visualization
#
# Run AFTER the main pipeline. Loads saved models from the last run.
# Usage:
#   python project/Scripts/wsn_novel_c4c5c6.py
# ============================================================================

import os, sys, json, time, pickle, logging, warnings, traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from collections import Counter

# ── pip install lime if not present ──────────────────────────────────────────
try:
    import lime
    import lime.lime_tabular
    LIME_OK = True
except ImportError:
    LIME_OK = False
    print("⚠ LIME not installed. Run: pip install lime")

# ── Config ───────────────────────────────────────────────────────────────────
DEVICE       = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
DATA_PATH    = '/root/amlan/Iot/project/data/raw/WSN-DS.csv'

# ── Auto-detect latest run ───────────────────────────────────────────────────
RUNS_BASE    = Path('/root/amlan/Iot/runs')
PREV_RUN_DIR = sorted([p for p in RUNS_BASE.glob('*') if p.is_dir()])[-1]  # picks latest run folder
print(f"📂 Using run: {PREV_RUN_DIR}")


# ============================================================================
# LOGGING
# ============================================================================
def setup_logger(out_dir):
    logger = logging.getLogger('WSN_C456')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(out_dir / 'logs' / 'novel_c456.log',
                             mode='w', encoding='utf-8')
    fh.setFormatter(fmt); fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt);  ch.setLevel(logging.INFO)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger


# ============================================================================
# MODEL ARCHITECTURES  (must match main script exactly)
# ============================================================================
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        avg = x.mean(dim=1)
        att = F.relu(self.fc1(avg))
        att = torch.sigmoid(self.fc2(att))
        return x * att.unsqueeze(1)


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        scores  = self.fc(torch.tanh(x))          # (B, T, 1)
        weights = F.softmax(scores, dim=1)         # (B, T, 1)
        context = (x * weights).sum(dim=1)         # (B, H)
        return context, weights


class CNNBiLSTMAttentionModel(nn.Module):
    def __init__(self, input_size, n_classes):
        super().__init__()
        self.conv1    = nn.Conv1d(1, 64,  3, padding=1)
        self.bn1      = nn.BatchNorm1d(64)
        self.conv2    = nn.Conv1d(64, 128, 3, padding=1)
        self.bn2      = nn.BatchNorm1d(128)
        self.pool     = nn.MaxPool1d(2)
        self.drop1    = nn.Dropout(0.3)
        self.chan_att = ChannelAttention(128, reduction=8)
        self.lstm     = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        self.drop2    = nn.Dropout(0.3)
        self.temp_att = TemporalAttention(128)
        self.fc1      = nn.Linear(128, 64)
        self.drop3    = nn.Dropout(0.3)
        self.fc2      = nn.Linear(64, n_classes)

    def forward(self, x, return_attn=False):
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.drop1(x)
        x = x.permute(0, 2, 1)
        x = self.chan_att(x)
        x, _ = self.lstm(x)
        x = self.drop2(x)
        context, attn_weights = self.temp_att(x)
        x = F.relu(self.fc1(context))
        x = self.drop3(x)
        logits = self.fc2(x)
        if return_attn:
            return logits, attn_weights
        return logits


# ============================================================================
# DATA LOADING  (identical preprocessing to main script)
# ============================================================================
def manual_smote(X, y, target_class, k=5, n_samples=1000):
    idx   = np.where(y == target_class)[0]
    k     = max(1, min(k, len(idx) - 1))
    X_min = X[idx]
    syn   = []
    for _ in range(n_samples):
        i      = np.random.randint(0, len(X_min))
        sample = X_min[i]
        dists  = np.sum((X_min - sample) ** 2, axis=1)
        nn_    = np.argsort(dists)[1:k + 1]
        if len(nn_) == 0:
            continue
        nb = X_min[np.random.choice(nn_)]
        syn.append(sample + np.random.random() * (nb - sample))
    return np.array(syn) if syn else X_min[:n_samples]


def balance_dataset(X, y, max_per_class=10000):
    counts = Counter(y)
    target = min(int(max(counts.values()) * 0.8), max_per_class)
    Xb, yb = X.copy(), y.copy()
    for label, count in counts.items():
        if count < target:
            syn = manual_smote(X, y, label, n_samples=target - count)
            Xb  = np.vstack([Xb, syn])
            yb  = np.concatenate([yb, [label] * (target - count)])
    return Xb, yb


def engineer_features(X, feature_names):
    df  = pd.DataFrame(X, columns=feature_names)
    eng = df.copy()
    eng['stat_mean']  = df.mean(axis=1)
    eng['stat_std']   = df.std(axis=1)
    eng['stat_max']   = df.max(axis=1)
    eng['stat_min']   = df.min(axis=1)
    eng['stat_range'] = eng['stat_max'] - eng['stat_min']
    eng['stat_skew']  = df.skew(axis=1)
    eng['stat_kurt']  = df.kurt(axis=1)
    top = df.var().sort_values(ascending=False).head(4).index.tolist()
    for i in range(len(top) - 1):
        eng[f'inter_{top[i]}_x_{top[i+1]}'] = df[top[i]] * df[top[i+1]]
    for col in df.columns[:5]:
        if df[col].abs().sum() > 0:
            eng[f'ratio_{col}'] = df[col] / (df.abs().sum(axis=1) + 1e-10)
    return eng.values, eng.columns.tolist()


def load_data(logger):
    logger.info("Loading dataset …")
    df         = pd.read_csv(DATA_PATH)
    target_col = 'Attack type'
    X_raw      = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).values
    feat_names = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).columns.tolist()
    y_raw      = df[target_col].values

    le        = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)

    X_eng, eng_names = engineer_features(X_raw, feat_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X_eng, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    X_train_b, y_train_b = balance_dataset(X_train, y_train, max_per_class=10000)

    # Load saved scaler (from previous run)
    scaler_path = PREV_RUN_DIR / 'models' / 'scaler.pkl'
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    X_train_sc = scaler.transform(X_train_b)
    X_test_sc  = scaler.transform(X_test)

    logger.info(f"✓ Test samples  : {X_test_sc.shape[0]:,}")
    logger.info(f"✓ Features      : {X_test_sc.shape[1]}")
    logger.info(f"✓ Classes       : {list(le.classes_)}")
    return X_train_sc, X_test_sc, y_train_b, y_test, le, eng_names


def load_model(n_features, n_classes, logger):
    model_path = PREV_RUN_DIR / 'models' / 'CNN_BiLSTM_Attention.pth'
    model = CNNBiLSTMAttentionModel(n_features, n_classes).to(DEVICE)
    ckpt  = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    logger.info(f"✓ Loaded CNN_BiLSTM_Attention (val_acc={ckpt['val_acc']:.4f})")
    return model


# ============================================================================
# C4: LIME LOCAL EXPLAINABILITY
# ============================================================================
def novel_c4_lime(model, X_train_sc, X_test_sc, y_test,
                  feature_names, class_names, out_dir, logger):
    logger.info("\n" + "="*70)
    logger.info("NOVEL C4: LIME LOCAL EXPLAINABILITY")
    logger.info("="*70)

    if not LIME_OK:
        logger.warning("LIME not installed — skipping C4.")
        return

    # PyTorch predict_proba wrapper for LIME
    def predict_proba(X_np):
        model.eval()
        with torch.no_grad():
            t = torch.FloatTensor(X_np).unsqueeze(2).to(DEVICE)  # (B,F,1)
            probs = F.softmax(model(t), dim=1).cpu().numpy()
        return probs

    t0 = time.time()
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train_sc[:5000],                 # background (subsample for speed)
        feature_names=feature_names,
        class_names=class_names,
        mode='classification',
        random_state=42
    )

    n_classes  = len(class_names)
    fig, axes  = plt.subplots(1, n_classes, figsize=(7 * n_classes, 7))

    for cls_idx, cls_name in enumerate(class_names):
        # Find one test sample of this class
        idxs = np.where(y_test == cls_idx)[0]
        if len(idxs) == 0:
            continue
        sample = X_test_sc[idxs[0]]

        exp = explainer.explain_instance(
            sample,
            predict_proba,
            num_features=12,
            top_labels=1,
            num_samples=500          # keep fast
        )

        top_label  = exp.top_labels[0]
        exp_list   = exp.as_list(label=top_label)
        features   = [x[0] for x in exp_list]
        values     = [x[1] for x in exp_list]
        colors     = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]

        ax = axes[cls_idx]
        ax.barh(features[::-1], values[::-1], color=colors[::-1],
                edgecolor='black', height=0.6)
        ax.axvline(0, color='black', linewidth=1.5)
        ax.set_title(f'{cls_name}\n(pred={class_names[top_label]})',
                     fontweight='bold', fontsize=11)
        ax.set_xlabel('LIME Weight', fontweight='bold')
        ax.tick_params(axis='y', labelsize=8)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        conf = exp.predict_proba[top_label]
        ax.text(0.01, 0.01, f'Confidence: {conf:.3f}',
                transform=ax.transAxes, fontsize=9,
                color='navy', style='italic')

    plt.suptitle(
        'Novel C4: LIME — Local Explanations Per Attack Class'
        '(Green = supports prediction, Red = contradicts)',
        fontweight='bold', fontsize=13
    )
    plt.tight_layout()
    plt.savefig(out_dir / 'plots' / 'C4_lime_explanations.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved: plots/C4_lime_explanations.png  [{time.time()-t0:.1f}s]")

    # Also save raw weights to JSON for reporting
    lime_data = {}
    for cls_idx, cls_name in enumerate(class_names):
        idxs = np.where(y_test == cls_idx)[0]
        if len(idxs) == 0:
            continue
        sample = X_test_sc[idxs[0]]
        exp = explainer.explain_instance(sample, predict_proba,
                                         num_features=12, top_labels=1,
                                         num_samples=500)
        top_label = exp.top_labels[0]
        lime_data[cls_name] = {
            'top_features': exp.as_list(label=top_label),
            'confidence'  : float(exp.predict_proba[top_label])
        }
    with open(out_dir / 'results' / 'C4_lime.json', 'w') as f:
        json.dump(lime_data, f, indent=4)
    logger.info("✓ Saved: results/C4_lime.json")


# ============================================================================
# C5: ADVERSARIAL ROBUSTNESS (FGSM)
# ============================================================================
def novel_c5_adversarial(model, X_test_sc, y_test, class_names,
                          out_dir, logger, n_samples=2000):
    logger.info("\n" + "="*70)
    logger.info("NOVEL C5: ADVERSARIAL ROBUSTNESS (FGSM)")
    logger.info("="*70)

    t0       = time.time()
    epsilons = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    results  = {}

    # Subsample for speed
    idx    = np.random.choice(len(X_test_sc), n_samples, replace=False)
    X_sub  = torch.FloatTensor(X_test_sc[idx]).to(DEVICE)    # (N, F)
    y_sub  = torch.LongTensor(y_test[idx]).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    for eps in epsilons:
        if eps == 0.0:
            model.eval()
            with torch.no_grad():
                preds = model(X_sub.unsqueeze(2)).argmax(1)
            acc = (preds == y_sub).float().mean().item()
        else:
            # RNN backward requires training mode
            model.train()
            X_adv = X_sub.clone().requires_grad_(True)
            logits = model(X_adv.unsqueeze(2))
            loss   = criterion(logits, y_sub)
            loss.backward()

            # Apply FGSM perturbation, then evaluate in eval mode
            X_perturbed = (X_adv + eps * X_adv.grad.sign()).detach()
            model.eval()
            with torch.no_grad():
                preds = model(X_perturbed.unsqueeze(2)).argmax(1)
            acc = (preds == y_sub).float().mean().item()

        results[eps] = round(acc, 4)
        logger.info(f"   ε={eps:.3f}  →  Accuracy: {acc:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    eps_vals = list(results.keys())
    acc_vals = list(results.values())

    axes[0].plot(eps_vals, acc_vals, 'bo-', lw=2.5, ms=8,
                 markerfacecolor='white', markeredgewidth=2)
    axes[0].fill_between(eps_vals, acc_vals, alpha=0.15, color='blue')
    axes[0].set_xlabel('Perturbation Magnitude ε (FGSM)', fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontweight='bold')
    axes[0].set_title('Model Accuracy Under FGSM Attack', fontweight='bold')
    axes[0].set_ylim([0, 1.05])
    axes[0].grid(alpha=0.4)
    for eps, acc in zip(eps_vals, acc_vals):
        axes[0].annotate(f'{acc:.3f}', (eps, acc),
                         textcoords='offset points', xytext=(0, 10),
                         fontsize=8, ha='center', color='navy')

    # Accuracy degradation
    degradation = [results[0.0] - v for v in acc_vals]
    bar_colors  = ['#2ecc71' if d < 0.02 else
                   '#f39c12' if d < 0.10 else '#e74c3c'
                   for d in degradation]
    axes[1].bar([str(e) for e in eps_vals], degradation,
                color=bar_colors, edgecolor='black')
    axes[1].axhline(0.02, color='green',  linestyle='--', lw=1.5,
                    label='Robust zone (<2%)')
    axes[1].axhline(0.10, color='orange', linestyle='--', lw=1.5,
                    label='Warning zone (<10%)')
    axes[1].set_xlabel('Perturbation ε', fontweight='bold')
    axes[1].set_ylabel('Accuracy Drop', fontweight='bold')
    axes[1].set_title('Accuracy Degradation vs ε', fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3, axis='y')

    plt.suptitle(
        'Novel C5: Adversarial Robustness — CNN_BiLSTM_Attention vs FGSM\n'
        f'Clean accuracy: {results[0.0]:.4f} | '
        f'At ε=0.1: {results[0.10]:.4f} | '
        f'At ε=0.3: {results[0.30]:.4f}',
        fontweight='bold', fontsize=12
    )
    plt.tight_layout()
    plt.savefig(out_dir / 'plots' / 'C5_adversarial_robustness.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved: plots/C5_adversarial_robustness.png  [{time.time()-t0:.1f}s]")

    with open(out_dir / 'results' / 'C5_adversarial.json', 'w') as f:
        json.dump({'fgsm_results': results,
                   'clean_acc': results[0.0],
                   'robustness_drop_at_eps0.1': round(results[0.0]-results[0.10],4),
                   'robustness_drop_at_eps0.3': round(results[0.0]-results[0.30],4)},
                  f, indent=4)
    logger.info("✓ Saved: results/C5_adversarial.json")
    return results


# ============================================================================
# C6: ATTENTION WEIGHT VISUALIZATION
# ============================================================================
def novel_c6_attention(model, X_test_sc, y_test, class_names,
                        feature_names, out_dir, logger, n_samples=3000):
    logger.info("\n" + "="*70)
    logger.info("NOVEL C6: ATTENTION WEIGHT VISUALIZATION")
    logger.info("="*70)

    t0 = time.time()
    n  = min(n_samples, len(X_test_sc))
    X_sub = torch.FloatTensor(X_test_sc[:n]).unsqueeze(2).to(DEVICE)  # (N,F,1)
    y_sub = y_test[:n]

    model.eval()
    all_attn_weights = []

    with torch.no_grad():
        BATCH = 512
        for i in range(0, n, BATCH):
            Xb        = X_sub[i:i+BATCH]
            _, weights = model(Xb, return_attn=True)  # (B, T, 1)
            all_attn_weights.append(weights.squeeze(-1).cpu().numpy())

    all_attn = np.vstack(all_attn_weights)  # (N, T)
    T        = all_attn.shape[1]

    n_classes = len(class_names)
    fig, axes = plt.subplots(2, n_classes, figsize=(5 * n_classes, 10))

    class_avg_attn = {}
    for cls_idx, cls_name in enumerate(class_names):
        mask = y_sub == cls_idx
        if mask.sum() == 0:
            continue
        avg_attn = all_attn[mask].mean(0)   # (T,)
        class_avg_attn[cls_name] = avg_attn

        # Top row: bar chart of attention weights
        ax0 = axes[0, cls_idx]
        colors = plt.cm.YlOrRd(avg_attn / (avg_attn.max() + 1e-10))
        ax0.bar(range(T), avg_attn, color=colors, edgecolor='black', linewidth=0.3)
        ax0.set_title(f'{cls_name}\nAvg Temporal Attention', fontweight='bold', fontsize=10)
        ax0.set_xlabel('Temporal Position (after CNN+Pool)', fontsize=8)
        ax0.set_ylabel('Attention Weight', fontsize=8)
        ax0.grid(alpha=0.3, axis='y')

        # Bottom row: heatmap-style (single row)
        ax1 = axes[1, cls_idx]
        sns.heatmap(avg_attn.reshape(1, -1), ax=ax1,
                    cmap='YlOrRd', cbar=True, linewidths=0.2,
                    xticklabels=False, yticklabels=[cls_name],
                    vmin=0, vmax=all_attn.max())
        ax1.set_title(f'{cls_name} — Attention Heatmap', fontsize=10)
        ax1.set_xlabel('Temporal Position', fontsize=8)

    plt.suptitle(
        'Novel C6: Temporal Attention Weights Per Attack Class\n'
        '(Shows WHICH temporal positions the model focuses on per attack type)',
        fontweight='bold', fontsize=13
    )
    plt.tight_layout()
    plt.savefig(out_dir / 'plots' / 'C6_attention_visualization.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved: plots/C6_attention_visualization.png  [{time.time()-t0:.1f}s]")

    # ── Cross-class attention comparison ────────────────────────────────────
    fig2, ax = plt.subplots(figsize=(12, 6))
    for cls_name, attn in class_avg_attn.items():
        ax.plot(range(T), attn, marker='o', ms=4, lw=2, label=cls_name)
    ax.set_xlabel('Temporal Position', fontweight='bold')
    ax.set_ylabel('Average Attention Weight', fontweight='bold')
    ax.set_title('C6: Cross-Class Attention Comparison\n'
                 '(Different attacks focus on different temporal regions)',
                 fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'plots' / 'C6_attention_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Saved: plots/C6_attention_comparison.png")

    with open(out_dir / 'results' / 'C6_attention.json', 'w') as f:
        json.dump({k: v.tolist() for k, v in class_avg_attn.items()}, f, indent=4)
    logger.info("✓ Saved: results/C6_attention.json")


# ============================================================================
# MAIN
# ============================================================================
def main():
    torch.manual_seed(42)
    np.random.seed(42)

    out_dir = PREV_RUN_DIR   # save into same run folder
    logger  = setup_logger(out_dir)

    logger.info("=" * 70)
    logger.info("NOVEL CONTRIBUTIONS C4 / C5 / C6")
    logger.info("=" * 70)
    logger.info(f"Device    : {DEVICE}")
    logger.info(f"Output    : {out_dir}")

    t_start = time.time()

    # ── Load data ──────────────────────────────────────────────────────────
    X_train_sc, X_test_sc, y_train_b, y_test, le, eng_names = load_data(logger)
    n_features = X_test_sc.shape[1]
    n_classes  = len(le.classes_)
    class_names = list(le.classes_)

    # ── Load model ────────────────────────────────────────────────────────
    model = load_model(n_features, n_classes, logger)

    # ── C4: LIME ──────────────────────────────────────────────────────────
    novel_c4_lime(model, X_train_sc, X_test_sc, y_test,
                  eng_names, class_names, out_dir, logger)

    # ── C5: Adversarial ───────────────────────────────────────────────────
    novel_c5_adversarial(model, X_test_sc, y_test,
                          class_names, out_dir, logger, n_samples=2000)

    # ── C6: Attention Viz ─────────────────────────────────────────────────
    novel_c6_attention(model, X_test_sc, y_test,
                        class_names, eng_names, out_dir, logger, n_samples=3000)

    elapsed = time.time() - t_start
    logger.info("\n" + "=" * 70)
    logger.info(f"✅  C4 / C5 / C6 COMPLETE — {elapsed:.1f}s total")
    logger.info("=" * 70)
    logger.info("\n📁 Output files:")
    logger.info(f"   plots/C4_lime_explanations.png")
    logger.info(f"   plots/C5_adversarial_robustness.png")
    logger.info(f"   plots/C6_attention_visualization.png")
    logger.info(f"   plots/C6_attention_comparison.png")
    logger.info(f"   results/C4_lime.json")
    logger.info(f"   results/C5_adversarial.json")
    logger.info(f"   results/C6_attention.json")


if __name__ == '__main__':
    main()
