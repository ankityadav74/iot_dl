# ============================================================================
# WSN-DS DEEP LEARNING IDS — PYTORCH IMPLEMENTATION
# Models: LSTM, BiLSTM, CNN1D, CNN-BiLSTM, CNN-BiLSTM+Attention, Transformer
# Novel C1: MC Dropout Uncertainty
# Novel C2: Chunk-Based Concept Drift
# Novel C3: Energy-Complexity Trade-off
# ============================================================================

import os, sys, json, time, pickle, logging, warnings, traceback
from datetime import datetime
from collections import Counter
from itertools import cycle
from pathlib import Path

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
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_curve, auc, f1_score
)
from sklearn.preprocessing import label_binarize

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ── Device setup (auto GPU/CPU) ────────────────────────────────────────────
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# SECTION 0: RUN DIRECTORY & LOGGING
# ============================================================================

def setup_run_environment(base_dir='runs'):
    timestamp = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    run_dir   = Path(base_dir) / timestamp

    for sub in ['plots', 'results', 'models', 'logs', 'logs/epoch_history']:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger('WSN_IDS')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh = logging.FileHandler(run_dir / 'logs' / 'training.log',
                             mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("=" * 70)
    logger.info("WSN-DS DEEP LEARNING IDS — PYTORCH PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Run Dir    : {run_dir.resolve()}")
    logger.info(f"Timestamp  : {timestamp}")
    logger.info(f"PyTorch    : {torch.__version__}")
    logger.info(f"Device     : {DEVICE}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem  = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"  GPU {i}    : {name} ({mem:.1f} GB)")
    else:
        logger.warning("GPU not found — running on CPU")

    config = {
        'timestamp' : timestamp,
        'run_dir'   : str(run_dir.resolve()),
        'pytorch'   : torch.__version__,
        'device'    : str(DEVICE),
        'cuda'      : torch.cuda.is_available(),
        'xgboost'   : XGBOOST_AVAILABLE,
        'shap'      : SHAP_AVAILABLE
    }
    with open(run_dir / 'run_config.json', 'w') as f:
        json.dump(config, f, indent=4)

    return run_dir, logger


# ============================================================================
# SECTION 1: DATA LOADING & PREPROCESSING
# ============================================================================

def load_wsn_dataset(filepath, logger, run_dir):
    logger.info("=" * 70)
    logger.info("STEP 1/7 — DATA LOADING")
    logger.info("=" * 70)

    t0 = time.time()
    df = pd.read_csv(filepath)
    logger.info(f"✓ Shape       : {df.shape[0]:,} × {df.shape[1]}")
    logger.info(f"✓ Load time   : {time.time()-t0:.2f}s")

    candidates = ['Attack_Type','attack_type','Class','class',
                  'Label','label','Attack','target','Attack type']
    target_col = next((c for c in candidates if c in df.columns),
                      df.columns[-1])
    logger.info(f"✓ Target col  : '{target_col}'")

    dist = df[target_col].value_counts()
    logger.info("\n📊 Class Distribution:")
    for cls, cnt in dist.items():
        logger.info(f"   {str(cls):20s}: {cnt:8,} ({cnt/len(df)*100:.2f}%)")

    if df.isnull().sum().sum() > 0:
        df = df.dropna()
        logger.warning(f"Dropped NaN rows. New shape: {df.shape}")
    else:
        logger.info("✓ No missing values")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(range(len(dist)), dist.values,
                color='steelblue', edgecolor='black')
    axes[0].set_xticks(range(len(dist)))
    axes[0].set_xticklabels(dist.index, rotation=45, ha='right')
    axes[0].set_ylabel('Count', fontweight='bold')
    axes[0].set_title('Class Distribution', fontweight='bold')
    axes[1].pie(dist.values, labels=dist.index, autopct='%1.1f%%',
                colors=plt.cm.Set3(range(len(dist))))
    axes[1].set_title('Class Proportions', fontweight='bold')
    plt.tight_layout()
    plt.savefig(run_dir / 'plots' / 'class_distribution.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    return df, target_col


def engineer_features(X, feature_names, logger):
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

    logger.info(f"✓ Features: {X.shape[1]} → {eng.shape[1]}")
    return eng.values, eng.columns.tolist()


def manual_smote(X, y, target_class, k=5, n_samples=1000):
    idx   = np.where(y == target_class)[0]
    k     = max(1, min(k, len(idx) - 1))
    X_min = X[idx]
    syn   = []
    for _ in range(n_samples):
        i      = np.random.randint(0, len(X_min))
        sample = X_min[i]
        dists  = np.sum((X_min - sample) ** 2, axis=1)
        nn     = np.argsort(dists)[1:k + 1]
        if len(nn) == 0:
            continue
        nb = X_min[np.random.choice(nn)]
        syn.append(sample + np.random.random() * (nb - sample))
    return np.array(syn) if syn else X_min[:n_samples]


def balance_dataset(X, y, logger, max_per_class=10000):
    logger.info("STEP 3/7 — SMOTE BALANCING")
    counts = Counter(y)
    logger.info(f"  Before: { {k:v for k,v in sorted(counts.items())} }")
    target = min(int(max(counts.values()) * 0.8), max_per_class)
    Xb, yb = X.copy(), y.copy()
    for label, count in counts.items():
        if count < target:
            syn = manual_smote(X, y, label, n_samples=target - count)
            Xb  = np.vstack([Xb, syn])
            yb  = np.concatenate([yb, [label] * (target - count)])
    logger.info(f"  After : { {k:v for k,v in sorted(Counter(yb).items())} }")
    return Xb, yb


# ============================================================================
# SECTION 2: PYTORCH DATASET
# ============================================================================

class WSNDataset(Dataset):
    def __init__(self, X, y):
        # X shape: (N, features) → (N, features, 1) for Conv1D
        self.X = torch.FloatTensor(X).unsqueeze(2)   # (N, F, 1)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================================
# SECTION 3: PYTORCH MODEL ARCHITECTURES
# ============================================================================

class LSTMModel(nn.Module):
    def __init__(self, input_size, n_classes):
        super().__init__()
        self.lstm1   = nn.LSTM(input_size, 128, batch_first=True)
        self.drop1   = nn.Dropout(0.3)
        self.lstm2   = nn.LSTM(128, 64, batch_first=True)
        self.drop2   = nn.Dropout(0.3)
        self.fc1     = nn.Linear(64, 32)
        self.bn      = nn.BatchNorm1d(32)
        self.fc2     = nn.Linear(32, n_classes)

    def forward(self, x):
        # x: (B, F, 1) → (B, 1, F) for LSTM
        x = x.squeeze(2).unsqueeze(1)
        x, _ = self.lstm1(x)
        x    = self.drop1(x)
        x, _ = self.lstm2(x)
        x    = self.drop2(x[:, -1, :])
        x    = F.relu(self.bn(self.fc1(x)))
        return self.fc2(x)


class BiLSTMModel(nn.Module):
    def __init__(self, input_size, n_classes):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, 128, batch_first=True,
                             bidirectional=True)
        self.drop1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(256, 64, batch_first=True,
                             bidirectional=True)
        self.drop2 = nn.Dropout(0.3)
        self.fc1   = nn.Linear(128, 32)
        self.bn    = nn.BatchNorm1d(32)
        self.fc2   = nn.Linear(32, n_classes)

    def forward(self, x):
        x = x.squeeze(2).unsqueeze(1)
        x, _ = self.lstm1(x)
        x    = self.drop1(x)
        x, _ = self.lstm2(x)
        x    = self.drop2(x[:, -1, :])
        x    = F.relu(self.bn(self.fc1(x)))
        return self.fc2(x)


class CNN1DModel(nn.Module):
    def __init__(self, input_size, n_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64,  3, padding=1)
        self.bn1   = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn2   = nn.BatchNorm1d(128)
        self.pool  = nn.MaxPool1d(2)
        self.drop  = nn.Dropout(0.3)
        self.conv3 = nn.Conv1d(128, 64, 3, padding=1)
        self.gap   = nn.AdaptiveAvgPool1d(1)
        self.fc1   = nn.Linear(64, 64)
        self.drop2 = nn.Dropout(0.3)
        self.fc2   = nn.Linear(64, n_classes)

    def forward(self, x):
        # x: (B, F, 1) → (B, 1, F) for Conv1d
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.drop(x)
        x = F.relu(self.conv3(x))
        x = self.gap(x).squeeze(2)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        return self.fc2(x)


class CNNBiLSTMModel(nn.Module):
    def __init__(self, input_size, n_classes):
        super().__init__()
        self.conv1  = nn.Conv1d(1, 64,  3, padding=1)
        self.bn1    = nn.BatchNorm1d(64)
        self.conv2  = nn.Conv1d(64, 128, 3, padding=1)
        self.bn2    = nn.BatchNorm1d(128)
        self.pool   = nn.MaxPool1d(2)
        self.drop1  = nn.Dropout(0.3)
        self.lstm   = nn.LSTM(128, 64, batch_first=True,
                              bidirectional=True)
        self.drop2  = nn.Dropout(0.3)
        self.fc1    = nn.Linear(128, 64)
        self.drop3  = nn.Dropout(0.3)
        self.fc2    = nn.Linear(64, n_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)                    # (B, 1, F)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.drop1(x)
        x = x.permute(0, 2, 1)                    # (B, T, 128)
        x, _ = self.lstm(x)
        x = self.drop2(x[:, -1, :])               # last timestep
        x = F.relu(self.fc1(x))
        x = self.drop3(x)
        return self.fc2(x)


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        # x: (B, T, C)
        avg = x.mean(dim=1)                        # (B, C)
        att = F.relu(self.fc1(avg))
        att = torch.sigmoid(self.fc2(att))         # (B, C)
        return x * att.unsqueeze(1)                # (B, T, C)


class TemporalAttention(nn.Module):
    """Soft temporal attention over LSTM outputs"""
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (B, T, H)
        scores = self.fc(torch.tanh(x))            # (B, T, 1)
        weights = F.softmax(scores, dim=1)         # (B, T, 1)
        context = (x * weights).sum(dim=1)         # (B, H)
        return context, weights


class CNNBiLSTMAttentionModel(nn.Module):
    """
    NOVEL ARCHITECTURE:
    CNN-BiLSTM with Dual Attention (Channel + Temporal)
    Main contribution for the research paper.
    """
    def __init__(self, input_size, n_classes):
        super().__init__()
        # CNN block
        self.conv1    = nn.Conv1d(1, 64,  3, padding=1)
        self.bn1      = nn.BatchNorm1d(64)
        self.conv2    = nn.Conv1d(64, 128, 3, padding=1)
        self.bn2      = nn.BatchNorm1d(128)
        self.pool     = nn.MaxPool1d(2)
        self.drop1    = nn.Dropout(0.3)

        # Channel Attention
        self.chan_att = ChannelAttention(128, reduction=8)

        # BiLSTM
        self.lstm     = nn.LSTM(128, 64, batch_first=True,
                                bidirectional=True)
        self.drop2    = nn.Dropout(0.3)

        # Temporal Attention
        self.temp_att = TemporalAttention(128)  # 64*2 bidirectional

        # Classifier
        self.fc1      = nn.Linear(128, 64)
        self.drop3    = nn.Dropout(0.3)
        self.fc2      = nn.Linear(64, n_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)                    # (B, 1, F)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.drop1(x)
        x = x.permute(0, 2, 1)                    # (B, T, 128)

        # Channel attention
        x = self.chan_att(x)

        # BiLSTM
        x, _ = self.lstm(x)                       # (B, T, 128)
        x = self.drop2(x)

        # Temporal attention → context vector
        x, _ = self.temp_att(x)                   # (B, 128)

        x = F.relu(self.fc1(x))
        x = self.drop3(x)
        return self.fc2(x)


class TransformerIDS(nn.Module):
    """Transformer with Multi-Head Self-Attention"""
    def __init__(self, input_size, n_classes,
                 num_heads=4, ff_dim=128, dropout=0.1):
        super().__init__()
        self.proj     = nn.Linear(1, 64)
        self.attn     = nn.MultiheadAttention(64, num_heads,
                                              dropout=dropout,
                                              batch_first=True)
        self.norm1    = nn.LayerNorm(64)
        self.ff       = nn.Sequential(
            nn.Linear(64, ff_dim), nn.ReLU(),
            nn.Linear(ff_dim, 64)
        )
        self.norm2    = nn.LayerNorm(64)
        self.drop     = nn.Dropout(dropout)
        self.pool     = nn.AdaptiveAvgPool1d(1)
        self.fc1      = nn.Linear(64, 64)
        self.drop2    = nn.Dropout(0.3)
        self.fc2      = nn.Linear(64, n_classes)

    def forward(self, x):
        # x: (B, F, 1)
        x  = self.proj(x)                          # (B, F, 64)
        a, _ = self.attn(x, x, x)
        x  = self.norm1(x + self.drop(a))
        ff = self.ff(x)
        x  = self.norm2(x + self.drop(ff))
        x  = x.permute(0, 2, 1)                   # (B, 64, F)
        x  = self.pool(x).squeeze(2)              # (B, 64)
        x  = F.relu(self.fc1(x))
        x  = self.drop2(x)
        return self.fc2(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# SECTION 4: TRAINING & EVALUATION
# ============================================================================

def train_model(model, train_loader, val_loader,
                model_name, run_dir, logger,
                epochs=30, lr=0.001, patience=10):

    logger.info("\n" + "=" * 60)
    logger.info(f"TRAINING: {model_name}")
    logger.info("=" * 60)
    logger.info(f"  Parameters  : {count_parameters(model):,}")
    logger.info(f"  Device      : {DEVICE}")
    logger.info(f"  Epochs      : {epochs}")

    model    = model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc  = 0.0
    best_epoch    = 0
    patience_cnt  = 0
    history       = {'train_loss': [], 'train_acc': [],
                     'val_loss':   [], 'val_acc':   []}

    model_path = run_dir / 'models' / f'{model_name}.pth'
    t0         = time.time()

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(Xb)
            loss   = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_loss    += loss.item() * len(yb)
            tr_correct += (logits.argmax(1) == yb).sum().item()
            tr_total   += len(yb)

        tr_loss /= tr_total
        tr_acc   = tr_correct / tr_total

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        va_loss, va_correct, va_total = 0.0, 0, 0

        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb  = Xb.to(DEVICE), yb.to(DEVICE)
                logits   = model(Xb)
                loss     = criterion(logits, yb)
                va_loss += loss.item() * len(yb)
                va_correct += (logits.argmax(1) == yb).sum().item()
                va_total   += len(yb)

        va_loss /= va_total
        va_acc   = va_correct / va_total

        scheduler.step(va_loss)

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(va_loss)
        history['val_acc'].append(va_acc)

        logger.info(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"Train: loss={tr_loss:.4f} acc={tr_acc:.4f} | "
            f"Val: loss={va_loss:.4f} acc={va_acc:.4f} | "
            f"LR={optimizer.param_groups[0]['lr']:.6f}"
        )

        # Early stopping + save best
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_epoch   = epoch
            patience_cnt = 0
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'optimizer'  : optimizer.state_dict(),
                'val_acc'    : va_acc,
                'val_loss'   : va_loss
            }, model_path)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                logger.info(f"  ⏹  Early stop at epoch {epoch} "
                            f"(best={best_val_acc:.4f} @ ep {best_epoch})")
                break

    # Load best weights
    ckpt = torch.load(model_path, weights_only=False)
    model.load_state_dict(ckpt['model_state'])

    train_time = time.time() - t0
    logger.info(f"✓ Best val acc : {best_val_acc:.4f} (epoch {best_epoch})")
    logger.info(f"✓ Train time   : {train_time:.1f}s ({train_time/60:.1f} min)")

    # Save history CSV
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(
        run_dir / 'logs' / 'epoch_history' / f'{model_name}_history.csv',
        index=False)

    return model, history, train_time


def evaluate_model(model, test_loader, X_test_raw,
                   model_name, le, history,
                   run_dir, logger):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb      = Xb.to(DEVICE)
            logits  = model(Xb)
            probs   = F.softmax(logits, dim=1).cpu().numpy()
            preds   = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(yb.numpy())

    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    y_true = np.array(all_labels)

    acc    = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0)

    best_val  = max(history['val_acc'])
    best_train = max(history['train_acc'])

    logger.info(f"\n📊 {model_name}")
    logger.info(f"   Train Acc : {best_train:.4f}")
    logger.info(f"   Val Acc   : {best_val:.4f}")
    logger.info(f"   Test Acc  : {acc:.4f}")
    logger.info(f"   F1 Score  : {f1:.4f}")
    logger.info(f"\n{classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0)}")

    # Save report
    report = classification_report(y_true, y_pred,
                                   target_names=le.classes_,
                                   zero_division=0)
    with open(run_dir / 'results' / f'{model_name}_report.txt', 'w') as f:
        f.write(f"Model        : {model_name}\n")
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score     : {f1:.4f}\n\n")
        f.write(report)

    return {
        'model_name'     : model_name,
        'model_type'     : 'Deep Learning',
        'train_accuracy' : float(best_train),
        'val_accuracy'   : float(best_val),
        'test_accuracy'  : float(acc),
        'precision'      : float(p),
        'recall'         : float(r),
        'f1_score'       : float(f1),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'y_pred_prob'    : y_prob,
        'history'        : history
    }
    # ============================================================================
# SECTION 4b: ML BASELINE EVALUATION
# ============================================================================

def evaluate_ml_model(model, X_test, y_test, model_name, le,
                      train_time, run_dir, logger):
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)

    acc    = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0)

    logger.info(f"\n📊 {model_name}")
    logger.info(f"   Test Acc  : {acc:.4f}")
    logger.info(f"   F1 Score  : {f1:.4f}")
    logger.info(f"   Train time: {train_time:.1f}s")

    report = classification_report(
        y_test, y_pred, target_names=le.classes_, zero_division=0)
    logger.info(f"\n{report}")

    with open(run_dir / 'results' / f'{model_name}_report.txt', 'w') as f:
        f.write(f"Model        : {model_name}\n")
        f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score     : {f1:.4f}\n\n")
        f.write(report)

    return {
        'model_name'      : model_name,
        'model_type'      : 'Traditional ML',
        'train_accuracy'  : float(acc),
        'val_accuracy'    : float(acc),
        'test_accuracy'   : float(acc),
        'precision'       : float(p),
        'recall'          : float(r),
        'f1_score'        : float(f1),
        'train_time_s'    : round(train_time, 2),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'y_pred_prob'     : y_pred_prob
    }



# ============================================================================
# SECTION 5: NOVEL CONTRIBUTION 1 — MC DROPOUT UNCERTAINTY
# ============================================================================

def run_contribution_1(model, test_loader, y_test,
                        le, run_dir, logger,
                        model_name, n_mc=30,
                        threshold=0.70, max_samples=10000):
    logger.info("\n" + "=" * 70)
    logger.info("NOVEL C1: MC DROPOUT UNCERTAINTY QUANTIFICATION")
    logger.info("=" * 70)

    # Enable dropout during inference
    model.train()
    all_probs = []
    all_labels = []
    count = 0

    t0 = time.time()
    with torch.no_grad():
        for Xb, yb in test_loader:
            if count >= max_samples:
                break
            Xb = Xb.to(DEVICE)
            batch_preds = []
            for _ in range(n_mc):
                logits = model(Xb)
                probs  = F.softmax(logits, dim=1).cpu().numpy()
                batch_preds.append(probs)

            mean_probs = np.mean(batch_preds, axis=0)
            all_probs.append(mean_probs)
            all_labels.extend(yb.numpy())
            count += len(yb)

    model.eval()  # restore eval mode

    mean_probs  = np.vstack(all_probs)[:max_samples]
    y_eval      = np.array(all_labels)[:max_samples]
    predictions = np.argmax(mean_probs, axis=1)

    eps        = 1e-10
    entropy    = -np.sum(mean_probs * np.log(mean_probs + eps), axis=1)
    max_ent    = np.log(mean_probs.shape[1])
    confidence = 1.0 - (entropy / (max_ent + eps))
    novel_flag = confidence < threshold

    acc    = accuracy_score(y_eval, predictions)
    hc_mask = ~novel_flag
    acc_hc  = (accuracy_score(y_eval[hc_mask], predictions[hc_mask])
               if hc_mask.sum() > 0 else 0.0)

    elapsed = time.time() - t0
    logger.info(f"  MC samples             : {n_mc}")
    logger.info(f"  Overall Accuracy       : {acc:.4f}")
    logger.info(f"  High-Conf Accuracy     : {acc_hc:.4f}")
    logger.info(f"  Novel Flagged          : "
                f"{novel_flag.sum():,} ({novel_flag.mean()*100:.1f}%)")
    logger.info(f"  Avg Confidence         : {confidence.mean():.4f}")
    logger.info(f"  Time                   : {elapsed:.1f}s")

    logger.info("\n  Per-Class:")
    for i, cls in enumerate(le.classes_):
        mask = y_eval == i
        if mask.sum() > 0:
            logger.info(f"    {cls:15s} | Conf:{confidence[mask].mean():.3f} "
                        f"| Novel:{novel_flag[mask].sum()}/{mask.sum()}")

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].hist(confidence, bins=60, color='steelblue',
                 edgecolor='black', alpha=0.8)
    axes[0].axvline(threshold, color='red', linestyle='--',
                    lw=2, label=f'Threshold ({threshold})')
    axes[0].set_xlabel('Confidence Score', fontweight='bold')
    axes[0].set_ylabel('Count', fontweight='bold')
    axes[0].set_title('MC Dropout Confidence Distribution', fontweight='bold')
    axes[0].legend()

    axes[1].pie([hc_mask.sum(), novel_flag.sum()],
                labels=['Known\n(High Conf)', 'Novel/Zero-Day\n(Low Conf)'],
                colors=['#2ecc71', '#e74c3c'], autopct='%1.1f%%')
    axes[1].set_title('Known vs Novel Attack Detection', fontweight='bold')

    n_show = min(600, len(confidence))
    axes[2].scatter(range(n_show), confidence[:n_show],
                    c=['#2ecc71' if not f else '#e74c3c'
                       for f in novel_flag[:n_show]],
                    alpha=0.5, s=12)
    axes[2].axhline(threshold, color='black', linestyle='--', lw=2)
    axes[2].set_xlabel('Sample Index', fontweight='bold')
    axes[2].set_ylabel('Confidence', fontweight='bold')
    axes[2].set_title('Per-Sample Confidence\n(Green=Known, Red=Novel)',
                      fontweight='bold')

    plt.suptitle(f'Novel C1: MC Dropout Uncertainty — {model_name}',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(run_dir / 'plots' / 'C1_uncertainty.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Saved: plots/C1_uncertainty.png")

    results = {
        'model_name'         : model_name,
        'overall_accuracy'   : float(acc),
        'high_conf_accuracy' : float(acc_hc),
        'novel_count'        : int(novel_flag.sum()),
        'novel_pct'          : float(novel_flag.mean() * 100),
        'avg_confidence'     : float(confidence.mean()),
        'elapsed_seconds'    : round(elapsed, 2)
    }
    with open(run_dir / 'results' / 'C1_uncertainty.json', 'w') as f:
        json.dump(results, f, indent=4)
    return results


# ============================================================================
# SECTION 6: NOVEL CONTRIBUTION 2 — CONCEPT DRIFT
# ============================================================================

def run_contribution_2(model, X_test_sc, y_test,
                        X_train_sc, y_train_b,
                        run_dir, logger, n_chunks=20):
    logger.info("\n" + "=" * 70)
    logger.info("NOVEL C2: CHUNK-BASED CONCEPT DRIFT DETECTION")
    logger.info("=" * 70)

    t0         = time.time()
    chunk_size = len(X_test_sc) // n_chunks
    total      = n_chunks * chunk_size

    X_stream = X_test_sc[:total].copy()
    y_stream = y_test[:total].copy()

    for i in range(n_chunks // 2, n_chunks):
        s = i * chunk_size
        e = s + chunk_size
        strength = ((i - n_chunks//2) / (n_chunks//2)) * 0.3
        X_stream[s:e] += np.random.normal(0, strength, X_stream[s:e].shape)

    static_accs, adaptive_accs = [], []
    drift_pts, retrain_pts      = [], []
    acc_history                 = []

    adaptive_rf = RandomForestClassifier(
        n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    adaptive_rf.fit(X_train_sc[:10000], y_train_b[:10000])

    model.eval()
    for i in range(n_chunks):
        s  = i * chunk_size
        e  = s + chunk_size
        Xc = X_stream[s:e]
        yc = y_stream[s:e]

        # Static DL
        Xt = torch.FloatTensor(Xc).unsqueeze(2).to(DEVICE)
        with torch.no_grad():
            preds_dl = model(Xt).argmax(1).cpu().numpy()
        static_acc = accuracy_score(yc, preds_dl)
        static_accs.append(static_acc)

        # Adaptive RF
        adap_acc = accuracy_score(yc, adaptive_rf.predict(Xc))
        acc_history.append(adap_acc)
        adaptive_accs.append(adap_acc)

        # Drift detection
        drift = False
        if len(acc_history) >= 4:
            recent   = np.mean(acc_history[-3:])
            baseline = np.mean(acc_history[:3])
            if (baseline - recent) > 0.03:
                drift = True
                drift_pts.append(i)
                logger.info(f"  ⚠️  DRIFT chunk {i:2d} | "
                            f"Baseline:{baseline:.3f} → Recent:{recent:.3f}")

        if drift:
            lookback = chunk_size * 3
            rX = X_stream[max(0, s-lookback):e]
            ry = y_stream[max(0, s-lookback):e]
            if len(np.unique(ry)) > 1:
                adaptive_rf = RandomForestClassifier(
                    n_estimators=50, max_depth=10,
                    random_state=42, n_jobs=-1)
                adaptive_rf.fit(rX, ry)
                retrain_pts.append(i)
                logger.info(f"  🔄 Retrained on {len(rX):,} samples")

    elapsed = time.time() - t0
    logger.info(f"\n  ✓ Done in {elapsed:.1f}s")
    logger.info(f"  Drift events   : {len(drift_pts)}")
    logger.info(f"  Retrains       : {len(retrain_pts)}")
    logger.info(f"  Static avg     : {np.mean(static_accs):.4f}")
    logger.info(f"  Adaptive avg   : {np.mean(adaptive_accs):.4f}")
    logger.info(f"  Improvement    : "
                f"+{(np.mean(adaptive_accs)-np.mean(static_accs))*100:.2f}%")

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].plot(adaptive_accs, 'g-o', ms=5, lw=2, label='Adaptive')
    axes[0].plot(static_accs,   'r-s', ms=5, lw=2, label='Static DL')
    for dp in drift_pts:
        axes[0].axvline(dp, color='orange', linestyle='--', alpha=0.8)
    for rp in retrain_pts:
        axes[0].axvline(rp, color='blue',   linestyle='-.', alpha=0.8)
    axes[0].set_xlabel('Chunk (Time →)', fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontweight='bold')
    axes[0].set_title('Static vs Adaptive Under Drift', fontweight='bold')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    diff = np.array(adaptive_accs) - np.array(static_accs)
    axes[1].bar(range(n_chunks), diff,
                color=['#2ecc71' if d >= 0 else '#e74c3c' for d in diff])
    axes[1].axhline(0, color='black', lw=1.5)
    axes[1].set_xlabel('Chunk Index', fontweight='bold')
    axes[1].set_ylabel('Accuracy Gain', fontweight='bold')
    axes[1].set_title('Adaptive − Static per Chunk', fontweight='bold')

    plt.suptitle('Novel C2: Chunk-Based Concept Drift Detection',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(run_dir / 'plots' / 'C2_concept_drift.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Saved: plots/C2_concept_drift.png")

    results = {
        'drift_events'         : len(drift_pts),
        'retrain_events'       : len(retrain_pts),
        'static_avg_accuracy'  : round(float(np.mean(static_accs)),  4),
        'adaptive_avg_accuracy': round(float(np.mean(adaptive_accs)), 4),
        'improvement_pct'      : round(float(
            (np.mean(adaptive_accs)-np.mean(static_accs))*100), 4),
        'elapsed_seconds'      : round(elapsed, 2)
    }
    with open(run_dir / 'results' / 'C2_drift.json', 'w') as f:
        json.dump(results, f, indent=4)
    return results


# ============================================================================
# SECTION 7: NOVEL CONTRIBUTION 3 — ENERGY TRADE-OFF
# ============================================================================

def run_contribution_3(dl_models_dict, ml_models_dict,
                        test_loader_small,
                        X_test_ml, y_test,
                        le, run_dir, logger,
                        eval_samples=1000):
    logger.info("\n" + "=" * 70)
    logger.info("NOVEL C3: ENERGY-COMPLEXITY TRADE-OFF FRAMEWORK")
    logger.info("=" * 70)

    results = []
    y_eval  = y_test[:eval_samples]

    for name, (model, _) in dl_models_dict.items():
        model.eval()
        preds, count = [], 0
        t_inf = time.perf_counter()
        with torch.no_grad():
            for Xb, _ in test_loader_small:
                if count >= eval_samples: break
                out = model(Xb.to(DEVICE)).argmax(1).cpu().numpy()
                preds.extend(out); count += len(out)
        inf_us = (time.perf_counter() - t_inf) / eval_samples * 1e6

        y_pred    = np.array(preds)[:eval_samples]
        acc       = accuracy_score(y_eval, y_pred)
        f1        = f1_score(y_eval, y_pred, average='weighted',
                             zero_division=0)
        params    = count_parameters(model)
        size_kb   = params * 4 / 1024
        flops_M   = params / 1e6

        energy = (acc * f1) / (
            np.log1p(inf_us) * np.log1p(size_kb) * np.log1p(flops_M))
        tier = ("🟢 Edge"    if energy > 0.5 else
                "🟡 Gateway" if energy > 0.2 else
                "🔴 Cloud")
        results.append({
            'model_name'    : name, 'model_type': 'Deep Learning',
            'accuracy'      : round(acc,    4),
            'f1_score'      : round(f1,     4),
            'inf_time_us'   : round(inf_us, 3),
            'model_size_kb' : round(size_kb,1),
            'params_M'      : round(flops_M,4),
            'energy_score'  : round(energy, 4),
            'deployment_tier': tier
        })
        logger.info(f"  {name:30s} | Acc:{acc:.4f} | "
                    f"Inf:{inf_us:.2f}µs | Size:{size_kb:.0f}KB | "
                    f"Score:{energy:.4f} | {tier}")

    for name, model in ml_models_dict.items():
        t_inf  = time.perf_counter()
        y_pred = model.predict(X_test_ml[:eval_samples])
        inf_us = (time.perf_counter() - t_inf) / eval_samples * 1e6

        acc     = accuracy_score(y_eval, y_pred)
        f1      = f1_score(y_eval, y_pred, average='weighted',
                           zero_division=0)
        size_kb = len(pickle.dumps(model)) / 1024
        energy  = (acc * f1) / (
            np.log1p(inf_us) * np.log1p(size_kb) * np.log1p(size_kb/100))
        tier = ("🟢 Edge"    if energy > 0.5 else
                "🟡 Gateway" if energy > 0.2 else "🔴 Cloud")
        results.append({
            'model_name'    : name, 'model_type': 'Traditional ML',
            'accuracy'      : round(acc,    4),
            'f1_score'      : round(f1,     4),
            'inf_time_us'   : round(inf_us, 3),
            'model_size_kb' : round(size_kb,1),
            'params_M'      : round(size_kb/100, 4),
            'energy_score'  : round(energy, 4),
            'deployment_tier': tier
        })
        logger.info(f"  {name:30s} | Acc:{acc:.4f} | "
                    f"Inf:{inf_us:.2f}µs | Size:{size_kb:.0f}KB | "
                    f"Score:{energy:.4f} | {tier}")

    energy_df = pd.DataFrame(results).sort_values('energy_score',
                                                    ascending=False)
    energy_df.to_csv(run_dir / 'results' / 'C3_energy.csv', index=False)
    logger.info(f"\n{energy_df[['model_name','accuracy','inf_time_us','model_size_kb','energy_score','deployment_tier']].to_string(index=False)}")

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    tier_clrs = {'🟢 Edge':'#2ecc71','🟡 Gateway':'#f39c12',
                 '🔴 Cloud':'#e74c3c'}
    bar_colors = [tier_clrs.get(t,'#95a5a6')
                  for t in energy_df['deployment_tier']]
    axes[0].barh(energy_df['model_name'], energy_df['energy_score'],
                 color=bar_colors, edgecolor='black', height=0.6)
    axes[0].axvline(0.5, color='green',  linestyle='--', lw=2,
                    label='Edge (0.5)')
    axes[0].axvline(0.2, color='orange', linestyle='--', lw=2,
                    label='Gateway (0.2)')
    axes[0].set_xlabel('Energy-Efficiency Score', fontweight='bold')
    axes[0].set_title('Deployment Tier Recommendation', fontweight='bold')
    axes[0].legend()

    sc = axes[1].scatter(energy_df['inf_time_us'], energy_df['accuracy'],
                         s=energy_df['model_size_kb'] / 5,
                         c=energy_df['energy_score'],
                         cmap='RdYlGn', alpha=0.85, edgecolors='black')
    for _, row in energy_df.iterrows():
        axes[1].annotate(row['model_name'],
                         (row['inf_time_us'], row['accuracy']),
                         xytext=(4,4), textcoords='offset points',
                         fontsize=7)
    plt.colorbar(sc, ax=axes[1], label='Energy Score')
    axes[1].set_xlabel('Inference Time (µs)', fontweight='bold')
    axes[1].set_ylabel('Accuracy', fontweight='bold')
    axes[1].set_title('Accuracy vs Speed', fontweight='bold')

    x, w = np.arange(len(energy_df)), 0.2
    norm = lambda v: np.array(v) / (max(np.array(v)) + 1e-10)
    axes[2].bar(x-1.5*w, norm(energy_df['accuracy']),  w, label='Accuracy',  color='#3498db')
    axes[2].bar(x-0.5*w, norm(1/(energy_df['inf_time_us']+1e-10)), w, label='Speed', color='#2ecc71')
    axes[2].bar(x+0.5*w, norm(1/(energy_df['model_size_kb']+1e-10)), w, label='Compact', color='#e67e22')
    axes[2].bar(x+1.5*w, norm(energy_df['energy_score']), w, label='Energy', color='#9b59b6')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(energy_df['model_name'],
                             rotation=45, ha='right', fontsize=8)
    axes[2].set_ylabel('Normalised Score', fontweight='bold')
    axes[2].set_title('Multi-Criteria Analysis', fontweight='bold')
    axes[2].legend(fontsize=8)

    plt.suptitle('Novel C3: Energy-Complexity Trade-off Framework',
                 fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(run_dir / 'plots' / 'C3_energy_tradeoff.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Saved: plots/C3_energy_tradeoff.png")
    logger.info("✓ Saved: results/C3_energy.csv")
    return energy_df


# ============================================================================
# SECTION 8: STANDARD VISUALISATIONS
# ============================================================================

def plot_training_histories(histories_dict, run_dir, logger):
    n = len(histories_dict)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
    if n == 1: axes = axes.reshape(1, 2)

    for i, (name, h) in enumerate(histories_dict.items()):
        axes[i,0].plot(h['train_acc'], lw=2, label='Train')
        axes[i,0].plot(h['val_acc'],   lw=2, label='Val')
        axes[i,0].set_title(f'{name} — Accuracy', fontweight='bold')
        axes[i,0].set_xlabel('Epoch'); axes[i,0].legend()
        axes[i,0].grid(alpha=0.3)

        axes[i,1].plot(h['train_loss'], lw=2, color='red',    label='Train')
        axes[i,1].plot(h['val_loss'],   lw=2, color='orange', label='Val')
        axes[i,1].set_title(f'{name} — Loss', fontweight='bold')
        axes[i,1].set_xlabel('Epoch'); axes[i,1].legend()
        axes[i,1].grid(alpha=0.3)

    plt.suptitle('Training History', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(run_dir / 'plots' / 'training_history.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Saved: plots/training_history.png")


def plot_model_comparison(all_results, run_dir, logger):
    df = pd.DataFrame([{
        'Model'    : r['model_name'],
        'Type'     : r.get('model_type','Unknown'),
        'Test Acc' : r['test_accuracy'],
        'F1 Score' : r['f1_score'],
        'Precision': r['precision'],
        'Recall'   : r['recall']
    } for r in all_results]).sort_values('Test Acc', ascending=False)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    type_clr  = {'Deep Learning':'#3498db', 'Traditional ML':'#e67e22'}
    for ax, metric in zip(axes.flat,
                          ['Test Acc','F1 Score','Precision','Recall']):
        sd   = df.sort_values(metric, ascending=True)
        clrs = [type_clr.get(t,'steelblue') for t in sd['Type']]
        ax.barh(sd['Model'], sd[metric], color=clrs, edgecolor='navy')
        ax.set_xlim([0, 1.05])
        ax.set_xlabel('Score', fontweight='bold')
        ax.set_title(metric, fontweight='bold')
        for i, (_, row) in enumerate(sd.iterrows()):
            ax.text(row[metric]+0.005, i, f"{row[metric]:.4f}",
                    va='center', fontsize=8)

    from matplotlib.patches import Patch
    fig.legend(
        handles=[Patch(color='#3498db', label='Deep Learning'),
                 Patch(color='#e67e22', label='Traditional ML')],
        loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.01)
    )
    plt.suptitle('All Models — Performance Comparison',
                 fontweight='bold', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(run_dir / 'plots' / 'model_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Saved: plots/model_comparison.png")
    return df


def plot_confusion_matrices(all_results, le, run_dir, logger):
    for res in all_results:
        cm      = np.array(res['confusion_matrix']).astype(float)
        rs      = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, rs, where=rs != 0)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title(f"Confusion Matrix — {res['model_name']}",
                  fontweight='bold')
        plt.ylabel('True', fontweight='bold')
        plt.xlabel('Predicted', fontweight='bold')
        plt.tight_layout()
        fname = f"cm_{res['model_name'].replace(' ','_')}.png"
        plt.savefig(run_dir / 'plots' / fname, dpi=300, bbox_inches='tight')
        plt.close()
    logger.info(f"✓ Saved {len(all_results)} confusion matrices")


def plot_roc_curves(all_results, y_test, le, run_dir, logger):
    n_cls      = len(le.classes_)
    y_bin      = label_binarize(y_test, classes=range(n_cls))
    colors_it  = cycle(['#1abc9c','#e74c3c','#3498db','#f39c12',
                        '#9b59b6','#2ecc71','#e67e22','#e91e63'])
    plt.figure(figsize=(12, 8))
    for res, color in zip(all_results, colors_it):
        if 'y_pred_prob' not in res: continue
        try:
            fpr, tpr, _ = roc_curve(y_bin.ravel(),
                                     res['y_pred_prob'].ravel())
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, lw=2,
                     label=f"{res['model_name']} (AUC={roc_auc:.3f})")
        except Exception: pass

    plt.plot([0,1],[0,1],'k--', lw=2, label='Random')
    plt.xlabel('FPR', fontweight='bold')
    plt.ylabel('TPR', fontweight='bold')
    plt.title('ROC Curves — All Models', fontweight='bold')
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / 'plots' / 'roc_curves.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Saved: plots/roc_curves.png")


# ============================================================================
# SECTION 9: MAIN PIPELINE
# ============================================================================

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    run_dir, logger = setup_run_environment('runs')
    pipeline_start  = time.time()

    try:
        # ── Load ──────────────────────────────────────────────────────────
        df, target_col = load_wsn_dataset(
            '/root/amlan/Iot/project/data/raw/WSN-DS.csv',
            logger, run_dir)

        X          = (df.drop(columns=[target_col])
                       .select_dtypes(include=[np.number]).values)
        y_raw      = df[target_col].values
        feat_names = (df.drop(columns=[target_col])
                       .select_dtypes(include=[np.number])
                       .columns.tolist())

        le        = LabelEncoder()
        y_encoded = le.fit_transform(y_raw)
        n_classes = len(le.classes_)
        logger.info(f"✓ Classes ({n_classes}): "
                    f"{dict(zip(le.classes_, range(n_classes)))}")

        # ── Feature Engineering ───────────────────────────────────────────
        X_eng, eng_names = engineer_features(X, feat_names, logger)
        n_features = X_eng.shape[1]

        # ── Split ─────────────────────────────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X_eng, y_encoded, test_size=0.2,
            random_state=42, stratify=y_encoded)
        logger.info(f"  Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

        # ── SMOTE ─────────────────────────────────────────────────────────
        X_train_b, y_train_b = balance_dataset(
            X_train, y_train, logger, max_per_class=10000)

        # ── Scale ─────────────────────────────────────────────────────────
        scaler     = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train_b)
        X_test_sc  = scaler.transform(X_test)
        with open(run_dir / 'models' / 'scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        logger.info("✓ Scaler saved")

        # Val split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_sc, y_train_b, test_size=0.1,
            random_state=42, stratify=y_train_b)

        # ── DataLoaders ───────────────────────────────────────────────────
        BATCH = 1024
        train_loader = DataLoader(
            WSNDataset(X_tr, y_tr), batch_size=BATCH,
            shuffle=True,  num_workers=4, pin_memory=True)
        val_loader   = DataLoader(
            WSNDataset(X_val, y_val), batch_size=BATCH,
            shuffle=False, num_workers=4, pin_memory=True)
        test_loader  = DataLoader(
            WSNDataset(X_test_sc, y_test), batch_size=BATCH,
            shuffle=False, num_workers=4, pin_memory=True)
        # Small loader for C3
        X_small = X_test_sc[:1000]
        y_small = y_test[:1000]
        small_loader = DataLoader(
            WSNDataset(X_small, y_small), batch_size=256,
            shuffle=False)

        # ── DL Models ─────────────────────────────────────────────────────
        logger.info("\n" + "=" * 70)
        logger.info("STEP 5/7 — DEEP LEARNING TRAINING")
        logger.info("=" * 70)

        dl_builders = {
            'LSTM'                : LSTMModel,
            'BiLSTM'              : BiLSTMModel,
            'CNN_1D'              : CNN1DModel,
            'CNN_BiLSTM'          : CNNBiLSTMModel,
            'CNN_BiLSTM_Attention': CNNBiLSTMAttentionModel,
            'Transformer_IDS'     : TransformerIDS
        }

        all_dl_results = []
        dl_models_dict = {}
        histories_dict = {}

        for name, ModelClass in dl_builders.items():
            try:
                model = ModelClass(n_features, n_classes)
                model, history, train_time = train_model(
                    model, train_loader, val_loader,
                    name, run_dir, logger, epochs=30, patience=10)
                res = evaluate_model(
                    model, test_loader, X_test_sc,
                    name, le, history, run_dir, logger)
                res['train_time_s'] = round(train_time, 1)
                all_dl_results.append(res)
                dl_models_dict[name]  = (model, history)
                histories_dict[name]  = history
            except Exception as e:
                logger.error(f"❌ {name}: {e}")
                logger.debug(traceback.format_exc())

        # ── ML Baselines ───────────────────────────────────────────────────
        logger.info("\n" + "=" * 70)
        logger.info("STEP 6/7 — ML BASELINES")
        logger.info("=" * 70)

        ml_configs = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=15,
                class_weight='balanced', random_state=42, n_jobs=-1)
        }
        if XGBOOST_AVAILABLE:
            ml_configs['XGBoost'] = xgb.XGBClassifier(
             n_estimators=100,
            max_depth=7,
            learning_rate=0.1,
            tree_method='hist',
            device='cuda',
            random_state=42,
            eval_metric='mlogloss')


        all_ml_results = []
        trained_ml     = {}
        for name, model in ml_configs.items():
            try:
                t0 = time.time()
                model.fit(X_train_sc, y_train_b)
                train_time = time.time() - t0
                res = evaluate_ml_model(
                    model, X_test_sc, y_test,
                    name, le, train_time, run_dir, logger)
                all_ml_results.append(res)
                trained_ml[name] = model
                with open(run_dir / 'models' / f'{name}.pkl', 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"✓ {name} saved")
            except Exception as e:
                logger.error(f"❌ {name}: {e}")

        all_results = all_dl_results + all_ml_results

        # ── Save Results ───────────────────────────────────────────────────
        logger.info("\n" + "=" * 70)
        logger.info("STEP 7/7 — SAVING & NOVEL CONTRIBUTIONS")
        logger.info("=" * 70)

        records = []
        for r in all_results:
            records.append({
                'Model'        : r.get('model_name',    'Unknown'),
                'Type'         : r.get('model_type',    'Unknown'),
                'Test_Accuracy': r.get('test_accuracy', 0.0),
                'Precision'    : r.get('precision',     0.0),
                'Recall'       : r.get('recall',        0.0),
                'F1_Score'     : r.get('f1_score',      0.0),
                'Train_Time_s' : r.get('train_time_s',  0.0)
            })

        if records:
            results_df = pd.DataFrame(records).sort_values(
                'Test_Accuracy', ascending=False)
        else:
            logger.warning("⚠️  No results to save — all models failed")
            results_df = pd.DataFrame(columns=[
                'Model','Type','Test_Accuracy','Precision',
                'Recall','F1_Score','Train_Time_s'])


        results_df.to_csv(run_dir / 'results' / 'model_comparison.csv',
                          index=False)
        safe = [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                 for k, v in r.items() if k != 'y_pred_prob'}
                for r in all_results]
        with open(run_dir / 'results' / 'all_results.json', 'w') as f:
            json.dump(safe, f, indent=4)

        # Standard plots
        if histories_dict:
            plot_training_histories(histories_dict, run_dir, logger)
        plot_model_comparison(all_results, run_dir, logger)
        plot_confusion_matrices(all_results, le, run_dir, logger)
        plot_roc_curves(all_results, y_test, le, run_dir, logger)

        # Best DL model for contributions
        best_name = ('CNN_BiLSTM_Attention'
                     if 'CNN_BiLSTM_Attention' in dl_models_dict
                     else list(dl_models_dict.keys())[0]
                     if dl_models_dict else None)

        # ── C1 ─────────────────────────────────────────────────────────────
        c1_results = None
        if best_name:
            c1_results = run_contribution_1(
                dl_models_dict[best_name][0],
                test_loader, y_test, le, run_dir, logger,
                model_name=best_name, n_mc=30, threshold=0.70)

        # ── C2 ─────────────────────────────────────────────────────────────
        c2_results = None
        if best_name:
            c2_results = run_contribution_2(
                dl_models_dict[best_name][0],
                X_test_sc, y_test, X_train_sc, y_train_b,
                run_dir, logger, n_chunks=20)

        # ── C3 ─────────────────────────────────────────────────────────────
        energy_df = run_contribution_3(
            dl_models_dict, trained_ml, small_loader,
            X_test_sc, y_test, le, run_dir, logger)

        # ── SHAP ──────────────────────────────────────────────────────────
        if SHAP_AVAILABLE and 'Random Forest' in trained_ml:
            try:
                logger.info("\nRunning SHAP analysis...")
                exp = shap.TreeExplainer(trained_ml['Random Forest'])
                sv  = exp.shap_values(X_test_sc[:300])
                plt.figure(figsize=(12, 7))
                shap.summary_plot(
                    sv[0] if isinstance(sv, list) else sv,
                    X_test_sc[:300], feature_names=eng_names,
                    show=False, max_display=20)
                plt.title('SHAP Feature Importance', fontweight='bold')
                plt.tight_layout()
                plt.savefig(run_dir / 'plots' / 'shap_summary.png',
                            dpi=300, bbox_inches='tight')
                plt.close()
                logger.info("✓ Saved: plots/shap_summary.png")
            except Exception as e:
                logger.warning(f"⚠️ SHAP: {e}")

        # ── Summary ────────────────────────────────────────────────────────
        total   = time.time() - pipeline_start
        h, m, s = int(total//3600), int((total%3600)//60), int(total%60)

        logger.info("\n" + "=" * 70)
        logger.info("✅  PIPELINE COMPLETE")
        logger.info("=" * 70)
        logger.info(f"  Total runtime : {h}h {m}m {s}s")
        logger.info(f"  Run directory : {run_dir.resolve()}")
        logger.info(f"\n🏆 Top 5 Models:")
        logger.info(results_df.head(5)[
            ['Model','Type','Test_Accuracy','F1_Score','Train_Time_s']
        ].to_string(index=False))

        summary = {
            'timestamp'      : datetime.now().isoformat(),
            'run_dir'        : str(run_dir.resolve()),
            'device'         : str(DEVICE),
            'total_runtime'  : f"{h}h {m}m {s}s",
            'total_seconds'  : round(total, 1),
            'best_model'     : results_df.iloc[0]['Model'],
            'best_accuracy'  : float(results_df.iloc[0]['Test_Accuracy']),
            'best_f1'        : float(results_df.iloc[0]['F1_Score']),
            'c1_results'     : c1_results,
            'c2_results'     : c2_results,
            'energy_ranking' : energy_df[
                ['model_name','energy_score','deployment_tier']
            ].to_dict(orient='records')
        }
        with open(run_dir / 'results' / 'final_summary.json', 'w') as f:
            json.dump(summary, f, indent=4)
        logger.info("✓ Saved: results/final_summary.json")

        return results_df, energy_df, summary

    except Exception as e:
        logger.critical(f"\n💥 PIPELINE CRASHED: {e}")
        logger.critical(traceback.format_exc())
        raise


if __name__ == '__main__':
    results_df, energy_df, summary = main()
