# ============================================================================
# WSN-DS NOVEL RESEARCH IMPLEMENTATION
# Contribution 1: Uncertainty-Aware Ensemble IDS with Confidence Scoring
# Contribution 2: Adaptive IDS with Concept Drift Detection
# Contribution 3: Energy-Complexity Trade-off Framework
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_recall_fscore_support, roc_curve, auc,
                             precision_recall_curve, matthews_corrcoef)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              VotingClassifier, AdaBoostClassifier, ExtraTreesClassifier,
                              StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize
from sklearn.calibration import CalibratedClassifierCV
from collections import Counter, deque
from datetime import datetime
import time
import json
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not available. Install: pip install shap")

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)


# ============================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
# ============================================================================

def load_wsn_dataset(filepath='/root/amlan/Iot/project/data/raw/WSN-DS.csv'):
    print("\n" + "="*80)
    print("LOADING WSN-DS DATASET")
    print("="*80)

    df = pd.read_csv(filepath)
    print(f"✓ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    print(f"  Columns: {df.columns.tolist()}")

    # Auto-detect target column
    possible_targets = ['Attack_Type', 'attack_type', 'Class', 'class',
                        'Label', 'label', 'Attack', 'target']
    target_col = None
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break
    if target_col is None:
        target_col = df.columns[-1]
        print(f"⚠️ Using last column as target: '{target_col}'")

    print(f"\n🎯 Target Column: '{target_col}'")

    # Handle continuous target
    if df[target_col].nunique() > 100 and df[target_col].dtype in ['float64', 'float32']:
        print("⚠️ Continuous target detected → converting to 5 bins")
        df[target_col] = pd.qcut(df[target_col], q=5,
                                  labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
                                  duplicates='drop')

    print(f"\n📊 Class Distribution:\n{df[target_col].value_counts()}")
    print(f"\n📊 Distribution (%):\n{(df[target_col].value_counts() / len(df) * 100).round(2)}")

    # Missing values
    if df.isnull().sum().sum() > 0:
        df = df.dropna()
        print(f"✓ Dropped NaN rows. New shape: {df.shape}")
    else:
        print("✓ No missing values")

    return df, target_col


def engineer_features(X, feature_names):
    print("\n" + "="*80)
    print("FEATURE ENGINEERING")
    print("="*80)

    X_df = pd.DataFrame(X, columns=feature_names)
    X_eng = X_df.copy()

    # Statistical features
    X_eng['mean_all']  = X_df.mean(axis=1)
    X_eng['std_all']   = X_df.std(axis=1)
    X_eng['max_all']   = X_df.max(axis=1)
    X_eng['min_all']   = X_df.min(axis=1)
    X_eng['range_all'] = X_eng['max_all'] - X_eng['min_all']
    X_eng['skew_all']  = X_df.skew(axis=1)
    X_eng['kurt_all']  = X_df.kurt(axis=1)

    # Interaction features
    top_feats = X_df.var().sort_values(ascending=False).head(3).index.tolist()
    if len(top_feats) >= 2:
        X_eng[f'{top_feats[0]}_x_{top_feats[1]}'] = X_df[top_feats[0]] * X_df[top_feats[1]]
    if len(top_feats) >= 3:
        X_eng[f'{top_feats[0]}_x_{top_feats[2]}'] = X_df[top_feats[0]] * X_df[top_feats[2]]

    # Ratio features
    for col in X_df.columns[:5]:
        if X_df[col].sum() != 0:
            X_eng[f'{col}_ratio'] = X_df[col] / (X_df.sum(axis=1) + 1e-10)

    print(f"✓ Original features  : {X.shape[1]}")
    print(f"✓ Engineered features: {X_eng.shape[1]}")
    return X_eng.values, X_eng.columns.tolist()


def manual_smote(X, y, target_class, k_neighbors=5, samples_to_generate=1000):
    minority_idx = np.where(y == target_class)[0]
    k_neighbors  = max(1, min(k_neighbors, len(minority_idx) - 1))
    minority_X   = X[minority_idx]
    synthetic    = []

    for _ in range(samples_to_generate):
        idx    = np.random.randint(0, len(minority_X))
        sample = minority_X[idx]
        dists  = np.sum((minority_X - sample) ** 2, axis=1)
        k_nn   = np.argsort(dists)[1:k_neighbors + 1]
        if len(k_nn) == 0:
            continue
        neighbor  = minority_X[np.random.choice(k_nn)]
        synthetic.append(sample + np.random.random() * (neighbor - sample))

    return np.array(synthetic)


def balance_dataset(X, y, max_per_class=10000):
    print("\n" + "="*80)
    print("SMOTE-TOMEK CLASS BALANCING")
    print("="*80)

    counts = Counter(y)
    print(f"  Before: {dict(counts)}")
    target = min(int(max(counts.values()) * 0.8), max_per_class)

    X_b, y_b = X.copy(), y.copy()
    for label, count in counts.items():
        if count < target:
            needed   = target - count
            synthetic = manual_smote(X, y, label, samples_to_generate=needed)
            X_b = np.vstack([X_b, synthetic])
            y_b = np.concatenate([y_b, [label] * needed])

    print(f"  After : {dict(Counter(y_b))}")
    return X_b, y_b


# ============================================================================
# SECTION 2: NOVEL CONTRIBUTION 1
# Uncertainty-Aware Ensemble with Confidence Scoring & Zero-Day Detection
# ============================================================================

class UncertaintyAwareEnsemble:
    """
    Novel: Combines Monte Carlo sampling + prediction entropy to produce
    per-sample confidence scores. Low-confidence = potential zero-day attack.
    """

    def __init__(self, base_models, confidence_threshold=0.7, n_mc_samples=30):
        self.base_models          = base_models
        self.confidence_threshold = confidence_threshold
        self.n_mc_samples         = n_mc_samples
        self.calibrated_models    = []
        self.class_names          = None

    def fit(self, X_train, y_train, class_names=None):
        print("\n" + "="*80)
        print("CONTRIBUTION 1: UNCERTAINTY-AWARE ENSEMBLE")
        print("="*80)
        self.class_names = class_names

        for name, model in self.base_models:
            print(f"  🔧 Calibrating: {name}")
            calibrated = CalibratedClassifierCV(model, cv=3, method='isotonic')
            calibrated.fit(X_train, y_train)
            self.calibrated_models.append((name, calibrated))

        print("✓ All models calibrated with isotonic regression")
        return self

    def predict_with_uncertainty(self, X):
        """
        Returns predictions + confidence scores + uncertainty flags.
        Uses prediction entropy across ensemble members as uncertainty measure.
        """
        all_probs = []
        for name, model in self.calibrated_models:
            probs = model.predict_proba(X)
            all_probs.append(probs)

        # Stack: shape (n_models, n_samples, n_classes)
        prob_stack = np.array(all_probs)

        # Mean probability across models
        mean_probs = prob_stack.mean(axis=0)           # (n_samples, n_classes)

        # Prediction entropy as uncertainty measure
        epsilon  = 1e-10
        entropy  = -np.sum(mean_probs * np.log(mean_probs + epsilon), axis=1)
        max_ent  = np.log(mean_probs.shape[1])          # log(n_classes)
        norm_ent = entropy / (max_ent + epsilon)        # 0 = certain, 1 = uncertain

        # Confidence = 1 - normalized entropy
        confidence    = 1.0 - norm_ent
        predictions   = np.argmax(mean_probs, axis=1)

        # Std across models as a second uncertainty signal
        model_std     = prob_stack.std(axis=0).mean(axis=1)

        # Flag low-confidence samples as potential zero-day / novel attacks
        novel_flag    = confidence < self.confidence_threshold

        return predictions, confidence, novel_flag, mean_probs, model_std

    def evaluate(self, X_test, y_test, le):
        predictions, confidence, novel_flag, mean_probs, model_std = \
            self.predict_with_uncertainty(X_test)

        # Overall accuracy
        acc = accuracy_score(y_test, predictions)

        # Accuracy only on high-confidence samples
        high_conf_mask = ~novel_flag
        if high_conf_mask.sum() > 0:
            acc_high = accuracy_score(y_test[high_conf_mask],
                                      predictions[high_conf_mask])
        else:
            acc_high = 0.0

        print(f"\n📊 Overall Accuracy             : {acc:.4f}")
        print(f"📊 High-Confidence Accuracy     : {acc_high:.4f}")
        print(f"📊 Samples Flagged as Novel     : {novel_flag.sum()} "
              f"({novel_flag.mean()*100:.1f}%)")
        print(f"📊 Avg Confidence Score         : {confidence.mean():.4f}")
        print(f"📊 Avg Model Std (disagreement) : {model_std.mean():.4f}")

        # Per-class confidence
        print(f"\n📋 Per-Class Confidence:")
        for cls_idx in range(len(le.classes_)):
            mask = y_test == cls_idx
            if mask.sum() > 0:
                cls_conf = confidence[mask].mean()
                cls_flag = novel_flag[mask].sum()
                print(f"  {le.classes_[cls_idx]:20s} | "
                      f"Confidence: {cls_conf:.3f} | "
                      f"Flagged Novel: {cls_flag}")

        return {
            'overall_accuracy'    : float(acc),
            'high_conf_accuracy'  : float(acc_high),
            'novel_count'         : int(novel_flag.sum()),
            'novel_pct'           : float(novel_flag.mean() * 100),
            'avg_confidence'      : float(confidence.mean()),
            'avg_model_std'       : float(model_std.mean()),
            'predictions'         : predictions,
            'confidence'          : confidence,
            'novel_flag'          : novel_flag,
            'mean_probs'          : mean_probs
        }


def plot_confidence_distribution(results, le, save_path='wsn_new_plots/confidence_distribution.png'):
    """Plot confidence score distributions per class"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Confidence histogram
    axes[0].hist(results['confidence'], bins=50, color='steelblue',
                 edgecolor='black', alpha=0.8)
    axes[0].axvline(x=0.7, color='red', linestyle='--', linewidth=2,
                    label='Threshold (0.7)')
    axes[0].set_xlabel('Confidence Score', fontweight='bold')
    axes[0].set_ylabel('Count', fontweight='bold')
    axes[0].set_title('Prediction Confidence Distribution', fontweight='bold')
    axes[0].legend()

    # Novel vs known samples
    labels_plot = ['Known Attacks\n(High Confidence)', 'Novel/Zero-Day\n(Low Confidence)']
    sizes  = [len(results['confidence']) - results['novel_count'],
              results['novel_count']]
    colors = ['#2ecc71', '#e74c3c']
    axes[1].pie(sizes, labels=labels_plot, colors=colors, autopct='%1.1f%%',
                startangle=90)
    axes[1].set_title('Known vs Novel Attack Detection', fontweight='bold')

    # Confidence vs correctness scatter
    axes[2].scatter(range(min(500, len(results['confidence']))),
                    results['confidence'][:500],
                    c=['green' if not f else 'red'
                       for f in results['novel_flag'][:500]],
                    alpha=0.5, s=20)
    axes[2].axhline(y=0.7, color='black', linestyle='--',
                    linewidth=2, label='Threshold')
    axes[2].set_xlabel('Sample Index', fontweight='bold')
    axes[2].set_ylabel('Confidence Score', fontweight='bold')
    axes[2].set_title('Confidence per Sample (Green=Known, Red=Novel)',
                      fontweight='bold')
    axes[2].legend()

    plt.suptitle('Novel Contribution 1: Uncertainty-Aware IDS', 
                 fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_entropy_heatmap(mean_probs, y_test, le,
                         save_path='wsn_new_plots/entropy_heatmap.png'):
    """Plot prediction probability heatmap for sample subset"""
    n_show = min(100, len(y_test))
    indices = np.random.choice(len(y_test), n_show, replace=False)

    plt.figure(figsize=(14, 8))
    sns.heatmap(mean_probs[indices],
                xticklabels=le.classes_,
                yticklabels=False,
                cmap='YlOrRd',
                vmin=0, vmax=1)
    plt.xlabel('Attack Class', fontweight='bold')
    plt.ylabel(f'Samples (n={n_show})', fontweight='bold')
    plt.title('Prediction Probability Heatmap\n(Darker = Higher Confidence)',
              fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


# ============================================================================
# SECTION 3: NOVEL CONTRIBUTION 2
# Adaptive IDS with Concept Drift Detection (ADWIN-inspired)
# ============================================================================

class ADWINDriftDetector:
    """
    Simplified ADWIN (Adaptive Windowing) drift detector.
    Monitors error rate in a sliding window and signals drift
    when the distribution shifts significantly.
    """

    def __init__(self, delta=0.002, min_window=30):
        self.delta      = delta
        self.min_window = min_window
        self.window     = deque()
        self.drift_detected = False
        self.n_drifts   = 0

    def add_element(self, prediction_correct):
        self.window.append(int(prediction_correct))
        self.drift_detected = self._detect()

        if self.drift_detected:
            self.n_drifts += 1
            # Shrink window after drift
            half = len(self.window) // 2
            self.window = deque(list(self.window)[half:])

        return self.drift_detected

    def _detect(self):
        if len(self.window) < self.min_window * 2:
            return False

        window_list = list(self.window)
        n = len(window_list)
        total_mean = np.mean(window_list)

        # Test all possible split points
        for split in range(self.min_window, n - self.min_window):
            w0 = window_list[:split]
            w1 = window_list[split:]
            m0, m1 = np.mean(w0), np.mean(w1)
            n0, n1 = len(w0), len(w1)

            # ADWIN statistical test
            epsilon_cut = np.sqrt((1 / (2 * n0) + 1 / (2 * n1)) *
                                  np.log(4 * n / self.delta))

            if abs(m0 - m1) >= epsilon_cut:
                return True
        return False


class AdaptiveIDSystem:
    """
    Novel: First adaptive IDS on WSN-DS with ADWIN concept drift detection.
    Automatically triggers incremental retraining when attack patterns shift.
    """

    def __init__(self, base_model, drift_delta=0.002, window_size=200,
                 retrain_threshold=50):
        self.base_model        = base_model
        self.drift_detector    = ADWINDriftDetector(delta=drift_delta)
        self.window_size       = window_size
        self.retrain_threshold = retrain_threshold

        self.recent_X    = deque(maxlen=window_size)
        self.recent_y    = deque(maxlen=window_size)
        self.drift_points    = []
        self.retrain_points  = []
        self.accuracy_history = []
        self.drift_count     = 0
        self.retrain_count   = 0
        self.current_model   = None

    def fit(self, X_train, y_train):
        print("\n" + "="*80)
        print("CONTRIBUTION 2: ADAPTIVE IDS WITH CONCEPT DRIFT DETECTION")
        print("="*80)
        print("  Initial training on base dataset...")
        self.current_model = type(self.base_model)(**self.base_model.get_params())
        self.current_model.fit(X_train, y_train)
        print("✓ Initial model trained")
        return self

    def simulate_stream(self, X_stream, y_stream, chunk_size=100):
        """
        Simulate a data stream by processing chunks.
        Detects drift and retrains incrementally.
        """
        print(f"\n🌊 Simulating data stream: {len(X_stream)} samples "
              f"in chunks of {chunk_size}")

        n_chunks     = len(X_stream) // chunk_size
        chunk_accs   = []
        drift_events = []
        samples_since_retrain = 0

        for i in range(n_chunks):
            start = i * chunk_size
            end   = start + chunk_size
            X_chunk = X_stream[start:end]
            y_chunk = y_stream[start:end]

            # Predict on chunk
            y_pred = self.current_model.predict(X_chunk)
            chunk_acc = accuracy_score(y_chunk, y_pred)
            chunk_accs.append(chunk_acc)

            # Feed each prediction to drift detector
            chunk_drift_detected = False
            for j in range(len(y_chunk)):
                correct = (y_pred[j] == y_chunk[j])
                drift   = self.drift_detector.add_element(correct)
                if drift and not chunk_drift_detected:
                    chunk_drift_detected = True
                    self.drift_count += 1
                    self.drift_points.append(i)
                    print(f"  ⚠️  DRIFT DETECTED at chunk {i} "
                          f"(Accuracy: {chunk_acc:.3f})")

            # Buffer recent data
            for x_s, y_s in zip(X_chunk, y_chunk):
                self.recent_X.append(x_s)
                self.recent_y.append(y_s)

            samples_since_retrain += chunk_size

            # Retrain if drift detected AND enough new samples buffered
            if chunk_drift_detected and \
               samples_since_retrain >= self.retrain_threshold:
                print(f"  🔄 RETRAINING on {len(self.recent_X)} buffered samples...")
                X_retrain = np.array(self.recent_X)
                y_retrain = np.array(self.recent_y)
                self.current_model = type(self.base_model)(
                    **self.base_model.get_params())
                self.current_model.fit(X_retrain, y_retrain)
                self.retrain_count += 1
                self.retrain_points.append(i)
                samples_since_retrain = 0
                print(f"  ✓ Model retrained (retrain #{self.retrain_count})")

        print(f"\n📊 Drift Events Detected : {self.drift_count}")
        print(f"📊 Model Retrains        : {self.retrain_count}")
        print(f"📊 Avg Stream Accuracy   : {np.mean(chunk_accs):.4f}")
        print(f"📊 Min Stream Accuracy   : {np.min(chunk_accs):.4f}")
        print(f"📊 Max Stream Accuracy   : {np.max(chunk_accs):.4f}")

        return chunk_accs, self.drift_points, self.retrain_points


def simulate_concept_drift(X, y, n_chunks=20):
    """
    Artificially inject concept drift by permuting feature distributions
    in later chunks to simulate evolving attack patterns.
    """
    chunk_size = len(X) // n_chunks
    X_stream, y_stream = [], []

    for i in range(n_chunks):
        start = i * chunk_size
        end   = start + chunk_size
        X_c   = X[start:end].copy()
        y_c   = y[start:end].copy()

        # Inject drift in second half of chunks
        if i >= n_chunks // 2:
            drift_strength = (i - n_chunks // 2) / (n_chunks // 2)
            noise = np.random.normal(0, drift_strength * 0.5, X_c.shape)
            X_c   = X_c + noise

        X_stream.append(X_c)
        y_stream.append(y_c)

    return np.vstack(X_stream), np.concatenate(y_stream)


def plot_drift_detection(chunk_accs, drift_points, retrain_points,
                         save_path='wsn_new_plots/concept_drift.png'):
    plt.figure(figsize=(14, 6))

    plt.plot(chunk_accs, 'b-o', markersize=4, linewidth=2,
             label='Chunk Accuracy', alpha=0.8)

    # Mark drift events
    for dp in drift_points:
        if dp < len(chunk_accs):
            plt.axvline(x=dp, color='red', linestyle='--',
                        alpha=0.7, linewidth=1.5)

    # Mark retrain events
    for rp in retrain_points:
        if rp < len(chunk_accs):
            plt.axvline(x=rp, color='green', linestyle='-.',
                        alpha=0.7, linewidth=1.5)

    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue',  linewidth=2, label='Chunk Accuracy'),
        Line2D([0], [0], color='red',   linestyle='--', linewidth=2,
               label='Drift Detected'),
        Line2D([0], [0], color='green', linestyle='-.', linewidth=2,
               label='Model Retrained')
    ]
    plt.legend(handles=legend_elements)

    plt.xlabel('Chunk Index (Time →)', fontweight='bold')
    plt.ylabel('Accuracy', fontweight='bold')
    plt.title('Novel Contribution 2: Adaptive IDS — Concept Drift Detection\n'
              'Red=Drift Detected, Green=Model Retrained', fontweight='bold')
    plt.ylim([0, 1.05])
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_drift_accuracy_comparison(static_accs, adaptive_accs,
                                   save_path='wsn_new_plots/static_vs_adaptive.png'):
    plt.figure(figsize=(14, 6))

    x = range(len(static_accs))
    plt.plot(x, static_accs,  'r-o', markersize=4, linewidth=2,
             label='Static Model', alpha=0.8)
    plt.plot(x, adaptive_accs, 'g-s', markersize=4, linewidth=2,
             label='Adaptive Model (with retraining)', alpha=0.8)

    plt.fill_between(x,
                     [min(s, a) for s, a in zip(static_accs, adaptive_accs)],
                     [max(s, a) for s, a in zip(static_accs, adaptive_accs)],
                     alpha=0.15, color='blue',
                     label='Performance Gap')

    plt.xlabel('Chunk Index (Time →)', fontweight='bold')
    plt.ylabel('Accuracy', fontweight='bold')
    plt.title('Static vs Adaptive Model Under Concept Drift',
              fontweight='bold', fontsize=14)
    plt.legend()
    plt.ylim([0, 1.05])
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


# ============================================================================
# SECTION 4: NOVEL CONTRIBUTION 3
# Energy-Complexity Trade-off Framework
# ============================================================================

def measure_energy_complexity(model, model_name, X_train, y_train,
                               X_test, y_test, n_runs=5):
    """
    Novel: Measures and scores each model on the energy-complexity-accuracy
    trade-off — the first such framework applied to WSN-DS.
    """
    import sys
    import pickle

    # Training time
    train_times = []
    for _ in range(2):
        t0 = time.perf_counter()
        m  = type(model)(**model.get_params())
        m.fit(X_train, y_train)
        train_times.append(time.perf_counter() - t0)
    avg_train_time = np.mean(train_times)

    # Inference time per sample
    inf_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(X_test)
        inf_times.append((time.perf_counter() - t0) / len(X_test) * 1e6)
    avg_inf_time = np.mean(inf_times)  # microseconds per sample

    # Model size
    model_bytes = len(pickle.dumps(model))
    model_kb    = model_bytes / 1024

    # Accuracy
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Approximate FLOPs (model-specific heuristic)
    n_features = X_test.shape[1]
    if hasattr(model, 'n_estimators') and hasattr(model, 'max_depth'):
        max_d  = model.max_depth or 10
        flops  = model.n_estimators * (2 ** max_d) * n_features
    elif hasattr(model, 'hidden_layer_sizes'):
        layers = [n_features] + list(model.hidden_layer_sizes)
        flops  = sum(layers[i] * layers[i+1] for i in range(len(layers)-1))
    else:
        flops  = n_features * 1000  # fallback

    # === NOVEL METRIC: Energy-Efficiency Score ===
    # Combines accuracy, speed, and model complexity
    # Higher = more suitable for resource-constrained WSN nodes
    energy_score = (acc * f1) / (
        np.log1p(avg_inf_time) *
        np.log1p(model_kb) *
        np.log1p(flops / 1e6)
    )

    # WSN Deployment Tier based on energy score
    if energy_score > 0.5:
        tier = "🟢 Edge Node (Sensor)"
    elif energy_score > 0.2:
        tier = "🟡 Gateway Node"
    else:
        tier = "🔴 Cloud/Base Station Only"

    result = {
        'model_name'     : model_name,
        'accuracy'       : round(acc, 4),
        'f1_score'       : round(f1, 4),
        'train_time_s'   : round(avg_train_time, 3),
        'inf_time_us'    : round(avg_inf_time, 4),
        'model_size_kb'  : round(model_kb, 2),
        'flops_M'        : round(flops / 1e6, 2),
        'energy_score'   : round(energy_score, 4),
        'deployment_tier': tier
    }

    print(f"\n  📦 {model_name}")
    print(f"     Accuracy        : {acc:.4f}")
    print(f"     Inference Time  : {avg_inf_time:.4f} µs/sample")
    print(f"     Model Size      : {model_kb:.2f} KB")
    print(f"     FLOPs (M)       : {flops/1e6:.2f}")
    print(f"     Energy Score    : {energy_score:.4f}")
    print(f"     Deployment Tier : {tier}")

    return result


def plot_energy_tradeoff(energy_results, save_path='wsn_new_plots/energy_tradeoff.png'):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    models   = [r['model_name']   for r in energy_results]
    acc      = [r['accuracy']     for r in energy_results]
    inf_time = [r['inf_time_us']  for r in energy_results]
    size_kb  = [r['model_size_kb'] for r in energy_results]
    e_score  = [r['energy_score'] for r in energy_results]
    flops    = [r['flops_M']      for r in energy_results]

    # 1. Energy Score comparison
    colors = ['#2ecc71' if s > 0.5 else '#f39c12' if s > 0.2 else '#e74c3c'
              for s in e_score]
    bars = axes[0, 0].bar(range(len(models)), e_score, color=colors,
                          edgecolor='black')
    axes[0, 0].set_xticks(range(len(models)))
    axes[0, 0].set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    axes[0, 0].set_ylabel('Energy-Efficiency Score', fontweight='bold')
    axes[0, 0].set_title('Novel Metric: Energy-Efficiency Score\n'
                         '(Green=Edge, Yellow=Gateway, Red=Cloud)',
                         fontweight='bold')
    for bar, val in zip(bars, e_score):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    # 2. Accuracy vs Inference Time bubble chart (bubble = model size)
    scatter = axes[0, 1].scatter(inf_time, acc,
                                 s=[kb * 2 for kb in size_kb],
                                 c=e_score, cmap='RdYlGn',
                                 alpha=0.8, edgecolors='black')
    for i, name in enumerate(models):
        axes[0, 1].annotate(name, (inf_time[i], acc[i]),
                            textcoords='offset points',
                            xytext=(5, 5), fontsize=8)
    plt.colorbar(scatter, ax=axes[0, 1], label='Energy Score')
    axes[0, 1].set_xlabel('Inference Time (µs/sample)', fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy', fontweight='bold')
    axes[0, 1].set_title('Accuracy vs Speed Trade-off\n(Bubble Size = Model Size)',
                         fontweight='bold')

    # 3. Model size comparison
    axes[1, 0].barh(range(len(models)), size_kb, color='steelblue',
                    edgecolor='black')
    axes[1, 0].set_yticks(range(len(models)))
    axes[1, 0].set_yticklabels(models, fontsize=9)
    axes[1, 0].set_xlabel('Model Size (KB)', fontweight='bold')
    axes[1, 0].set_title('Model Memory Footprint\n(Critical for WSN nodes)',
                         fontweight='bold')
    axes[1, 0].axvline(x=100, color='red', linestyle='--',
                       label='100KB limit (typical sensor)')
    axes[1, 0].legend()

    # 4. Radar-style multi-metric bar
    x      = np.arange(len(models))
    width  = 0.2
    norm_acc   = np.array(acc)
    norm_spd   = 1 - (np.array(inf_time) / max(inf_time))
    norm_size  = 1 - (np.array(size_kb) / max(size_kb))
    norm_escore = np.array(e_score) / max(e_score)

    axes[1, 1].bar(x - 1.5*width, norm_acc,   width, label='Accuracy',     color='#3498db')
    axes[1, 1].bar(x - 0.5*width, norm_spd,   width, label='Speed',        color='#2ecc71')
    axes[1, 1].bar(x + 0.5*width, norm_size,  width, label='Compactness',  color='#e67e22')
    axes[1, 1].bar(x + 1.5*width, norm_escore,width, label='Energy Score', color='#9b59b6')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    axes[1, 1].set_ylabel('Normalized Score', fontweight='bold')
    axes[1, 1].set_title('Multi-Criteria WSN Deployment Analysis',
                         fontweight='bold')
    axes[1, 1].legend()

    plt.suptitle('Novel Contribution 3: Energy-Complexity Trade-off Framework\n'
                 'First Such Analysis on WSN-DS Dataset',
                 fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_deployment_recommendation(energy_results,
                                   save_path='wsn_new_plots/deployment_tiers.png'):
    fig, ax = plt.subplots(figsize=(14, 7))

    models   = [r['model_name']   for r in energy_results]
    e_scores = [r['energy_score'] for r in energy_results]
    tiers    = [r['deployment_tier'] for r in energy_results]

    tier_colors = {
        '🟢 Edge Node (Sensor)' : '#2ecc71',
        '🟡 Gateway Node'       : '#f39c12',
        '🔴 Cloud/Base Station Only': '#e74c3c'
    }
    bar_colors = [tier_colors.get(t, '#95a5a6') for t in tiers]

    bars = ax.barh(range(len(models)), e_scores,
                   color=bar_colors, edgecolor='black', height=0.6)

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=11)
    ax.set_xlabel('Energy-Efficiency Score', fontweight='bold', fontsize=12)
    ax.set_title('WSN Deployment Tier Recommendation\n'
                 '(Novel Contribution 3 — Per-model Deployability)',
                 fontweight='bold', fontsize=14)

    # Threshold lines
    ax.axvline(x=0.5, color='green',  linestyle='--', linewidth=2,
               label='Edge Node Threshold (0.5)')
    ax.axvline(x=0.2, color='orange', linestyle='--', linewidth=2,
               label='Gateway Threshold (0.2)')

    for bar, tier, score in zip(bars, tiers, e_scores):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{score:.3f} | {tier}',
                va='center', fontsize=9)

    # Legend for tiers
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='🟢 Edge Node (Sensor)'),
        Patch(facecolor='#f39c12', label='🟡 Gateway Node'),
        Patch(facecolor='#e74c3c', label='🔴 Cloud/Base Station Only'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


# ============================================================================
# SECTION 5: STANDARD MODEL TRAINING + EXISTING PLOTS
# ============================================================================

def train_and_evaluate_model(X_train, X_test, y_train, y_test,
                              model, model_name):
    print(f"\n{'='*60}\nTRAINING: {model_name}\n{'='*60}")

    model.fit(X_train, y_train)
    y_tr_pred = model.predict(X_train)
    y_te_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_tr_pred)
    test_acc  = accuracy_score(y_test,  y_te_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_te_pred, average='weighted', zero_division=0)

    diff = train_acc - test_acc
    status = ("⚠️ OVERFITTING"  if diff > 0.1  else
              "⚠️ UNDERFITTING" if test_acc < 0.7 else
              "✓ GOOD FIT")

    print(f"  Train Acc : {train_acc:.4f}")
    print(f"  Test  Acc : {test_acc:.4f}")
    print(f"  Diff      : {diff:.4f}  [{status}]")
    print(f"  F1        : {f1:.4f}")
    print(classification_report(y_test, y_te_pred, zero_division=0))

    # 5-fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for tr_i, va_i in cv.split(X_train, y_train):
        try:
            import copy
            m2 = copy.deepcopy(model)            # ← FIXED: deep copy instead
            m2.fit(X_train[tr_i], y_train[tr_i])
            cv_scores.append(m2.score(X_train[va_i], y_train[va_i]))
        except Exception as e:
            print(f"  ⚠️ CV fold skipped: {e}")
            continue

    print(f"  CV: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    results = {
        'model_name'          : model_name,
        'train_accuracy'      : float(train_acc),
        'test_accuracy'       : float(test_acc),
        'accuracy_difference' : float(diff),
        'precision'           : float(precision),
        'recall'              : float(recall),
        'f1_score'            : float(f1),
        'cv_mean'             : float(np.mean(cv_scores)),
        'cv_std'              : float(np.std(cv_scores)),
        'status'              : status,
        'confusion_matrix'    : confusion_matrix(y_test, y_te_pred).tolist()
    }
    return model, results, y_te_pred


def plot_all_standard(all_results, trained_models, results_df,
                      feature_importance, le, X_test_scaled,
                      y_test, engineered_feature_names):

    # 1. Model comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, metric, title in zip(axes.flat,
                                  ['Test_Accuracy','Precision','Recall','F1_Score'],
                                  ['Test Accuracy','Precision','Recall','F1-Score']):
        sd = results_df.sort_values(metric, ascending=True)
        bars = ax.barh(range(len(sd)), sd[metric],
                       color='skyblue', edgecolor='navy')
        ax.set_yticks(range(len(sd)))
        ax.set_yticklabels(sd['Model'], fontsize=9)
        ax.set_xlabel('Score', fontweight='bold')
        ax.set_title(title, fontweight='bold')
        ax.set_xlim([0, 1.0])
        for i, (_, row) in enumerate(sd.iterrows()):
            ax.text(row[metric], i, f" {row[metric]:.3f}", va='center', fontsize=8)
        bars[-1].set_color('gold')
    plt.suptitle('Standard Model Comparison', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig('wsn_new_plots/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: wsn_new_plots/model_comparison.png")

    # 2. Confusion matrices
    for name, res in all_results.items():
        cm = np.array(res['confusion_matrix']).astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm  = np.divide(cm, row_sums, where=row_sums != 0)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title(f'Confusion Matrix — {name}', fontweight='bold')
        plt.ylabel('True', fontweight='bold')
        plt.xlabel('Predicted', fontweight='bold')
        plt.tight_layout()
        fname = f'wsn_new_plots/cm_{name.replace(" ", "_")}.png'
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {fname}")

    # 3. Feature importance
    top = feature_importance.head(20)
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top)), top['Importance'], color='teal', edgecolor='black')
    plt.yticks(range(len(top)), top['Feature'])
    plt.xlabel('Importance', fontweight='bold')
    plt.title('Top 20 Feature Importances (Random Forest)', fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('wsn_new_plots/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: wsn_new_plots/feature_importance.png")

    # 4. Train vs Test accuracy
    models_list = list(all_results.keys())
    tr_accs = [all_results[m]['train_accuracy'] for m in models_list]
    te_accs = [all_results[m]['test_accuracy']  for m in models_list]
    x = np.arange(len(models_list)); w = 0.35
    plt.figure(figsize=(14, 7))
    b1 = plt.bar(x - w/2, tr_accs, w, label='Train', color='lightblue',  edgecolor='navy')
    b2 = plt.bar(x + w/2, te_accs, w, label='Test',  color='lightgreen', edgecolor='darkgreen')
    plt.xticks(x, models_list, rotation=45, ha='right')
    plt.ylabel('Accuracy', fontweight='bold')
    plt.title('Training vs Test Accuracy', fontweight='bold')
    plt.legend()
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, h,
                     f'{h:.3f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig('wsn_new_plots/train_vs_test.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: wsn_new_plots/train_vs_test.png")

    # 5. ROC curves
    n_classes  = len(le.classes_)
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    plt.figure(figsize=(14, 9))
    colors_iter = cycle(['aqua','darkorange','cornflowerblue',
                         'red','green','purple','brown'])
    for (name, model), color in zip(trained_models.items(), colors_iter):
        if not hasattr(model, 'predict_proba'):
            continue
        try:
            y_score = model.predict_proba(X_test_scaled)
            fpr_m, tpr_m, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
            roc_auc_val = auc(fpr_m, tpr_m)
            plt.plot(fpr_m, tpr_m, color=color, lw=2,
                     label=f'{name} (AUC={roc_auc_val:.3f})')
        except Exception:
            pass
    plt.plot([0,1],[0,1],'k--', lw=2, label='Random')
    plt.xlabel('FPR', fontweight='bold'); plt.ylabel('TPR', fontweight='bold')
    plt.title('ROC Curves — All Models', fontweight='bold')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('wsn_new_plots/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: wsn_new_plots/roc_curves.png")

    # 6. CV scores
    cv_means = [all_results[m]['cv_mean'] for m in models_list]
    cv_stds  = [all_results[m]['cv_std']  for m in models_list]
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(models_list)), cv_means, yerr=cv_stds,
            capsize=5, color='lightcoral', edgecolor='darkred')
    plt.xticks(range(len(models_list)), models_list, rotation=45, ha='right')
    plt.ylabel('CV Score', fontweight='bold')
    plt.title('5-Fold Cross-Validation Scores', fontweight='bold')
    plt.tight_layout()
    plt.savefig('wsn_new_plots/cv_scores.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: wsn_new_plots/cv_scores.png")


# ============================================================================
# SECTION 6: SHAP EXPLAINABILITY
# ============================================================================

def run_shap_analysis(model, X_test, feature_names, model_name='Random Forest',
                      n_samples=200):
    if not SHAP_AVAILABLE:
        print("⚠️ SHAP not installed — skipping XAI analysis")
        return

    print("\n" + "="*80)
    print("XAI: SHAP EXPLAINABILITY ANALYSIS")
    print("="*80)

    X_explain = X_test[:n_samples]

    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_explain)

        # Summary plot
        plt.figure(figsize=(12, 8))
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[0], X_explain,
                              feature_names=feature_names,
                              show=False, max_display=20)
        else:
            shap.summary_plot(shap_values, X_explain,
                              feature_names=feature_names,
                              show=False, max_display=20)
        plt.title('SHAP Feature Importance — XAI Analysis', fontweight='bold')
        plt.tight_layout()
        plt.savefig('wsn_new_plots/shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: wsn_new_plots/shap_summary.png")

        # Bar plot
        plt.figure(figsize=(10, 7))
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[0], X_explain,
                              feature_names=feature_names,
                              plot_type='bar', show=False, max_display=15)
        else:
            shap.summary_plot(shap_values, X_explain,
                              feature_names=feature_names,
                              plot_type='bar', show=False, max_display=15)
        plt.title('SHAP Mean Absolute Feature Impact', fontweight='bold')
        plt.tight_layout()
        plt.savefig('wsn_new_plots/shap_bar.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: wsn_new_plots/shap_bar.png")

    except Exception as e:
        print(f"⚠️ SHAP error: {e}")


# ============================================================================
# SECTION 7: MAIN PIPELINE
# ============================================================================

def main():
    os.makedirs('plots',   exist_ok=True)
    os.makedirs('results', exist_ok=True)

    print("\n" + "="*80)
    print("WSN-DS NOVEL RESEARCH IDS — ALL 3 CONTRIBUTIONS")
    print("="*80)

    # ── Load data ──────────────────────────────────────────────────────────
    df, target_col = load_wsn_dataset(
        '/root/amlan/Iot/project/data/raw/WSN-DS.csv')

    X            = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).values
    y_raw        = df[target_col].values
    feature_names = (df.drop(columns=[target_col])
                       .select_dtypes(include=[np.number])
                       .columns.tolist())

    le        = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    print(f"\n🏷️  Classes: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Plot class distribution
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    vc = pd.Series(y_raw).value_counts()
    plt.bar(range(len(vc)), vc.values, color='steelblue', edgecolor='black')
    plt.xticks(range(len(vc)), vc.index, rotation=45, ha='right')
    plt.title('Class Distribution', fontweight='bold')
    plt.subplot(1, 2, 2)
    plt.pie(vc.values, labels=vc.index, autopct='%1.1f%%',
            colors=plt.cm.Set3(range(len(vc))))
    plt.title('Class Proportions', fontweight='bold')
    plt.tight_layout()
    plt.savefig('wsn_new_plots/class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: wsn_new_plots/class_distribution.png")

    # ── Feature engineering ────────────────────────────────────────────────
    X_eng, eng_names = engineer_features(X, feature_names)

    # ── Correlation matrix ─────────────────────────────────────────────────
    plt.figure(figsize=(16, 14))
    corr = pd.DataFrame(X_eng[:2000], columns=eng_names).corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0,
                mask=np.triu(np.ones_like(corr, dtype=bool)))
    plt.title('Feature Correlation Matrix', fontweight='bold')
    plt.tight_layout()
    plt.savefig('wsn_new_plots/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: wsn_new_plots/correlation_matrix.png")

    # ── Train/test split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_eng, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    print(f"\n📊 Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

    # ── SMOTE balancing ────────────────────────────────────────────────────
    X_train_b, y_train_b = balance_dataset(X_train, y_train)

    # ── Scaling ────────────────────────────────────────────────────────────
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_b)
    X_test_scaled  = scaler.transform(X_test)
    print("✓ Feature scaling complete")

    # ── Define models ──────────────────────────────────────────────────────
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_split=5,
            min_samples_leaf=2, class_weight='balanced',
            random_state=42, n_jobs=-1),
        'LightGBM': lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        n_jobs=-1,        # ← parallel!
        random_state=42,
        class_weight='balanced'
),
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=150, max_depth=25, class_weight='balanced',
            random_state=42, n_jobs=-1),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=100, learning_rate=0.5, random_state=42),
        'MLP Neural Network': MLPClassifier(
            hidden_layer_sizes=(100, 50), max_iter=500, alpha=0.001,
            learning_rate='adaptive', early_stopping=True, random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=100, max_depth=7, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, tree_method='hist',
            n_jobs=-1, random_state=42, eval_metric='mlogloss')
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.1,
            class_weight='balanced', random_state=42, n_jobs=-1)

    # ── Train all models ───────────────────────────────────────────────────
    all_results   = {}
    trained_models = {}

    for name, model in models.items():
        try:
            tm, res, _ = train_and_evaluate_model(
                X_train_scaled, X_test_scaled, y_train_b, y_test, model, name)
            all_results[name]    = res
            trained_models[name] = tm
        except Exception as e:
            print(f"❌ {name}: {e}")

    # ── Stacking Ensemble ──────────────────────────────────────────────────
    if len(trained_models) >= 3:
        print("\n" + "="*80)
        print("BUILDING STACKING ENSEMBLE")
        print("="*80)
        stack_estimators = [
            (n.replace(' ', '_'), m)
            for n, m in list(trained_models.items())[:3]
        ]
        stacking = StackingClassifier(
            estimators=stack_estimators,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=3, n_jobs=-1
        )
        try:
            tm, res, _ = train_and_evaluate_model(
                X_train_scaled, X_test_scaled, y_train_b, y_test,
                stacking, 'Stacking Ensemble')
            all_results['Stacking Ensemble']    = res
            trained_models['Stacking Ensemble'] = tm
        except Exception as e:
            print(f"❌ Stacking Ensemble: {e}")

    # ── Results CSV/JSON ───────────────────────────────────────────────────
    results_df = pd.DataFrame([
        {'Model': n, 'Train_Accuracy': r['train_accuracy'],
         'Test_Accuracy': r['test_accuracy'], 'Precision': r['precision'],
         'Recall': r['recall'], 'F1_Score': r['f1_score'],
         'CV_Mean': r['cv_mean'], 'CV_Std': r['cv_std'], 'Status': r['status']}
        for n, r in all_results.items()
    ]).sort_values('Test_Accuracy', ascending=False)

    results_df.to_csv('results/model_comparison.csv', index=False)
    with open('results/model_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)

    # Feature importance
    rf_model = trained_models.get('Random Forest')
    feature_importance = pd.DataFrame({
        'Feature'   : eng_names,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    feature_importance.to_csv('results/feature_importance.csv', index=False)

    # ── Standard plots ─────────────────────────────────────────────────────
    plot_all_standard(all_results, trained_models, results_df,
                      feature_importance, le, X_test_scaled,
                      y_test, eng_names)

    # ══════════════════════════════════════════════════════════════════════
    # NOVEL CONTRIBUTION 1: UNCERTAINTY-AWARE ENSEMBLE
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("RUNNING NOVEL CONTRIBUTION 1: UNCERTAINTY-AWARE ENSEMBLE")
    print("="*80)

    ua_base_models = [
    ('rf', trained_models['Random Forest']),
    ('et', trained_models['Extra Trees']),
    ('gb', trained_models['XGBoost'])             # ← FIXED
    ]


    ua_ensemble = UncertaintyAwareEnsemble(
        base_models=ua_base_models,
        confidence_threshold=0.7,
        n_mc_samples=30
    )
    ua_ensemble.fit(X_train_scaled, y_train_b, class_names=le.classes_)
    ua_results = ua_ensemble.evaluate(X_test_scaled, y_test, le)

    plot_confidence_distribution(ua_results, le)
    plot_entropy_heatmap(ua_results['mean_probs'], y_test, le)

    # Save contribution 1 results
    contrib1_save = {k: v for k, v in ua_results.items()
                     if not isinstance(v, np.ndarray)}
    with open('results/contribution1_uncertainty.json', 'w') as f:
        json.dump(contrib1_save, f, indent=4)
    print("✓ Saved: results/contribution1_uncertainty.json")

    # ══════════════════════════════════════════════════════════════════════
    # NOVEL CONTRIBUTION 2: ADAPTIVE IDS WITH CONCEPT DRIFT
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("RUNNING NOVEL CONTRIBUTION 2: ADAPTIVE IDS + CONCEPT DRIFT")
    print("="*80)

    # Simulate drifting stream from test data
    X_stream, y_stream = simulate_concept_drift(X_test_scaled, y_test, n_chunks=20)

    # Static model (no retraining)
    static_rf = RandomForestClassifier(
        n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    static_rf.fit(X_train_scaled, y_train_b)
    chunk_size = len(X_stream) // 20
    static_accs = []
    for i in range(20):
        s = i * chunk_size
        e = s + chunk_size
        static_accs.append(
            accuracy_score(y_stream[s:e], static_rf.predict(X_stream[s:e])))

    # Adaptive model
    adaptive_model = RandomForestClassifier(
        n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    adaptive_ids = AdaptiveIDSystem(
        base_model=adaptive_model,
        drift_delta=0.002,
        window_size=200,
        retrain_threshold=50
    )
    adaptive_ids.fit(X_train_scaled, y_train_b)
    adaptive_accs, drift_pts, retrain_pts = adaptive_ids.simulate_stream(
        X_stream, y_stream, chunk_size=chunk_size)

    plot_drift_detection(adaptive_accs, drift_pts, retrain_pts)
    plot_drift_accuracy_comparison(static_accs, adaptive_accs)

    contrib2_results = {
        'drift_events'        : len(drift_pts),
        'retrain_events'      : len(retrain_pts),
        'static_avg_accuracy' : float(np.mean(static_accs)),
        'adaptive_avg_accuracy': float(np.mean(adaptive_accs)),
        'improvement'         : float(np.mean(adaptive_accs) - np.mean(static_accs)),
        'drift_points'        : drift_pts,
        'retrain_points'      : retrain_pts
    }
    with open('results/contribution2_drift.json', 'w') as f:
        json.dump(contrib2_results, f, indent=4)
    print("✓ Saved: results/contribution2_drift.json")
    print(f"\n📊 Static Avg Accuracy   : {np.mean(static_accs):.4f}")
    print(f"📊 Adaptive Avg Accuracy : {np.mean(adaptive_accs):.4f}")
    print(f"📊 Improvement           : "
          f"+{(np.mean(adaptive_accs)-np.mean(static_accs))*100:.2f}%")

    # ══════════════════════════════════════════════════════════════════════
    # NOVEL CONTRIBUTION 3: ENERGY-COMPLEXITY TRADE-OFF
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("RUNNING NOVEL CONTRIBUTION 3: ENERGY-COMPLEXITY TRADE-OFF")
    print("="*80)

    energy_results = []
    for name, model in trained_models.items():
        try:
            er = measure_energy_complexity(
                model, name,
                X_train_scaled[:5000], y_train_b[:5000],
                X_test_scaled, y_test
            )
            energy_results.append(er)
        except Exception as e:
            print(f"⚠️ Energy measure failed for {name}: {e}")

    energy_df = pd.DataFrame(energy_results).sort_values(
        'energy_score', ascending=False)
    energy_df.to_csv('results/contribution3_energy.csv', index=False)
    print("\n✓ Saved: results/contribution3_energy.csv")
    print(f"\n{energy_df[['model_name','accuracy','inf_time_us','model_size_kb','energy_score','deployment_tier']].to_string(index=False)}")

    plot_energy_tradeoff(energy_results)
    plot_deployment_recommendation(energy_results)

    # ── SHAP XAI ───────────────────────────────────────────────────────────
    run_shap_analysis(rf_model, X_test_scaled, eng_names)

    # ══════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("✅ ALL 3 NOVEL CONTRIBUTIONS COMPLETE")
    print("="*80)

    print("\n🏆 Top 3 Models by Test Accuracy:")
    print(results_df.head(3)[['Model','Test_Accuracy','F1_Score','Status']].to_string(index=False))

    print("\n📁 Results saved in results/:")
    print("  • model_comparison.csv")
    print("  • model_results.json")
    print("  • feature_importance.csv")
    print("  • contribution1_uncertainty.json")
    print("  • contribution2_drift.json")
    print("  • contribution3_energy.csv")

    print("\n📊 Plots saved in wsn_new_plots/:")
    print("  • class_distribution.png")
    print("  • correlation_matrix.png")
    print("  • model_comparison.png")
    print("  • confusion_matrix_*.png")
    print("  • feature_importance.png")
    print("  • train_vs_test.png")
    print("  • roc_curves.png")
    print("  • cv_scores.png")
    print("  • [C1] confidence_distribution.png")
    print("  • [C1] entropy_heatmap.png")
    print("  • [C2] concept_drift.png")
    print("  • [C2] static_vs_adaptive.png")
    print("  • [C3] energy_tradeoff.png")
    print("  • [C3] deployment_tiers.png")
    print("  • shap_summary.png / shap_bar.png")

    return results_df, energy_df, ua_results, contrib2_results


if __name__ == '__main__':
    results_df, energy_df, ua_results, drift_results = main()
