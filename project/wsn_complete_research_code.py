# ============================================================================
# WSN-DS INTRUSION DETECTION - RESEARCH IMPLEMENTATION (WITH PLOTS)
# ============================================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                             precision_recall_fscore_support, f1_score, matthews_corrcoef,
                             roc_curve, auc, precision_recall_curve, roc_auc_score)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              VotingClassifier, AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import json
from collections import Counter
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Optional XGBoost import with fallback
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available. Install with: pip install xgboost")


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_class_distribution(y, y_encoded, le, save_path='plots/class_distribution.png'):
    '''Plot class distribution'''
    plt.figure(figsize=(12, 6))
    
    # Original distribution
    plt.subplot(1, 2, 1)
    class_counts = Counter(y)
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    bars = plt.bar(range(len(classes)), counts, color='steelblue', edgecolor='black')
    plt.xlabel('Class', fontweight='bold')
    plt.ylabel('Count', fontweight='bold')
    plt.title('Original Class Distribution', fontweight='bold', fontsize=14)
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    # Percentage distribution
    plt.subplot(1, 2, 2)
    percentages = [count/sum(counts)*100 for count in counts]
    colors = plt.cm.Set3(range(len(classes)))
    plt.pie(percentages, labels=classes, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('Class Distribution (%)', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_confusion_matrix(cm, classes, model_name, save_path):
    '''Plot confusion matrix heatmap'''
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Normalized Count'})
    
    plt.title(f'Confusion Matrix - {model_name}', fontweight='bold', fontsize=14)
    plt.ylabel('True Label', fontweight='bold')
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_roc_curves(models_data, X_test, y_test, le, save_path='plots/roc_curves.png'):
    '''Plot ROC curves for all models'''
    n_classes = len(le.classes_)
    
    # Binarize the output for multi-class
    from sklearn.preprocessing import label_binarize
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    
    plt.figure(figsize=(14, 10))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown'])
    
    for (model_name, model), color in zip(models_data.items(), colors):
        try:
            # Get probability predictions
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)
            else:
                continue
            
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            
            # Plot micro-average ROC curve
            plt.plot(fpr["micro"], tpr["micro"], color=color, lw=2,
                    label=f'{model_name} (AUC = {roc_auc["micro"]:.3f})')
        except Exception as e:
            print(f"⚠️ Could not plot ROC for {model_name}: {str(e)}")
            continue
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title('ROC Curves - All Models', fontweight='bold', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_feature_importance(feature_importance_df, top_n=20, save_path='plots/feature_importance.png'):
    '''Plot feature importance'''
    plt.figure(figsize=(12, 8))
    
    top_features = feature_importance_df.head(top_n)
    
    bars = plt.barh(range(len(top_features)), top_features['Importance'], color='teal', edgecolor='black')
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Importance Score', fontweight='bold')
    plt.ylabel('Feature', fontweight='bold')
    plt.title(f'Top {top_n} Most Important Features', fontweight='bold', fontsize=14)
    plt.gca().invert_yaxis()
    
    # Add value labels
    for i, (idx, row) in enumerate(top_features.iterrows()):
        plt.text(row['Importance'], i, f" {row['Importance']:.4f}", 
                va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_model_comparison(results_df, save_path='plots/model_comparison.png'):
    '''Plot model comparison across metrics'''
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    metrics = ['Test_Accuracy', 'Precision', 'Recall', 'F1_Score']
    titles = ['Test Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for ax, metric, title in zip(axes.flat, metrics, titles):
        sorted_df = results_df.sort_values(metric, ascending=True)
        
        bars = ax.barh(range(len(sorted_df)), sorted_df[metric], 
                       color='skyblue', edgecolor='navy', linewidth=1.5)
        
        ax.set_yticks(range(len(sorted_df)))
        ax.set_yticklabels(sorted_df['Model'], fontsize=10)
        ax.set_xlabel('Score', fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_xlim([0, 1.0])
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (idx, row) in enumerate(sorted_df.iterrows()):
            ax.text(row[metric], i, f" {row[metric]:.3f}", 
                   va='center', fontsize=9, fontweight='bold')
        
        # Color top performer
        bars[-1].set_color('gold')
        bars[-1].set_edgecolor('darkgoldenrod')
    
    plt.suptitle('Model Performance Comparison', fontweight='bold', fontsize=16, y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_cv_scores(all_results, save_path='plots/cv_scores.png'):
    '''Plot cross-validation scores comparison'''
    plt.figure(figsize=(12, 6))
    
    models = list(all_results.keys())
    cv_means = [all_results[m]['cv_mean'] for m in models]
    cv_stds = [all_results[m]['cv_std'] for m in models]
    
    x_pos = np.arange(len(models))
    
    bars = plt.bar(x_pos, cv_means, yerr=cv_stds, capsize=5, 
                   color='lightcoral', edgecolor='darkred', linewidth=1.5,
                   error_kw={'linewidth': 2, 'ecolor': 'black'})
    
    plt.xticks(x_pos, models, rotation=45, ha='right')
    plt.ylabel('Cross-Validation Score', fontweight='bold')
    plt.title('5-Fold Cross-Validation Scores (Mean ± Std)', fontweight='bold', fontsize=14)
    plt.ylim([0, 1.0])
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (mean, std) in enumerate(zip(cv_means, cv_stds)):
        plt.text(i, mean + std + 0.02, f'{mean:.3f}±{std:.3f}', 
                ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_learning_curves(model, X_train, y_train, model_name, save_path):
    '''Plot learning curves to detect overfitting/underfitting'''
    plt.figure(figsize=(10, 6))
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy', random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score', linewidth=2)
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.2, color='blue')
    
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score', linewidth=2)
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                     alpha=0.2, color='red')
    
    plt.xlabel('Training Set Size', fontweight='bold')
    plt.ylabel('Accuracy Score', fontweight='bold')
    plt.title(f'Learning Curves - {model_name}', fontweight='bold', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_training_vs_test_accuracy(all_results, save_path='plots/train_vs_test.png'):
    '''Plot training vs test accuracy to visualize overfitting'''
    plt.figure(figsize=(12, 8))
    
    models = list(all_results.keys())
    train_accs = [all_results[m]['train_accuracy'] for m in models]
    test_accs = [all_results[m]['test_accuracy'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, train_accs, width, label='Training Accuracy', 
                    color='lightblue', edgecolor='navy', linewidth=1.5)
    bars2 = plt.bar(x + width/2, test_accs, width, label='Test Accuracy',
                    color='lightgreen', edgecolor='darkgreen', linewidth=1.5)
    
    plt.xlabel('Models', fontweight='bold')
    plt.ylabel('Accuracy', fontweight='bold')
    plt.title('Training vs Test Accuracy Comparison', fontweight='bold', fontsize=14)
    plt.xticks(x, models, rotation=45, ha='right')
    plt.legend()
    plt.ylim([0, 1.1])
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Add diagonal line for perfect fit
    plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, 
             ls='--', c='red', alpha=0.5, label='Perfect Fit')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_precision_recall_curves(models_data, X_test, y_test, le, save_path='plots/precision_recall.png'):
    '''Plot Precision-Recall curves'''
    n_classes = len(le.classes_)
    
    from sklearn.preprocessing import label_binarize
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    
    plt.figure(figsize=(14, 10))
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'red', 'purple'])
    
    for (model_name, model), color in zip(models_data.items(), colors):
        try:
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)
            else:
                continue
            
            precision = dict()
            recall = dict()
            
            for i in range(n_classes):
                precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
            
            # Compute micro-average
            precision["micro"], recall["micro"], _ = precision_recall_curve(
                y_test_bin.ravel(), y_score.ravel()
            )
            
            plt.plot(recall["micro"], precision["micro"], color=color, lw=2,
                    label=f'{model_name}')
        except Exception as e:
            print(f"⚠️ Could not plot PR curve for {model_name}: {str(e)}")
            continue
    
    plt.xlabel('Recall', fontweight='bold')
    plt.ylabel('Precision', fontweight='bold')
    plt.title('Precision-Recall Curves - All Models', fontweight='bold', fontsize=14)
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_correlation_matrix(X, feature_names, save_path='plots/correlation_matrix.png'):
    '''Plot feature correlation heatmap'''
    plt.figure(figsize=(16, 14))
    
    # Calculate correlation matrix
    X_df = pd.DataFrame(X, columns=feature_names)
    corr_matrix = X_df.corr()
    
    # Plot heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    
    plt.title('Feature Correlation Matrix', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


# ============================================================================
# 1. DATA LOADING AND EXPLORATION
# ============================================================================

def load_wsn_dataset(filepath='/root/amlan/Iot/project/data/raw/WSN-DS.csv'):
    '''Load and explore WSN-DS dataset'''
    print("\n" + "="*80)
    print("LOADING WSN-DS DATASET")
    print("="*80)

    df = pd.read_csv(filepath)
    print(f"✓ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")

    # Display column names for debugging
    print(f"\nColumn names: {df.columns.tolist()}")

    # Identify target column
    possible_targets = ['Attack_Type', 'attack_type', 'Class', 'class', 'Label', 'label', 'Attack', 'target']
    target_col = None
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break
    if target_col is None:
        target_col = df.columns[-1]
        print(f"\n⚠️ No standard target column found. Using last column: '{target_col}'")

    print(f"\nTarget Column: '{target_col}'")
    
    # Check if target is categorical or continuous
    unique_values = df[target_col].nunique()
    print(f"Unique values in target: {unique_values}")
    
    if unique_values > 100:
        print("\n⚠️ WARNING: Target has many unique values. This might be a continuous variable.")
        print("Consider converting to categories or using regression instead.")
        
        # Option to bin continuous values
        if df[target_col].dtype in ['float64', 'float32']:
            print("\n📊 Auto-converting continuous target to 5 categories...")
            df[target_col] = pd.qcut(df[target_col], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'], duplicates='drop')
            print(f"✓ Target converted to categorical with {df[target_col].nunique()} classes")
    
    print(f"\nClass Distribution:")
    class_dist = df[target_col].value_counts()
    print(class_dist)
    print(f"\nClass Distribution (%):")
    print((class_dist / len(df) * 100).round(2))

    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n⚠️ Missing values found:\n{missing[missing > 0]}")
        df = df.dropna()
        print(f"✓ Dropped missing values. New shape: {df.shape}")
    else:
        print("\n✓ No missing values detected")

    return df, target_col


# ============================================================================
# 2. ADVANCED FEATURE ENGINEERING
# ============================================================================

def engineer_features(X, feature_names):
    '''Create advanced derived features'''
    print("\n" + "="*80)
    print("FEATURE ENGINEERING")
    print("="*80)

    X_df = pd.DataFrame(X, columns=feature_names)
    X_engineered = X_df.copy()

    # Statistical features
    X_engineered['mean_all'] = X_df.mean(axis=1)
    X_engineered['std_all'] = X_df.std(axis=1)
    X_engineered['max_all'] = X_df.max(axis=1)
    X_engineered['min_all'] = X_df.min(axis=1)
    X_engineered['range_all'] = X_engineered['max_all'] - X_engineered['min_all']

    # Interaction features (top 3 features with highest variance)
    variances = X_df.var().sort_values(ascending=False)
    top_features = variances.head(3).index.tolist()

    if len(top_features) >= 2:
        X_engineered[f'{top_features[0]}_x_{top_features[1]}'] = X_df[top_features[0]] * X_df[top_features[1]]

    # Ratio features (limit to first 5 to avoid too many features)
    for col in X_df.columns[:5]:
        col_sum = X_df[col].sum()
        if col_sum != 0:
            X_engineered[f'{col}_ratio'] = X_df[col] / (X_df.sum(axis=1) + 1e-10)

    print(f"✓ Original features: {X.shape[1]}")
    print(f"✓ Engineered features: {X_engineered.shape[1]}")
    print(f"✓ New features added: {X_engineered.shape[1] - X.shape[1]}")

    return X_engineered.values, X_engineered.columns.tolist()


# ============================================================================
# 3. MANUAL SMOTE IMPLEMENTATION FOR CLASS IMBALANCE
# ============================================================================

def manual_smote(X, y, target_class, k_neighbors=5, samples_to_generate=1000):
    '''Manual SMOTE implementation'''
    minority_indices = np.where(y == target_class)[0]
    
    if len(minority_indices) < k_neighbors:
        k_neighbors = max(1, len(minority_indices) - 1)
    
    minority_samples = X[minority_indices]

    synthetic_samples = []
    for _ in range(samples_to_generate):
        idx = np.random.randint(0, len(minority_samples))
        sample = minority_samples[idx]

        # Find k nearest neighbors
        distances = np.sum((minority_samples - sample) ** 2, axis=1)
        k_nearest = np.argsort(distances)[1:k_neighbors+1]

        if len(k_nearest) == 0:
            continue

        # Generate synthetic sample
        neighbor_idx = np.random.choice(k_nearest)
        neighbor = minority_samples[neighbor_idx]
        alpha = np.random.random()
        synthetic = sample + alpha * (neighbor - sample)
        synthetic_samples.append(synthetic)

    return np.array(synthetic_samples)


def balance_dataset(X, y, method='smote', max_samples_per_class=10000):
    '''Handle class imbalance with memory limit'''
    print("\n" + "="*80)
    print("HANDLING CLASS IMBALANCE")
    print("="*80)

    class_counts = Counter(y)
    print(f"\n📊 Original distribution: {dict(class_counts)}")

    if method == 'smote':
        max_count = max(class_counts.values())
        target_count = min(int(max_count * 0.8), max_samples_per_class)

        X_balanced = X.copy()
        y_balanced = y.copy()

        for class_label, count in class_counts.items():
            if count < target_count:
                samples_needed = target_count - count
                print(f"  Generating {samples_needed} samples for class {class_label}...")
                synthetic = manual_smote(X, y, class_label, samples_to_generate=samples_needed)
                X_balanced = np.vstack([X_balanced, synthetic])
                y_balanced = np.concatenate([y_balanced, [class_label] * samples_needed])

        print(f"\n✓ Balanced distribution: {dict(Counter(y_balanced))}")
        return X_balanced, y_balanced

    return X, y


# ============================================================================
# 4. MODEL TRAINING WITH OVERFITTING/UNDERFITTING DETECTION
# ============================================================================

def train_and_evaluate_model(X_train, X_test, y_train, y_test, model, model_name):
    '''Train model with comprehensive evaluation'''
    print(f"\n{'='*60}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*60}")

    try:
        # Train model
        model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Training scores
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_test_pred, average='weighted', zero_division=0)

        # Detect overfitting/underfitting
        acc_diff = train_acc - test_acc
        if acc_diff > 0.1:
            status = "⚠️ OVERFITTING DETECTED"
        elif test_acc < 0.7:
            status = "⚠️ UNDERFITTING DETECTED"
        else:
            status = "✓ GOOD FIT"

        print(f"\n📊 Training Accuracy: {train_acc:.4f}")
        print(f"📊 Testing Accuracy:  {test_acc:.4f}")
        print(f"📊 Accuracy Difference: {acc_diff:.4f}")
        print(f"📊 Model Status: {status}")
        print(f"\n📊 Weighted Precision: {precision:.4f}")
        print(f"📊 Weighted Recall: {recall:.4f}")
        print(f"📊 Weighted F1-Score: {f1:.4f}")

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        print("\n🔄 Running 5-fold cross-validation...")
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
            try:
                model_cv = type(model)(**model.get_params())
                model_cv.fit(X_train[train_idx], y_train[train_idx])
                score = model_cv.score(X_train[val_idx], y_train[val_idx])
                cv_scores.append(score)
                print(f"  Fold {fold}: {score:.4f}")
            except Exception as e:
                print(f"  Fold {fold}: Error - {str(e)}")

        if cv_scores:
            print(f"\n✓ CV Mean ± Std: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        else:
            print("\n⚠️ Cross-validation failed")
            cv_scores = [test_acc]

        # Classification report
        print(f"\n📋 Detailed Classification Report:")
        print(classification_report(y_test, y_test_pred, zero_division=0))

        results = {
            'model_name': model_name,
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'accuracy_difference': float(acc_diff),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'cv_mean': float(np.mean(cv_scores)),
            'cv_std': float(np.std(cv_scores)),
            'status': status,
            'confusion_matrix': confusion_matrix(y_test, y_test_pred).tolist()
        }

        return model, results, y_test_pred
    
    except Exception as e:
        print(f"\n❌ Error training {model_name}: {str(e)}")
        raise


# ============================================================================
# 5. MAIN EXECUTION PIPELINE
# ============================================================================

def main():
    '''Main execution pipeline'''
    
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    print("\n" + "="*80)
    print("WSN-DS INTRUSION DETECTION SYSTEM")
    print("="*80)

    # Load dataset
    df, target_col = load_wsn_dataset('/root/amlan/Iot/project/data/raw/WSN-DS.csv')

    # Separate features and target
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).values
    y = df[target_col].values
    feature_names = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).columns.tolist()

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"\n🏷️ Class mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # PLOT 1: Class Distribution
    plot_class_distribution(y, y_encoded, le)

    # Feature engineering
    X_engineered, engineered_feature_names = engineer_features(X, feature_names)

    # PLOT 2: Correlation Matrix
    plot_correlation_matrix(X_engineered[:1000], engineered_feature_names)  # Sample for speed

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_engineered, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"\n📊 Train set: {X_train.shape[0]} samples")
    print(f"📊 Test set: {X_test.shape[0]} samples")

    # Handle class imbalance
    X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train, method='smote')

    # Scale features
    print("\n🔧 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    print("✓ Feature scaling complete")

    # Define models
    models = {
        'Random Forest (Optimized)': RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_split=5,
            min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=150, max_depth=25, class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=100, learning_rate=0.5, random_state=42
        ),
        'MLP Neural Network': MLPClassifier(
            hidden_layer_sizes=(100, 50), max_iter=500, alpha=0.001,
            learning_rate='adaptive', early_stopping=True, random_state=42
        )
    }
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models['XGBoost (Fast)'] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            n_jobs=-1,
            random_state=42,
            eval_metric='mlogloss'
        )

    # Train all models
    all_results = {}
    trained_models = {}

    for name, model in models.items():
        try:
            trained_model, results, predictions = train_and_evaluate_model(
                X_train_scaled, X_test_scaled, y_train_balanced, y_test, model, name
            )
            all_results[name] = results
            trained_models[name] = trained_model
            
            # PLOT 3: Confusion Matrix for each model
            cm = np.array(results['confusion_matrix'])
            plot_confusion_matrix(cm, le.classes_, name, 
                                f'plots/confusion_matrix_{name.replace(" ", "_")}.png')
            
            # PLOT 4: Learning Curves (for selected models)
            if name in ['Random Forest (Optimized)', 'XGBoost (Fast)', 'MLP Neural Network']:
                print(f"\n🔄 Generating learning curve for {name}...")
                plot_learning_curves(trained_model, X_train_scaled[:2000], y_train_balanced[:2000], 
                                   name, f'plots/learning_curve_{name.replace(" ", "_")}.png')
                
        except Exception as e:
            print(f"❌ Error training {name}: {str(e)}")
            continue

    # Hybrid Ensemble Model
    if len(trained_models) >= 3:
        print(f"\n{'='*80}")
        print("BUILDING HYBRID ENSEMBLE MODEL")
        print(f"{'='*80}")

        ensemble_models = []
        if 'Random Forest (Optimized)' in trained_models:
            ensemble_models.append(('rf', trained_models['Random Forest (Optimized)']))
        if 'Gradient Boosting' in trained_models:
            ensemble_models.append(('gb', trained_models['Gradient Boosting']))
        if 'Extra Trees' in trained_models:
            ensemble_models.append(('et', trained_models['Extra Trees']))
        
        if len(ensemble_models) >= 2:
            ensemble = VotingClassifier(
                estimators=ensemble_models,
                voting='soft',
                n_jobs=-1
            )

            try:
                ensemble_model, ensemble_results, ensemble_pred = train_and_evaluate_model(
                    X_train_scaled, X_test_scaled, y_train_balanced, y_test, ensemble, 'Hybrid Ensemble'
                )
                all_results['Hybrid Ensemble'] = ensemble_results
                trained_models['Hybrid Ensemble'] = ensemble_model
                
                # Confusion matrix for ensemble
                cm = np.array(ensemble_results['confusion_matrix'])
                plot_confusion_matrix(cm, le.classes_, 'Hybrid Ensemble', 
                                    'plots/confusion_matrix_Hybrid_Ensemble.png')
                
            except Exception as e:
                print(f"❌ Error training ensemble: {str(e)}")

    # Save results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")

    # Save to JSON
    with open('wsn_model_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)
    print("\n✓ Results saved to: wsn_model_results.json")

    # Save to CSV
    results_df = pd.DataFrame([
        {
            'Model': name,
            'Train_Accuracy': res['train_accuracy'],
            'Test_Accuracy': res['test_accuracy'],
            'Precision': res['precision'],
            'Recall': res['recall'],
            'F1_Score': res['f1_score'],
            'CV_Mean': res['cv_mean'],
            'CV_Std': res['cv_std'],
            'Status': res['status']
        }
        for name, res in all_results.items()
    ])
    results_df = results_df.sort_values('Test_Accuracy', ascending=False)
    results_df.to_csv('wsn_model_comparison.csv', index=False)
    print("✓ Comparison saved to: wsn_model_comparison.csv")

    # Feature importance
    if 'Random Forest (Optimized)' in trained_models:
        rf_model = trained_models['Random Forest (Optimized)']
        feature_importance = pd.DataFrame({
            'Feature': engineered_feature_names,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        feature_importance.to_csv('wsn_feature_importance.csv', index=False)
        print("✓ Feature importance saved to: wsn_feature_importance.csv")
        
        # PLOT 5: Feature Importance
        plot_feature_importance(feature_importance)

    # PLOT 6: Model Comparison
    plot_model_comparison(results_df)

    # PLOT 7: Cross-Validation Scores
    plot_cv_scores(all_results)

    # PLOT 8: Training vs Test Accuracy
    plot_training_vs_test_accuracy(all_results)

    # PLOT 9: ROC Curves
    plot_roc_curves(trained_models, X_test_scaled, y_test, le)

    # PLOT 10: Precision-Recall Curves
    plot_precision_recall_curves(trained_models, X_test_scaled, y_test, le)

    print(f"\n{'='*80}")
    print("✅ EXECUTION COMPLETED SUCCESSFULLY")
    print(f"{'='*80}")
    print(f"\n🏆 Top 3 Models by Test Accuracy:")
    print(results_df.head(3).to_string(index=False))
    
    print(f"\n📊 All plots saved in 'plots/' directory:")
    print("  • class_distribution.png")
    print("  • correlation_matrix.png")
    print("  • confusion_matrix_*.png (for each model)")
    print("  • learning_curve_*.png (for selected models)")
    print("  • feature_importance.png")
    print("  • model_comparison.png")
    print("  • cv_scores.png")
    print("  • train_vs_test.png")
    print("  • roc_curves.png")
    print("  • precision_recall.png")

    return all_results, results_df


# Execute
if __name__ == '__main__':
    results, comparison = main()
