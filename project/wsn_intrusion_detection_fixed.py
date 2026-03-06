#!/usr/bin/env python3
"""
WSN Intrusion Detection System
Configured for local execution
"""

import os
import zipfile
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
import time
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# DATASET DOWNLOAD AND LOADING
# ============================================================

# Download using Kaggle API (requires: pip install kaggle)
if not os.path.exists('WSN-DS.csv'):
    print("Downloading dataset from Kaggle...")
    os.system('kaggle datasets download -d bassamkasasbeh1/wsnds')

    # Extract the downloaded zip file
    with zipfile.ZipFile('wsnds.zip', 'r') as zip_ref:
        zip_ref.extractall('.')

    # Clean up
    os.remove('wsnds.zip')
    print("Dataset downloaded and extracted!")

path = 'WSN-DS.csv'

if not os.path.exists(path):
    raise FileNotFoundError(f"Dataset not found: {path}\nPlease place WSN-DS.csv in the same directory or update the path variable")

print(f"Loading dataset from: {path}")
df = pd.read_csv(path)

# ============================================================
# DATA PREPROCESSING
# ============================================================

# Professional Header Cleaning
df.columns = df.columns.str.strip().str.replace(' ', '_')

# Feature Pruning - drop simulation artifacts
df = df.drop(columns=['id', 'Time'])

print(f"Dataset Cleaned. Final Shape: {df.shape}")
print(f"Columns ready for analysis: {df.columns.tolist()}")

# ============================================================
# TARGET ENCODING
# ============================================================

# Manual Mapping to ensure 'Normal' is the baseline (0)
mapping = {
    'Normal': 0,
    'Blackhole': 1,
    'Grayhole': 2,
    'Flooding': 3,
    'TDMA': 4
}

df['Attack_type_encoded'] = df['Attack_type'].map(mapping)

print("\nTarget Encoding Results:")
print(df[['Attack_type', 'Attack_type_encoded']].drop_duplicates())

# ============================================================
# TRAIN-TEST SPLIT
# ============================================================

# Define Features (X) and Target (y)
X = df.drop(columns=['Attack_type', 'Attack_type_encoded'])
y = df['Attack_type_encoded']

# Stratified Split (80% Train, 20% Test)
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

print(f"\nTraining Set Size: {len(X_train_raw)}")
print(f"Hold-out Test Set Size: {len(X_test_raw)}")

# ============================================================
# HANDLING CLASS IMBALANCE
# ============================================================

# Hybrid Strategy: RandomUnderSampler + SMOTE
rus = RandomUnderSampler(sampling_strategy={0: 50000}, random_state=42)
smote = SMOTE(random_state=42)
sampling_pipe = Pipeline(steps=[('u', rus), ('o', smote)])

# Execute Sampling on Training Data ONLY
X_train_final, y_train_final = sampling_pipe.fit_resample(X_train_raw, y_train_raw)

print(f"\nFinal Balanced Training Distribution: {Counter(y_train_final)}")

# ============================================================
# FEATURE SCALING
# ============================================================

scaler = StandardScaler()

# Fit on training data and transform both sets
X_train_scaled = scaler.fit_transform(X_train_final)
X_test_scaled = scaler.transform(X_test_raw)

print(f"\nTraining data scaled: {X_train_scaled.shape}")
print(f"Test data scaled: {X_test_scaled.shape}")
print(f"Feature means after scaling (should be ~0): {X_train_scaled.mean(axis=0)[:5]}")
print(f"Feature stds after scaling (should be ~1): {X_train_scaled.std(axis=0)[:5]}")

# ============================================================
# MODEL TRAINING
# ============================================================

print("\n" + "="*60)
print("Training Random Forest Classifier...")
print("="*60)
start_time = time.time()

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_model.fit(X_train_scaled, y_train_final)
training_time = time.time() - start_time

print(f"\nTraining completed in {training_time:.2f} seconds")

# ============================================================
# MODEL EVALUATION
# ============================================================

# Make predictions
y_pred = rf_model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test_raw, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Detailed Classification Report
attack_names = ['Normal', 'Blackhole', 'Grayhole', 'Flooding', 'TDMA']
print("\n" + "="*60)
print("Classification Report:")
print("="*60)
print(classification_report(y_test_raw, y_pred, target_names=attack_names, digits=4))

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test_raw, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=attack_names, yticklabels=attack_names)
plt.title('Confusion Matrix - WSN Intrusion Detection')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\nConfusion matrix saved as 'confusion_matrix.png'")
plt.show()

print("\nConfusion Matrix:")
print(cm)

# ============================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "="*60)
print("Top 10 Most Important Features:")
print("="*60)
print(feature_importance.head(10))

# Visualize top features
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'].head(10), 
         feature_importance['importance'].head(10))
plt.xlabel('Importance Score')
plt.title('Top 10 Feature Importances')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("\nFeature importance plot saved as 'feature_importance.png'")
plt.show()

# ============================================================
# PER-CLASS PERFORMANCE METRICS
# ============================================================

precision, recall, f1, support = precision_recall_fscore_support(
    y_test_raw, y_pred, average=None
)

performance_df = pd.DataFrame({
    'Attack_Type': attack_names,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Support': support
})

# Calculate per-class accuracy from confusion matrix
per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
performance_df['Accuracy'] = per_class_accuracy

print("\n" + "="*60)
print("Per-Class Performance Metrics:")
print("="*60)
print(performance_df.to_string(index=False))

print("\n" + "="*60)
print(f"Overall Accuracy: {accuracy:.4f}")
print(f"Macro Average F1-Score: {f1.mean():.4f}")
print("="*60)

# Save results to CSV
performance_df.to_csv('performance_metrics.csv', index=False)
print("\nPerformance metrics saved as 'performance_metrics.csv'")

print("\n✓ Analysis Complete!")
