# ============================================================================
# RESEARCH DIRECTION 3: ADVANCED CLASS IMBALANCE HANDLING
# Focuses on improving Grayhole detection (currently 75.6% accuracy)
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import json

class CostSensitiveLearning:
    """Implement cost-sensitive learning for imbalanced classes"""
    
    def __init__(self, base_model, cost_matrix=None):
        self.base_model = base_model
        self.cost_matrix = cost_matrix
    
    def fit(self, X, y):
        # Calculate class weights with emphasis on minority classes
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        
        # Extra weight for Grayhole (typically class with lowest accuracy)
        weight_dict = dict(zip(classes, class_weights))
        
        # Identify Grayhole class and increase its weight
        class_counts = pd.Series(y).value_counts()
        grayhole_class = class_counts.idxmin()  # Assume smallest class is Grayhole
        weight_dict[grayhole_class] *= 2.0  # Double the weight
        
        self.base_model.set_params(class_weight=weight_dict)
        self.base_model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.base_model.predict(X)
    
    def predict_proba(self, X):
        return self.base_model.predict_proba(X)

def synthetic_minority_oversampling(X, y, target_class, n_samples):
    """Enhanced SMOTE with boundary samples focus"""
    minority_indices = np.where(y == target_class)[0]
    X_minority = X[minority_indices]
    
    synthetic_samples = []
    
    for i in range(n_samples):
        # Random sample
        idx = np.random.randint(0, len(X_minority))
        sample = X_minority[idx]
        
        # Find k=5 nearest neighbors
        distances = np.sqrt(np.sum((X_minority - sample) ** 2, axis=1))
        k_nearest_idx = np.argsort(distances)[1:6]
        
        # Select random neighbor
        neighbor_idx = np.random.choice(k_nearest_idx)
        neighbor = X_minority[neighbor_idx]
        
        # Generate synthetic sample with random interpolation
        alpha = np.random.uniform(0.3, 0.7)  # Focus on middle range
        synthetic = sample + alpha * (neighbor - sample)
        
        # Add small random noise
        noise = np.random.normal(0, 0.01, synthetic.shape)
        synthetic += noise
        
        synthetic_samples.append(synthetic)
    
    return np.array(synthetic_samples)

def hybrid_sampling_strategy(X, y):
    """Combine oversampling minority and undersampling majority"""
    class_counts = pd.Series(y).value_counts()
    print(f"\nOriginal distribution:\n{class_counts}")
    
    # Target: balance all classes to 70% of majority
    max_count = class_counts.max()
    target_count = int(max_count * 0.7)
    
    X_resampled = []
    y_resampled = []
    
    for class_label in class_counts.index:
        class_indices = np.where(y == class_label)[0]
        class_samples = X[class_indices]
        current_count = len(class_samples)
        
        if current_count < target_count:
            # Oversample minority class
            n_synthetic = target_count - current_count
            synthetic = synthetic_minority_oversampling(X, y, class_label, n_synthetic)
            X_resampled.append(class_samples)
            X_resampled.append(synthetic)
            y_resampled.extend([class_label] * current_count)
            y_resampled.extend([class_label] * n_synthetic)
        elif current_count > target_count:
            # Undersample majority class
            sample_indices = np.random.choice(len(class_samples), target_count, replace=False)
            X_resampled.append(class_samples[sample_indices])
            y_resampled.extend([class_label] * target_count)
        else:
            X_resampled.append(class_samples)
            y_resampled.extend([class_label] * current_count)
    
    X_resampled = np.vstack(X_resampled)
    y_resampled = np.array(y_resampled)
    
    print(f"\nResampled distribution:\n{pd.Series(y_resampled).value_counts()}")
    
    return X_resampled, y_resampled

def focal_loss_weights(y_true, y_pred_proba, alpha=0.25, gamma=2.0):
    """Calculate focal loss to focus on hard examples"""
    # This returns sample weights that can be used in sklearn
    pt = y_pred_proba[np.arange(len(y_true)), y_true]
    focal_weight = alpha * (1 - pt) ** gamma
    return focal_weight

def main_imbalance_handling():
    # Load dataset
    df = pd.read_csv('/root/amlan/Iot/project/data/raw/WSN-DS.csv')
    
    # Find target column
    target_col = None
    for col in ['Attack_Type', 'attack_type', 'Class', 'class', 'Label']:
        if col in df.columns:
            target_col = col
            break
    if target_col is None:
        target_col = df.columns[-1]
    
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"Class mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Apply hybrid sampling
    X_train_balanced, y_train_balanced = hybrid_sampling_strategy(X_train, y_train)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple models with different strategies
    strategies = {
        'Baseline RF': RandomForestClassifier(n_estimators=200, random_state=42),
        'Class Weighted RF': RandomForestClassifier(
            n_estimators=200, class_weight='balanced', random_state=42
        ),
        'Cost-Sensitive RF': CostSensitiveLearning(
            RandomForestClassifier(n_estimators=200, random_state=42)
        )
    }
    
    results = {}
    
    for name, model in strategies.items():
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}")
        
        model.fit(X_train_scaled, y_train_balanced)
        y_pred = model.predict(X_test_scaled)
        
        # Overall metrics
        report = classification_report(y_test, y_pred, output_dict=True, target_names=le.classes_)
        
        print(f"\nOverall Accuracy: {report['accuracy']:.4f}")
        print(f"Macro F1-Score: {report['macro avg']['f1-score']:.4f}")
        print(f"Weighted F1-Score: {report['weighted avg']['f1-score']:.4f}")
        
        # Per-class metrics
        print("\nPer-Class Performance:")
        for class_name in le.classes_:
            if class_name in report:
                metrics = report[class_name]
                print(f"  {class_name:15s}: Precision={metrics['precision']:.4f}, "
                      f"Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")
        
        results[name] = {
            'accuracy': report['accuracy'],
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score'],
            'per_class': {cls: report[cls] for cls in le.classes_ if cls in report}
        }
    
    # Save results
    results_df = pd.DataFrame([
        {
            'Strategy': name,
            'Accuracy': res['accuracy'],
            'Macro_F1': res['macro_f1'],
            'Weighted_F1': res['weighted_f1']
        }
        for name, res in results.items()
    ])
    
    results_df.to_csv('wsn_imbalance_handling_results.csv', index=False)
    
    with open('wsn_imbalance_handling_results.json', 'w') as f:
        json.dump(results, f, indent=4, default=str)
    
    print("\n" + "="*60)
    print("RESULTS SAVED")
    print("="*60)
    print(results_df.to_string(index=False))
    print("\n✓ Results saved to: wsn_imbalance_handling_results.csv")
    print("✓ Results saved to: wsn_imbalance_handling_results.json")

if __name__ == '__main__':
    main_imbalance_handling()
