# ============================================================================
# RESEARCH DIRECTION 4: EXPLAINABLE AI FOR INTRUSION DETECTION
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier
import json

def calculate_feature_importance(model, X, feature_names):
    """Calculate multiple types of feature importance"""
    importances = {}
    
    # Built-in feature importance (for tree-based models)
    if hasattr(model, 'feature_importances_'):
        importances['gini_importance'] = dict(zip(feature_names, model.feature_importances_))
    
    return importances

def permutation_feature_importance(model, X, y, feature_names, n_repeats=10):
    """Calculate permutation importance"""
    perm_importance = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1
    )
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    return importance_df

def analyze_decision_paths(X_test, y_test, model, feature_names, n_samples=5):
    """Analyze decision paths for sample predictions"""
    if not isinstance(model, RandomForestClassifier):
        return None
    
    explanations = []
    
    # Get one tree for analysis
    tree = model.estimators_[0]
    
    for i in range(min(n_samples, len(X_test))):
        sample = X_test[i:i+1]
        prediction = model.predict(sample)[0]
        actual = y_test[i]
        
        # Get decision path
        node_indicator = tree.decision_path(sample)
        leaf_id = tree.apply(sample)
        
        feature_idx = tree.tree_.feature
        threshold = tree.tree_.threshold
        
        # Extract path
        node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
        
        path_description = []
        for node_id in node_index:
            if tree.tree_.feature[node_id] != -2:  # -2 indicates leaf node
                feat_idx = feature_idx[node_id]
                feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"Feature_{feat_idx}"
                thres = threshold[node_id]
                value = sample[0, feat_idx]
                
                if value <= thres:
                    path_description.append(f"{feat_name} <= {thres:.2f} (value: {value:.2f})")
                else:
                    path_description.append(f"{feat_name} > {thres:.2f} (value: {value:.2f})")
        
        explanations.append({
            'sample_id': i,
            'prediction': int(prediction),
            'actual': int(actual),
            'correct': prediction == actual,
            'decision_path': path_description
        })
    
    return explanations

def feature_correlation_analysis(X, feature_names):
    """Analyze feature correlations"""
    df = pd.DataFrame(X, columns=feature_names)
    corr_matrix = df.corr()
    
    # Find highly correlated feature pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': float(corr_val)
                })
    
    return high_corr_pairs, corr_matrix

def main_explainable_ai():
    print("="*80)
    print("EXPLAINABLE AI FOR WSN INTRUSION DETECTION")
    print("="*80)
    
    # Load data
    df = pd.read_csv('/root/amlan/Iot/project/data/raw/WSN-DS.csv')
    
    target_col = None
    for col in ['Attack_Type', 'attack_type', 'Class', 'class']:
        if col in df.columns:
            target_col = col
            break
    if target_col is None:
        target_col = df.columns[-1]
    
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    feature_names = df.drop(columns=[target_col]).columns.tolist()
    
    # Encode
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("\nTraining Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    test_accuracy = model.score(X_test_scaled, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Feature importance analysis
    print("\n" + "-"*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("-"*80)
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance_df.head(10).to_string(index=False))
    
    # Permutation importance
    print("\nCalculating permutation importance...")
    perm_importance_df = permutation_feature_importance(
        model, X_test_scaled, y_test, feature_names, n_repeats=5
    )
    
    print("\nTop 10 Features by Permutation Importance:")
    print(perm_importance_df.head(10).to_string(index=False))
    
    # Feature correlation
    print("\n" + "-"*80)
    print("FEATURE CORRELATION ANALYSIS")
    print("-"*80)
    
    high_corr, corr_matrix = feature_correlation_analysis(X_train, feature_names)
    
    if high_corr:
        print(f"\nFound {len(high_corr)} highly correlated feature pairs (|r| > 0.8):")
        for pair in high_corr[:10]:
            print(f"  {pair['feature1']} <-> {pair['feature2']}: r={pair['correlation']:.3f}")
    else:
        print("\nNo highly correlated features found (|r| > 0.8)")
    
    # Decision path analysis
    print("\n" + "-"*80)
    print("DECISION PATH ANALYSIS (Sample Predictions)")
    print("-"*80)
    
    explanations = analyze_decision_paths(
        X_test_scaled, y_test, model, feature_names, n_samples=3
    )
    
    if explanations:
        for exp in explanations:
            print(f"\nSample {exp['sample_id']}:")
            print(f"  Predicted: {le.classes_[exp['prediction']]}")
            print(f"  Actual: {le.classes_[exp['actual']]}")
            print(f"  Correct: {exp['correct']}")
            print(f"  Decision Path (first 5 steps):")
            for step in exp['decision_path'][:5]:
                print(f"    - {step}")
    
    # Save all results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    feature_importance_df.to_csv('wsn_feature_importance_detailed.csv', index=False)
    perm_importance_df.to_csv('wsn_permutation_importance.csv', index=False)
    corr_matrix.to_csv('wsn_feature_correlation_matrix.csv')
    
    # Save explanations
    xai_results = {
        'feature_importance': feature_importance_df.to_dict('records'),
        'permutation_importance': perm_importance_df.to_dict('records'),
        'high_correlations': high_corr,
        'sample_explanations': explanations,
        'model_accuracy': float(test_accuracy)
    }
    
    with open('wsn_explainable_ai_results.json', 'w') as f:
        json.dump(xai_results, f, indent=4)
    
    print("\n✓ Feature importance saved to: wsn_feature_importance_detailed.csv")
    print("✓ Permutation importance saved to: wsn_permutation_importance.csv")
    print("✓ Correlation matrix saved to: wsn_feature_correlation_matrix.csv")
    print("✓ XAI results saved to: wsn_explainable_ai_results.json")
    
    return xai_results

if __name__ == '__main__':
    results = main_explainable_ai()
