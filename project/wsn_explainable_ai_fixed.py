# ============================================================================
# RESEARCH DIRECTION 4: EXPLAINABLE AI FOR INTRUSION DETECTION (FIXED)
# NO GRAPHS - Only CSV/JSON outputs
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier
import json

def convert_to_serializable(obj):
    '''Convert numpy types to Python native types for JSON serialization'''
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

def calculate_feature_importance(model, X, feature_names):
    '''Calculate multiple types of feature importance'''
    importances = {}

    # Built-in feature importance (for tree-based models)
    if hasattr(model, 'feature_importances_'):
        importances['gini_importance'] = dict(zip(feature_names, model.feature_importances_))

    return importances

def permutation_feature_importance(model, X, y, feature_names, n_repeats=10):
    '''Calculate permutation importance'''
    print("\nCalculating permutation importance (this may take a few minutes)...")
    perm_importance = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1
    )

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)

    return importance_df

def analyze_decision_paths(X_test, y_test, model, feature_names, n_samples=10):
    '''Analyze decision paths for sample predictions'''
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
            'sample_id': int(i),
            'prediction': int(prediction),
            'actual': int(actual),
            'correct': bool(prediction == actual),
            'decision_path': path_description
        })

    return explanations

def feature_correlation_analysis(X, feature_names):
    '''Analyze feature correlations'''
    df = pd.DataFrame(X, columns=feature_names)
    corr_matrix = df.corr()

    # Find highly correlated feature pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:
                high_corr_pairs.append({
                    'feature1': str(corr_matrix.columns[i]),
                    'feature2': str(corr_matrix.columns[j]),
                    'correlation': float(corr_val)
                })

    return high_corr_pairs, corr_matrix

def analyze_feature_statistics(X, y, feature_names, class_names):
    '''Analyze feature statistics per class'''
    df = pd.DataFrame(X, columns=feature_names)
    df['class'] = y

    stats_per_class = []

    for class_label in np.unique(y):
        class_data = df[df['class'] == class_label]
        class_name = str(class_names[class_label])

        for feature in feature_names:
            stats_per_class.append({
                'class': class_name,
                'feature': str(feature),
                'mean': float(class_data[feature].mean()),
                'std': float(class_data[feature].std()),
                'min': float(class_data[feature].min()),
                'max': float(class_data[feature].max()),
                'median': float(class_data[feature].median())
            })

    return pd.DataFrame(stats_per_class)

def main_explainable_ai():
    print("="*80)
    print("EXPLAINABLE AI FOR WSN INTRUSION DETECTION")
    print("="*80)

    # Load data
    df = pd.read_csv('/root/amlan/Iot/project/data/raw/WSN-DS.csv')

    target_col = None
    for col in ['Attack_Type', 'attack_type', 'Class', 'class', 'Label']:
        if col in df.columns:
            target_col = col
            break
    if target_col is None:
        target_col = df.columns[-1]

    print(f"\nTarget column: {target_col}")

    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    feature_names = df.drop(columns=[target_col]).columns.tolist()

    print(f"Features: {len(feature_names)}")
    print(f"Samples: {len(X)}")

    # Encode
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"\nClasses: {le.classes_}")

    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    print("\n" + "-"*80)
    print("TRAINING RANDOM FOREST MODEL")
    print("-"*80)

    model = RandomForestClassifier(
        n_estimators=100, max_depth=20, random_state=42, n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)

    print(f"\nTrain Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

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
        for i, pair in enumerate(high_corr[:10]):
            print(f"  {i+1}. {pair['feature1']} <-> {pair['feature2']}: r={pair['correlation']:.3f}")
    else:
        print("\nNo highly correlated features found (|r| > 0.8)")

    # Feature statistics per class
    print("\n" + "-"*80)
    print("FEATURE STATISTICS PER CLASS")
    print("-"*80)

    feature_stats_df = analyze_feature_statistics(X_train, y_train, feature_names, le.classes_)

    print("\nSample statistics (first 5 features, first 2 classes):")
    sample_stats = feature_stats_df[
        (feature_stats_df['feature'].isin(feature_names[:5])) & 
        (feature_stats_df['class'].isin([str(c) for c in le.classes_[:2]]))
    ]
    print(sample_stats.to_string(index=False))

    # Decision path analysis
    print("\n" + "-"*80)
    print("DECISION PATH ANALYSIS")
    print("-"*80)

    explanations = analyze_decision_paths(
        X_test_scaled, y_test, model, feature_names, n_samples=5
    )

    if explanations:
        for exp in explanations[:3]:
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
    feature_stats_df.to_csv('wsn_feature_statistics_per_class.csv', index=False)

    # Save high correlations
    if high_corr:
        high_corr_df = pd.DataFrame(high_corr)
        high_corr_df.to_csv('wsn_high_correlations.csv', index=False)

    # Convert to JSON-serializable format
    xai_results = {
        'model_accuracy': {
            'train': float(train_acc),
            'test': float(test_acc)
        },
        'top_10_features': convert_to_serializable(feature_importance_df.head(10).to_dict('records')),
        'top_10_permutation': convert_to_serializable(perm_importance_df.head(10).to_dict('records')),
        'high_correlations': convert_to_serializable(high_corr),
        'sample_explanations': convert_to_serializable(explanations)
    }

    with open('wsn_explainable_ai_results.json', 'w') as f:
        json.dump(xai_results, f, indent=4)

    print("\n✓ Feature importance saved to: wsn_feature_importance_detailed.csv")
    print("✓ Permutation importance saved to: wsn_permutation_importance.csv")
    print("✓ Correlation matrix saved to: wsn_feature_correlation_matrix.csv")
    print("✓ Feature statistics saved to: wsn_feature_statistics_per_class.csv")
    if high_corr:
        print("✓ High correlations saved to: wsn_high_correlations.csv")
    print("✓ XAI results saved to: wsn_explainable_ai_results.json")

    print("\n" + "="*80)
    print("EXPLAINABLE AI ANALYSIS COMPLETED")
    print("="*80)

    return xai_results

if __name__ == '__main__':
    results = main_explainable_ai()
