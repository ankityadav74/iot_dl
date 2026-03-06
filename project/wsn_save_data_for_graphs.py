
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.neural_network import MLPClassifier

def save_all_visualization_data():
    '''Save all data needed for later graph generation'''

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
    feature_names = df.drop(columns=[target_col]).columns.tolist()

    # Encode
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models and save training history
    models_to_train = {
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42, verbose=0),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42, verbose=0),
        'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, verbose=False)
    }

    all_results = {}

    for model_name, model in models_to_train.items():
        print(f"\nTraining {model_name}...")

        # Train model
        model.fit(X_train_scaled, y_train)

        # Get predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        # Get probabilities (for ROC curves)
        y_train_proba = model.predict_proba(X_train_scaled)
        y_test_proba = model.predict_proba(X_test_scaled)

        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_test_pred, average='weighted'
        )

        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_test_pred)

        # ROC data for each class (one-vs-rest)
        n_classes = len(np.unique(y_encoded))
        roc_data = {}

        for i in range(n_classes):
            # Binary classification: class i vs rest
            y_test_binary = (y_test == i).astype(int)
            y_test_proba_class = y_test_proba[:, i]

            fpr, tpr, thresholds = roc_curve(y_test_binary, y_test_proba_class)
            auc_score = roc_auc_score(y_test_binary, y_test_proba_class)

            roc_data[f'class_{i}'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': thresholds.tolist(),
                'auc': float(auc_score),
                'class_name': le.classes_[i]
            }

        # For MLP, save training loss history
        training_loss = []
        if hasattr(model, 'loss_curve_'):
            training_loss = model.loss_curve_

        # Save everything
        all_results[model_name] = {
            'train_accuracy': float(train_acc),
            'test_accuracy': float(test_acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': conf_matrix.tolist(),
            'roc_data': roc_data,
            'training_loss': [float(x) for x in training_loss] if training_loss else None
        }

        # Save predictions and probabilities to CSV
        predictions_df = pd.DataFrame({
            'true_label': y_test,
            'predicted_label': y_test_pred,
            'correct': y_test == y_test_pred
        })

        # Add probability columns
        for i, class_name in enumerate(le.classes_):
            predictions_df[f'prob_{class_name}'] = y_test_proba[:, i]

        predictions_df.to_csv(f'predictions_{model_name}.csv', index=False)
        print(f"  ✓ Predictions saved to: predictions_{model_name}.csv")

        # Save trained model
        with open(f'trained_model_{model_name}.pkl', 'wb') as f:
            pickle.dump(model, f)
        print(f"  ✓ Model saved to: trained_model_{model_name}.pkl")

    # Save all metrics and data for graphs
    with open('graph_data_all_models.json', 'w') as f:
        json.dump(all_results, f, indent=4)

    # Save label encoder
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    # Save class names mapping
    class_mapping = {int(i): name for i, name in enumerate(le.classes_)}
    with open('class_mapping.json', 'w') as f:
        json.dump(class_mapping, f, indent=4)

    # Create summary CSV for accuracy tracking
    summary_df = pd.DataFrame([
        {
            'Model': name,
            'Train_Accuracy': res['train_accuracy'],
            'Test_Accuracy': res['test_accuracy'],
            'Precision': res['precision'],
            'Recall': res['recall'],
            'F1_Score': res['f1_score']
        }
        for name, res in all_results.items()
    ])
    summary_df.to_csv('model_metrics_summary.csv', index=False)

    print("\n" + "="*80)
    print("ALL DATA SAVED FOR FUTURE GRAPH GENERATION")
    print("="*80)
    print("\nFiles created:")
    print("  1. graph_data_all_models.json - ROC data, confusion matrices, training loss")
    print("  2. predictions_[model].csv - All predictions with probabilities")
    print("  3. trained_model_[model].pkl - Trained models")
    print("  4. label_encoder.pkl - Label encoder for class names")
    print("  5. class_mapping.json - Class ID to name mapping")
    print("  6. model_metrics_summary.csv - All metrics summary")
    print("\nYou can now generate ANY graphs later using these files!")

    return all_results

if __name__ == '__main__':
    results = save_all_visualization_data()
