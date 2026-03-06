
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
import json
import time

def create_lightweight_neural_network():
    '''Create optimized lightweight neural network'''
    print("\n" + "="*80)
    print("LIGHTWEIGHT NEURAL NETWORK FOR WSN RESOURCE CONSTRAINTS")
    print("="*80)

    models = {
        'Tiny-NN (1 layer)': MLPClassifier(
            hidden_layer_sizes=(32,),
            activation='relu',
            solver='adam',
            alpha=0.01,
            batch_size=128,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        ),
        'Small-NN (2 layers)': MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=128,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        ),
        'Medium-NN (3 layers)': MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=64,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
    }

    return models

def evaluate_with_time_memory(model, X_train, X_test, y_train, y_test, model_name):
    '''Evaluate model with time and memory efficiency metrics'''

    # Training time
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train

    # Inference time
    start_inference = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_inference
    avg_inference_per_sample = inference_time / len(X_test)

    # Accuracy
    train_acc = model.score(X_train, y_train)
    test_acc = accuracy_score(y_test, y_pred)

    # Model complexity
    total_params = sum(layer.size for layer in model.coefs_) + sum(layer.size for layer in model.intercepts_)

    results = {
        'model_name': model_name,
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'train_time_seconds': float(train_time),
        'total_inference_time_seconds': float(inference_time),
        'inference_per_sample_ms': float(avg_inference_per_sample * 1000),
        'total_parameters': int(total_params),
        'hidden_layers': str(model.hidden_layer_sizes),
        'n_iterations': int(model.n_iter_),
        'loss': float(model.loss_)
    }

    print(f"\n{model_name}:")
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  Training Time: {train_time:.2f}s")
    print(f"  Inference Time: {avg_inference_per_sample*1000:.4f}ms per sample")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Iterations: {model.n_iter_}")

    return results, y_pred

def main_lightweight():
    # Load data
    df = pd.read_csv('/root/amlan/Iot/project/data/raw/WSN-DS.csv')
    target_col = [col for col in df.columns if 'attack' in col.lower() or 'class' in col.lower()]
    if target_col:
        target_col = target_col[0]
    else:
        target_col = df.columns[-1]

    X = df.drop(columns=[target_col]).values
    y = df[target_col].values

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

    # Create and evaluate models
    models = create_lightweight_neural_network()
    all_results = []

    print("\nTraining lightweight models...")
    for name, model in models.items():
        results, predictions = evaluate_with_time_memory(
            model, X_train_scaled, X_test_scaled, y_train, y_test, name
        )
        all_results.append(results)

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df['efficiency_score'] = (
        results_df['test_accuracy'] / 
        (results_df['inference_per_sample_ms'] * results_df['total_parameters'] / 1000000)
    )
    results_df = results_df.sort_values('efficiency_score', ascending=False)

    results_df.to_csv('wsn_lightweight_nn_results.csv', index=False)

    with open('wsn_lightweight_nn_results.json', 'w') as f:
        json.dump(all_results, f, indent=4)

    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(results_df.to_string(index=False))
    print("\n✓ Results saved to: wsn_lightweight_nn_results.csv")
    print("✓ Results saved to: wsn_lightweight_nn_results.json")

if __name__ == '__main__':
    main_lightweight()
