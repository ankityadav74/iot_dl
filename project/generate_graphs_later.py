
# ============================================================================
# GENERATE GRAPHS LATER - Run this when you need visualizations
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle

def load_saved_data():
    '''Load all saved data'''
    with open('graph_data_all_models.json', 'r') as f:
        results = json.load(f)

    with open('class_mapping.json', 'r') as f:
        class_mapping = json.load(f)

    return results, class_mapping

def plot_training_loss(results):
    '''Plot training loss curves'''
    plt.figure(figsize=(10, 6))

    for model_name, data in results.items():
        if data['training_loss']:
            plt.plot(data['training_loss'], label=model_name, linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Curves', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Training loss plot saved: training_loss_curves.png")

def plot_accuracy_comparison(results):
    '''Plot train vs test accuracy'''
    models = list(results.keys())
    train_accs = [results[m]['train_accuracy'] for m in models]
    test_accs = [results[m]['test_accuracy'] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, train_accs, width, label='Train Accuracy', alpha=0.8)
    ax.bar(x + width/2, test_accs, width, label='Test Accuracy', alpha=0.8)

    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Train vs Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Accuracy comparison plot saved: accuracy_comparison.png")

def plot_confusion_matrices(results, class_mapping):
    '''Plot confusion matrices for all models'''
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))

    if n_models == 1:
        axes = [axes]

    class_names = [class_mapping[str(i)] for i in sorted([int(k) for k in class_mapping.keys()])]

    for idx, (model_name, data) in enumerate(results.items()):
        cm = np.array(data['confusion_matrix'])

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=class_names, yticklabels=class_names)
        axes[idx].set_title(f'{model_name}\nConfusion Matrix', fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig('confusion_matrices_all.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Confusion matrices saved: confusion_matrices_all.png")

def plot_roc_curves(results, class_mapping):
    '''Plot ROC curves for each model'''
    n_models = len(results)

    for model_name, data in results.items():
        roc_data = data['roc_data']
        n_classes = len(roc_data)

        plt.figure(figsize=(10, 8))

        for class_key, class_data in roc_data.items():
            plt.plot(class_data['fpr'], class_data['tpr'], 
                    label=f"{class_data['class_name']} (AUC = {class_data['auc']:.3f})",
                    linewidth=2)

        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curves - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'roc_curves_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ ROC curves saved: roc_curves_{model_name}.png")

def plot_metrics_comparison(results):
    '''Plot precision, recall, F1 comparison'''
    models = list(results.keys())
    precision = [results[m]['precision'] for m in models]
    recall = [results[m]['recall'] for m in models]
    f1 = [results[m]['f1_score'] for m in models]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)

    ax.set_xlabel('Models', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Precision, Recall, F1-Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Metrics comparison saved: metrics_comparison.png")

def generate_all_graphs():
    '''Generate all graphs from saved data'''
    print("="*80)
    print("GENERATING ALL GRAPHS FROM SAVED DATA")
    print("="*80)

    # Load data
    results, class_mapping = load_saved_data()

    # Generate all plots
    plot_training_loss(results)
    plot_accuracy_comparison(results)
    plot_confusion_matrices(results, class_mapping)
    plot_roc_curves(results, class_mapping)
    plot_metrics_comparison(results)

    print("\n" + "="*80)
    print("ALL GRAPHS GENERATED SUCCESSFULLY")
    print("="*80)
    print("\nGenerated files:")
    print("  - training_loss_curves.png")
    print("  - accuracy_comparison.png")
    print("  - confusion_matrices_all.png")
    print("  - roc_curves_[model].png (one per model)")
    print("  - metrics_comparison.png")

if __name__ == '__main__':
    generate_all_graphs()
