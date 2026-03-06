"""
Visualization Script for Trained Temporal Graph Transformer
Generates publication-ready figures for research paper
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, precision_recall_curve,
    accuracy_score, precision_recall_fscore_support
)
from sklearn.preprocessing import label_binarize
from torch_geometric.loader import DataLoader as GeoDataLoader
import os
import json
import warnings
warnings.filterwarnings('ignore')

# Import model
from tgt_model import TemporalGraphTransformer, TGT_Lightweight
from graph_builder import WSNGraphBuilder

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

class TGTVisualizer:
    """Generate visualizations from trained TGT model"""
    
    def __init__(self, checkpoint_path='checkpoints/best_model.pt'):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.attack_names = ['Normal', 'Blackhole', 'Grayhole', 'Flooding', 'TDMA']
        
        # Load checkpoint with weights_only=False
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = checkpoint['config']
        self.metrics = checkpoint.get('metrics', {})
        
        # Build model
        print("Building model...")
        if self.config.get('model_type', 'full') == 'lightweight':
            self.model = TGT_Lightweight(
                num_node_features=self.config['num_features'],
                num_classes=self.config['num_classes']
            )
        else:
            self.model = TemporalGraphTransformer(
                num_node_features=self.config['num_features'],
                hidden_dim=self.config.get('hidden_dim', 128),
                num_gnn_layers=self.config.get('num_gnn_layers', 3),
                lstm_hidden=self.config.get('lstm_hidden', 256),
                num_heads=self.config.get('num_heads', 8),
                num_classes=self.config['num_classes'],
                dropout=self.config.get('dropout', 0.3)
            )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded from {checkpoint_path}")
        print(f"✓ Validation Accuracy: {self.metrics.get('accuracy', 'N/A'):.2f}%")
    
    def load_test_data(self):
        """Load and prepare test data"""
        print("\nLoading test data...")
        
        # Load dataset
        data_path = self.config.get('data_path', 'data/raw/WSN-DS.csv')
        df = pd.read_csv(data_path)
        
        # Preprocess
        df.columns = df.columns.str.strip().str.replace(' ', '_')
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
        if 'Time' in df.columns:
            df = df.drop(columns=['Time'])
        
        # Encode
        mapping = {'Normal': 0, 'Blackhole': 1, 'Grayhole': 2, 'Flooding': 3, 'TDMA': 4}
        df['Attack_type_encoded'] = df['Attack_type'].map(mapping)
        
        # Generate graphs
        graph_builder = WSNGraphBuilder(
            num_nodes=self.config.get('num_nodes', 50),
            connection_radius=self.config.get('connection_radius', 0.3)
        )
        
        graphs = graph_builder.create_temporal_snapshots(
            df,
            window_size=self.config.get('window_size', 10),
            stride=self.config.get('stride', 5)
        )
        
        # Use last 20% as test set
        from sklearn.model_selection import train_test_split
        _, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
        
        test_loader = GeoDataLoader(
            test_graphs,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        print(f"✓ Test set: {len(test_graphs)} graphs")
        return test_loader
    
    @torch.no_grad()
    def get_predictions(self, test_loader):
        """Get model predictions on test set"""
        print("\nGenerating predictions...")
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch in test_loader:
            batch = batch.to(self.device)
            logits, _ = self.model(batch.x, batch.edge_index, batch.batch)
            
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
        
        all_probs = np.vstack(all_probs)
        
        print(f"✓ Generated predictions for {len(all_labels)} samples")
        return np.array(all_labels), np.array(all_preds), all_probs
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path='figures/confusion_matrix.png'):
        """Plot confusion matrix"""
        print("\n[1/6] Generating confusion matrix...")
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.attack_names,
                    yticklabels=self.attack_names,
                    cbar_kws={'label': 'Count'})
        
        plt.title('Confusion Matrix - Temporal Graph Transformer', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        os.makedirs('figures', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"      ✓ Saved: {save_path}")
        plt.close()
    
    def plot_per_class_metrics(self, y_true, y_pred, save_path='figures/per_class_metrics.png'):
        """Plot per-class precision, recall, F1-score"""
        print("[2/6] Generating per-class metrics...")
        
        # Specify all labels explicitly
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1, 2, 3, 4], average=None, zero_division=0
        )
        
        # Filter out classes with no samples
        present_classes = []
        present_precision = []
        present_recall = []
        present_f1 = []
        
        for i in range(len(self.attack_names)):
            if support[i] > 0:
                present_classes.append(self.attack_names[i])
                present_precision.append(precision[i])
                present_recall.append(recall[i])
                present_f1.append(f1[i])
        
        x = np.arange(len(present_classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width, present_precision, width, label='Precision', alpha=0.8)
        bars2 = ax.bar(x, present_recall, width, label='Recall', alpha=0.8)
        bars3 = ax.bar(x + width, present_f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Attack Type', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(present_classes, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"      ✓ Saved: {save_path}")
        plt.close()
    
    def plot_roc_curves(self, y_true, y_probs, save_path='figures/roc_curves.png'):
        """Plot ROC curves"""
        print("[3/6] Generating ROC curves...")
        
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4])
        n_classes = 5
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.Set2(np.linspace(0, 1, n_classes))
        
        for i, color in enumerate(colors):
            if y_true_bin[:, i].sum() > 0:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
                roc_auc = auc(fpr, tpr)
                
                ax.plot(fpr, tpr, color=color, lw=2,
                       label=f'{self.attack_names[i]} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - Multi-class Classification', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"      ✓ Saved: {save_path}")
        plt.close()
    
    def plot_precision_recall_curves(self, y_true, y_probs, save_path='figures/precision_recall_curves.png'):
        """Plot Precision-Recall curves"""
        print("[4/6] Generating Precision-Recall curves...")
        
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4])
        n_classes = 5
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.Set2(np.linspace(0, 1, n_classes))
        
        for i, color in enumerate(colors):
            if y_true_bin[:, i].sum() > 0:
                precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_probs[:, i])
                ax.plot(recall, precision, color=color, lw=2, label=f'{self.attack_names[i]}')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
        ax.legend(loc="lower left")
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"      ✓ Saved: {save_path}")
        plt.close()
    
    def plot_accuracy_comparison(self, y_true, y_pred, save_path='figures/accuracy_comparison.png'):
        """Plot accuracy comparison with real baseline results"""
        print("[5/6] Generating accuracy comparison...")
        
        # Load baseline results
        if os.path.exists('results/baseline_comparison.json'):
            with open('results/baseline_comparison.json', 'r') as f:
                baseline_results = json.load(f)
            
            # Extract accuracies in order
            methods = []
            accuracies = []
            
            method_order = ['Random Forest', 'Pure LSTM', 'Pure GNN', 'Pure Transformer', 'TGT (Ours)']
            
            for method in method_order:
                if method in baseline_results:
                    methods.append(method.replace(' ', '\n'))
                    accuracies.append(baseline_results[method]['accuracy'])
            
            print(f"      ✓ Loaded {len(methods)} baseline results")
        else:
            print("      ⚠ Baseline results not found, showing TGT only")
            our_accuracy = accuracy_score(y_true, y_pred) * 100
            methods = ['TGT\n(Ours)']
            accuracies = [our_accuracy]
        
        # Color scheme
        colors = ['#95a5a6'] * (len(methods) - 1) + ['#e74c3c']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Model Comparison - Test Accuracy', fontsize=14, fontweight='bold')
        ax.set_ylim([min(accuracies) - 2, 100])
        ax.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Highlight best
        best_idx = accuracies.index(max(accuracies))
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"      ✓ Saved: {save_path}")
        plt.close()
    
    def generate_summary_table(self, y_true, y_pred, save_path='figures/results_table.csv'):
        """Generate summary results table"""
        print("[6/6] Generating summary table...")
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1, 2, 3, 4], average=None, zero_division=0
        )
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
        
        per_class_acc = np.zeros(5)
        for i in range(5):
            if cm[i].sum() > 0:
                per_class_acc[i] = cm[i, i] / cm[i].sum()
        
        results_df = pd.DataFrame({
            'Attack Type': self.attack_names,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Accuracy': per_class_acc,
            'Support': support
        })
        
        overall = pd.DataFrame({
            'Attack Type': ['Overall'],
            'Precision': [precision[support > 0].mean()],
            'Recall': [recall[support > 0].mean()],
            'F1-Score': [f1[support > 0].mean()],
            'Accuracy': [accuracy_score(y_true, y_pred)],
            'Support': [support.sum()]
        })
        
        results_df = pd.concat([results_df, overall], ignore_index=True)
        
        results_df.to_csv(save_path, index=False, float_format='%.4f')
        print(f"      ✓ Saved: {save_path}")
        print("\n" + "="*60)
        print("SUMMARY RESULTS:")
        print("="*60)
        print(results_df.to_string(index=False))
        
        return results_df
    
    def generate_all_figures(self):
        """Generate all figures"""
        print("\n" + "="*60)
        print("GENERATING ALL VISUALIZATIONS")
        print("="*60)
        
        test_loader = self.load_test_data()
        y_true, y_pred, y_probs = self.get_predictions(test_loader)
        
        self.plot_confusion_matrix(y_true, y_pred)
        self.plot_per_class_metrics(y_true, y_pred)
        self.plot_roc_curves(y_true, y_probs)
        self.plot_precision_recall_curves(y_true, y_probs)
        self.plot_accuracy_comparison(y_true, y_pred)
        results_df = self.generate_summary_table(y_true, y_pred)
        
        print("\n" + "="*60)
        print("✅ ALL VISUALIZATIONS COMPLETE!")
        print("="*60)
        print("\nFiles saved in 'figures/' directory:")
        print("  - confusion_matrix.png")
        print("  - per_class_metrics.png")
        print("  - roc_curves.png")
        print("  - precision_recall_curves.png")
        print("  - accuracy_comparison.png")
        print("  - results_table.csv")
        print("\n🎉 Ready for your research paper!")
        
        return results_df

def main():
    visualizer = TGTVisualizer(checkpoint_path='checkpoints/best_model.pt')
    results = visualizer.generate_all_figures()

if __name__ == "__main__":
    main()
