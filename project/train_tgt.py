"""
Training Script for Temporal Graph Transformer
Optimized for H100 GPU with mixed precision and experiment tracking
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import wandb
from tqdm import tqdm
import yaml

# Import our custom modules
from tgt_model import TemporalGraphTransformer, TGT_Lightweight
from graph_builder import WSNGraphBuilder


class TGTTrainer:
    """Trainer class for Temporal Graph Transformer"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize wandb
        if config.get('use_wandb', True):
            wandb.init(
                project="temporal-graph-transformer-wsn",
                config=config,
                name=config.get('experiment_name', 'tgt-experiment')
            )

        # Build model
        self.model = self._build_model()
        self.model.to(self.device)

        # Setup optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-4)
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=1e-6
        )

        # Loss function with class weights
        self.criterion = nn.CrossEntropyLoss()

        # Mixed precision training (H100 optimization)
        self.use_amp = config.get('use_amp', True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # Metrics tracking
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_accuracies = []

    def _build_model(self):
        """Build the model based on config"""
        if self.config.get('model_type', 'full') == 'lightweight':
            model = TGT_Lightweight(
                num_node_features=self.config['num_features'],
                num_classes=self.config['num_classes']
            )
        else:
            model = TemporalGraphTransformer(
                num_node_features=self.config['num_features'],
                hidden_dim=self.config.get('hidden_dim', 128),
                num_gnn_layers=self.config.get('num_gnn_layers', 3),
                lstm_hidden=self.config.get('lstm_hidden', 256),
                num_heads=self.config.get('num_heads', 8),
                num_classes=self.config['num_classes'],
                dropout=self.config.get('dropout', 0.3)
            )

        return model

    def prepare_data(self):
        """Load and prepare WSN-DS dataset"""
        print("\n" + "="*60)
        print("PREPARING DATA")
        print("="*60)

        # Load WSN-DS dataset
        data_path = self.config.get('data_path', 'data/raw/WSN-DS.csv')
        print(f"Loading data from: {data_path}")

        # Use the preprocessed data from wsn_intrusion_detection_fixed.py
        df = pd.read_csv(data_path)

        # Clean headers
        df.columns = df.columns.str.strip().str.replace(' ', '_')

        # Drop simulation artifacts
        if 'id' in df.columns:
            df = df.drop(columns=['id'])
        if 'Time' in df.columns:
            df = df.drop(columns=['Time'])

        # Encode attack types
        mapping = {
            'Normal': 0,
            'Blackhole': 1,
            'Grayhole': 2,
            'Flooding': 3,
            'TDMA': 4
        }
        df['Attack_type_encoded'] = df['Attack_type'].map(mapping)

        print(f"Dataset shape: {df.shape}")
        print(f"Attack distribution:\n{df['Attack_type_encoded'].value_counts()}")

        # Create graph snapshots
        print("\nGenerating graph structures...")
        graph_builder = WSNGraphBuilder(
            num_nodes=self.config.get('num_nodes', 50),
            connection_radius=self.config.get('connection_radius', 0.3)
        )

        graphs = graph_builder.create_temporal_snapshots(
            df,
            window_size=self.config.get('window_size', 10),
            stride=self.config.get('stride', 5)
        )

        print(f"Generated {len(graphs)} graph snapshots")

        # Train/Val/Test split
        from sklearn.model_selection import train_test_split

        train_graphs, test_graphs = train_test_split(
            graphs, test_size=0.2, random_state=42
        )
        train_graphs, val_graphs = train_test_split(
            train_graphs, test_size=0.15, random_state=42
        )

        print(f"\nSplit: Train={len(train_graphs)}, Val={len(val_graphs)}, Test={len(test_graphs)}")

        # Create dataloaders
        train_loader = GeoDataLoader(
            train_graphs,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config.get('num_workers', 4)
        )

        val_loader = GeoDataLoader(
            val_graphs,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 4)
        )

        test_loader = GeoDataLoader(
            test_graphs,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config.get('num_workers', 4)
        )

        return train_loader, val_loader, test_loader

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}')

        for batch in pbar:
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            # Mixed precision forward pass
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits, _ = self.model(batch.x, batch.edge_index, batch.batch)
                    loss = self.criterion(logits, batch.y)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits, _ = self.model(batch.x, batch.edge_index, batch.batch)
                loss = self.criterion(logits, batch.y)
                loss.backward()
                self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    @torch.no_grad()
    def evaluate(self, loader, split='Val'):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch in tqdm(loader, desc=f'Evaluating {split}'):
            batch = batch.to(self.device)

            logits, _ = self.model(batch.x, batch.edge_index, batch.batch)
            loss = self.criterion(logits, batch.y)

            total_loss += loss.item()
            pred = logits.argmax(dim=1)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

        avg_loss = total_loss / len(loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )

        return {
            'loss': avg_loss,
            'accuracy': accuracy * 100,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_preds,
            'labels': all_labels
        }

    def train(self, train_loader, val_loader, test_loader):
        """Full training loop"""
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Model: {self.config.get('model_type', 'full').upper()}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Total Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        start_time = time.time()

        for epoch in range(self.config['epochs']):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = self.evaluate(val_loader, 'Val')

            # Scheduler step
            self.scheduler.step()

            # Log metrics
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}%")
            print(f"  Val F1: {val_metrics['f1']:.4f}")

            if self.config.get('use_wandb', True):
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'val_f1': val_metrics['f1'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })

            # Save best model
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self.save_checkpoint(f"best_model.pt", epoch, val_metrics)
                print(f"  ✓ New best model saved! (Val Acc: {self.best_val_acc:.2f}%)")

        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time/60:.2f} minutes")

        # Final test evaluation
        print("\n" + "="*60)
        print("FINAL TEST EVALUATION")
        print("="*60)
        test_metrics = self.evaluate(test_loader, 'Test')

        print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
        print(f"Test F1-Score: {test_metrics['f1']:.4f}")
        print(f"Test Precision: {test_metrics['precision']:.4f}")
        print(f"Test Recall: {test_metrics['recall']:.4f}")

        # Confusion matrix
        cm = confusion_matrix(test_metrics['labels'], test_metrics['predictions'])
        print(f"\nConfusion Matrix:\n{cm}")

        if self.config.get('use_wandb', True):
            wandb.log({
                'test_accuracy': test_metrics['accuracy'],
                'test_f1': test_metrics['f1'],
                'test_precision': test_metrics['precision'],
                'test_recall': test_metrics['recall']
            })

        return test_metrics

    def save_checkpoint(self, filename, epoch, metrics):
        """Save model checkpoint"""
        os.makedirs('checkpoints', exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }, f'checkpoints/{filename}')


def main():
    # Configuration
    config = {
        # Data
        'data_path': 'data/raw/WSN-DS.csv',
        'num_features': 16,  # Will be auto-detected
        'num_classes': 5,
        'num_nodes': 50,
        'connection_radius': 0.3,
        'window_size': 10,
        'stride': 5,

        # Model
        'model_type': 'full',  # 'full' or 'lightweight'
        'hidden_dim': 128,
        'num_gnn_layers': 3,
        'lstm_hidden': 256,
        'num_heads': 8,
        'dropout': 0.3,

        # Training
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'use_amp': True,  # Mixed precision for H100
        'num_workers': 4,

        # Experiment
        'experiment_name': 'tgt-baseline',
        'use_wandb': False,  # Set to True for experiment tracking
    }

    # Initialize trainer
    trainer = TGTTrainer(config)

    # Prepare data
    train_loader, val_loader, test_loader = trainer.prepare_data()

    # Train
    test_metrics = trainer.train(train_loader, val_loader, test_loader)

    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
