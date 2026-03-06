"""
Temporal Graph Transformer (TGT) Model
Novel architecture combining:
- Graph Convolution for spatial relationships
- Bi-LSTM for temporal dependencies
- Multi-head attention for feature importance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool

class TemporalGraphTransformer(nn.Module):
    """
    Temporal Graph Transformer for WSN Intrusion Detection

    Architecture:
    1. Graph Convolution Layers (spatial)
    2. Bi-LSTM Layer (temporal)
    3. Multi-Head Attention (feature importance)
    4. Graph Pooling + Classification
    """

    def __init__(self, 
                 num_node_features,
                 hidden_dim=128,
                 num_gnn_layers=3,
                 lstm_hidden=256,
                 num_heads=8,
                 num_classes=5,
                 dropout=0.3):
        super(TemporalGraphTransformer, self).__init__()

        self.num_node_features = num_node_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # ===== Graph Convolution Layers =====
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(GATConv(num_node_features, hidden_dim, heads=4, concat=True))

        for _ in range(num_gnn_layers - 1):
            self.gnn_layers.append(GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True))

        # Final GNN layer
        self.gnn_final = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False)

        self.gnn_norm = nn.ModuleList([nn.BatchNorm1d(hidden_dim * 4) for _ in range(num_gnn_layers)])
        self.gnn_norm.append(nn.BatchNorm1d(hidden_dim))

        # ===== Temporal Module (Bi-LSTM) =====
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # ===== Multi-Head Attention =====
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden * 2,  # bidirectional
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # ===== Graph-level Pooling =====
        self.pool_transform = nn.Linear(lstm_hidden * 2, hidden_dim)

        # ===== Classification Head =====
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for mean+max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch=None):
        """
        Forward pass

        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment vector [num_nodes]

        Returns:
            logits: Class predictions [batch_size, num_classes]
        """
        # ===== Step 1: Graph Convolution (Spatial) =====
        h = x
        for i, gnn_layer in enumerate(self.gnn_layers):
            h = gnn_layer(h, edge_index)
            h = self.gnn_norm[i](h)
            h = F.elu(h)
            h = self.dropout(h)

        # Final GNN layer
        h = self.gnn_final(h, edge_index)
        h = self.gnn_norm[-1](h)
        h = F.elu(h)

        # ===== Step 2: Prepare for Temporal Processing =====
        # Reshape to [batch_size, num_nodes, hidden_dim] for LSTM
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        batch_size = batch.max().item() + 1
        num_nodes = h.size(0) // batch_size

        # Reshape: [batch_size, num_nodes, hidden_dim]
        h_temporal = h.view(batch_size, num_nodes, -1)

        # ===== Step 3: Bi-LSTM (Temporal) =====
        lstm_out, (h_n, c_n) = self.lstm(h_temporal)
        # lstm_out: [batch_size, num_nodes, lstm_hidden*2]

        # ===== Step 4: Multi-Head Attention =====
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        # attn_out: [batch_size, num_nodes, lstm_hidden*2]

        # ===== Step 5: Graph-level Pooling =====
        # Transform back to graph representation
        attn_out = self.pool_transform(attn_out)

        # Reshape back to [num_nodes, hidden_dim]
        attn_out = attn_out.contiguous().view(-1, self.hidden_dim)

        # Global pooling (both mean and max)
        graph_embedding_mean = global_mean_pool(attn_out, batch)
        graph_embedding_max = global_max_pool(attn_out, batch)

        # Concatenate pooled representations
        graph_embedding = torch.cat([graph_embedding_mean, graph_embedding_max], dim=1)

        # ===== Step 6: Classification =====
        logits = self.classifier(graph_embedding)

        return logits, attn_weights

    def get_attention_weights(self):
        """Return attention weights for visualization"""
        return self.attention_weights


class TGT_Lightweight(nn.Module):
    """
    Lightweight version of TGT for faster training
    Use this for initial experiments
    """

    def __init__(self, num_node_features, num_classes=5):
        super(TGT_Lightweight, self).__init__()

        self.gnn1 = GCNConv(num_node_features, 64)
        self.gnn2 = GCNConv(64, 64)

        self.lstm = nn.LSTM(64, 128, batch_first=True, bidirectional=True)

        self.attention = nn.MultiheadAttention(256, num_heads=4, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, edge_index, batch=None):
        # GNN
        h = F.relu(self.gnn1(x, edge_index))
        h = F.relu(self.gnn2(h, edge_index))

        # Temporal
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)

        batch_size = batch.max().item() + 1
        num_nodes = h.size(0) // batch_size
        h_temporal = h.view(batch_size, num_nodes, -1)

        lstm_out, _ = self.lstm(h_temporal)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Pool - FIXED: use reshape instead of view
        attn_flat = attn_out.reshape(-1, 256)
        graph_emb = global_mean_pool(attn_flat, batch)

        # Classify
        logits = self.classifier(graph_emb)
        return logits, None


if __name__ == "__main__":
    # Test model
    print("Testing Temporal Graph Transformer...")
    print("="*60)

    # Create dummy data
    num_nodes = 50
    num_features = 16
    batch_size = 32

    x = torch.randn(num_nodes * batch_size, num_features)
    edge_index = torch.randint(0, num_nodes * batch_size, (2, 200))
    batch = torch.repeat_interleave(torch.arange(batch_size), num_nodes)

    # Test full model
    print("\n1. Testing Full TGT Model...")
    model = TemporalGraphTransformer(
        num_node_features=num_features,
        hidden_dim=128,
        num_classes=5
    )

    logits, attn_weights = model(x, edge_index, batch)
    print(f"   ✓ Output shape: {logits.shape}")
    print(f"   ✓ Attention weights shape: {attn_weights.shape}")

    # Test lightweight model
    print("\n2. Testing Lightweight TGT Model...")
    model_light = TGT_Lightweight(num_node_features=num_features)
    logits_light, _ = model_light(x, edge_index, batch)
    print(f"   ✓ Lightweight output shape: {logits_light.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n3. Model Statistics:")
    print(f"   Full model parameters: {total_params:,}")

    total_params_light = sum(p.numel() for p in model_light.parameters())
    print(f"   Lightweight parameters: {total_params_light:,}")
    print(f"   Size reduction: {(1 - total_params_light/total_params)*100:.1f}%")

    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
