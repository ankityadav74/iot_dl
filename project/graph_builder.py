"""
Graph Builder for WSN Topology Generation
Converts tabular WSN-DS data into graph structures
"""

import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import pickle

class WSNGraphBuilder:
    """
    Builds graph structures from WSN-DS dataset

    Graph Structure:
    - Nodes: Individual sensor readings (time steps)
    - Edges: Temporal connections + spatial proximity (simulated)
    - Node features: 16 features from WSN-DS
    - Labels: Attack types (0-4)
    """

    def __init__(self, num_nodes=50, connection_radius=0.3):
        """
        Args:
            num_nodes: Number of sensor nodes in WSN
            connection_radius: Proximity threshold for edge creation
        """
        self.num_nodes = num_nodes
        self.connection_radius = connection_radius
        self.scaler = StandardScaler()
        self.node_positions = None

    def generate_wsn_topology(self, topology_type='random'):
        """
        Generate WSN network topology

        Args:
            topology_type: 'random', 'grid', 'cluster', or 'hierarchical'
        """
        if topology_type == 'random':
            # Random deployment in 2D space
            positions = np.random.rand(self.num_nodes, 2)

        elif topology_type == 'grid':
            # Grid topology (common in WSN)
            grid_size = int(np.sqrt(self.num_nodes))
            x = np.linspace(0, 1, grid_size)
            y = np.linspace(0, 1, grid_size)
            xv, yv = np.meshgrid(x, y)
            positions = np.column_stack([xv.ravel(), yv.ravel()])[:self.num_nodes]

        elif topology_type == 'cluster':
            # Clustered topology (realistic for hierarchical WSN)
            num_clusters = 5
            positions = []
            nodes_per_cluster = self.num_nodes // num_clusters
            for i in range(num_clusters):
                center = np.random.rand(2)
                cluster_nodes = center + np.random.randn(nodes_per_cluster, 2) * 0.1
                positions.append(cluster_nodes)
            positions = np.vstack(positions)[:self.num_nodes]

        else:  # hierarchical
            # Base station at center, nodes in concentric circles
            angles = np.linspace(0, 2*np.pi, self.num_nodes, endpoint=False)
            radii = np.random.uniform(0.2, 0.9, self.num_nodes)
            positions = np.column_stack([
                0.5 + radii * np.cos(angles),
                0.5 + radii * np.sin(angles)
            ])

        self.node_positions = positions
        return positions

    def build_adjacency_matrix(self, positions):
        """
        Build adjacency matrix based on spatial proximity
        """
        num_nodes = len(positions)
        adj_matrix = np.zeros((num_nodes, num_nodes))

        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                distance = np.linalg.norm(positions[i] - positions[j])
                if distance < self.connection_radius:
                    adj_matrix[i, j] = 1
                    adj_matrix[j, i] = 1

        return adj_matrix

    def create_temporal_snapshots(self, df, window_size=10, stride=5):
        """
        Create temporal graph snapshots from sequential data

        Args:
            df: DataFrame with WSN-DS data
            window_size: Number of time steps per snapshot
            stride: Step size for sliding window

        Returns:
            List of PyG Data objects
        """
        graphs = []
        feature_cols = [col for col in df.columns if col not in ['Attack_type', 'Attack_type_encoded']]

        # Normalize features
        features = self.scaler.fit_transform(df[feature_cols].values)
        labels = df['Attack_type_encoded'].values

        # Generate topology once
        if self.node_positions is None:
            self.generate_wsn_topology('cluster')

        adj_matrix = self.build_adjacency_matrix(self.node_positions)
        edge_index = self._adj_to_edge_index(adj_matrix)

        # Create sliding windows
        for i in range(0, len(df) - window_size, stride):
            window_features = features[i:i+window_size]
            window_labels = labels[i:i+window_size]

            # Assign features to nodes (distribute across topology)
            node_features = self._distribute_features_to_nodes(window_features)

            # Majority vote for graph label
            graph_label = np.bincount(window_labels).argmax()

            # Create PyG Data object
            graph = Data(
                x=torch.FloatTensor(node_features),
                edge_index=torch.LongTensor(edge_index),
                y=torch.LongTensor([graph_label]),
                num_nodes=self.num_nodes,
                pos=torch.FloatTensor(self.node_positions)
            )

            graphs.append(graph)

        return graphs

    def _distribute_features_to_nodes(self, window_features):
        """
        Distribute temporal features across spatial nodes
        Strategy: Each node gets a subset of temporal features
        """
        window_size, num_features = window_features.shape

        # Repeat features across nodes with noise for diversity
        node_features = np.zeros((self.num_nodes, num_features))

        # Assign features based on time-node mapping
        for i in range(self.num_nodes):
            time_idx = i % window_size
            node_features[i] = window_features[time_idx]
            # Add small noise for variation
            node_features[i] += np.random.randn(num_features) * 0.01

        return node_features

    def _adj_to_edge_index(self, adj_matrix):
        """
        Convert adjacency matrix to edge_index format (COO)
        """
        edges = np.array(np.where(adj_matrix == 1))
        return edges

    def save_graphs(self, graphs, filepath):
        """Save generated graphs to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(graphs, f)
        print(f"Saved {len(graphs)} graphs to {filepath}")

    def visualize_topology(self, save_path='wsn_topology.png'):
        """Visualize the WSN topology"""
        import matplotlib.pyplot as plt

        if self.node_positions is None:
            print("Generate topology first!")
            return

        adj_matrix = self.build_adjacency_matrix(self.node_positions)
        G = nx.from_numpy_array(adj_matrix)

        plt.figure(figsize=(10, 10))
        pos = {i: self.node_positions[i] for i in range(len(self.node_positions))}
        nx.draw(G, pos, node_size=100, node_color='lightblue', 
                with_labels=True, font_size=8, edge_color='gray', alpha=0.6)
        plt.title(f'WSN Topology ({self.num_nodes} nodes, radius={self.connection_radius})')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Topology visualization saved to {save_path}")


if __name__ == "__main__":
    # Test the graph builder
    print("Testing WSN Graph Builder...")

    # Create sample data
    sample_df = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'Attack_type_encoded': np.random.randint(0, 5, 1000)
    })

    builder = WSNGraphBuilder(num_nodes=50, connection_radius=0.3)
    builder.generate_wsn_topology('cluster')
    builder.visualize_topology()

    graphs = builder.create_temporal_snapshots(sample_df, window_size=10, stride=5)
    print(f"Generated {len(graphs)} graph snapshots")
    print(f"Sample graph: {graphs[0]}")
