"""
=============================================================================
FraudGraph-AI — Graph Convolutional Network (GCN)
=============================================================================

MENTOR NOTE — How GCN Works:

A GCN layer performs NEIGHBORHOOD AGGREGATION. For each node v:

    h_v^(l+1) = σ( Σ_{u∈N(v)∪{v}} (1/√(d_u · d_v)) · W^(l) · h_u^(l) )

In plain English:
1. Each node COLLECTS feature vectors from its neighbors
2. Features are WEIGHTED by the inverse of neighbor degrees
   (popular nodes contribute less per-connection)
3. A learnable weight matrix W TRANSFORMS the aggregated features
4. A nonlinearity σ (ReLU) is applied

WHY THIS WORKS FOR FRAUD:
- Fraudulent transactions often form CLUSTERS (money laundering rings)
- A GCN can detect that "this transaction's neighbors are mostly illicit"
- Information propagates through the graph: even if a node looks normal
  in isolation, its 2-hop neighborhood reveals suspicious patterns

ARCHITECTURE:
    Input (165 features)
    → GCNConv(165 → 128) → BatchNorm → ReLU → Dropout(0.3)
    → GCNConv(128 → 64)  → BatchNorm → ReLU → Dropout(0.3)
    → Linear(64 → 1)     [Binary classification logit]
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm


class GCNFraudDetector(nn.Module):
    """
    2-layer GCN for binary node classification (fraud detection).
    """

    def __init__(self, in_channels, hidden_channels=128,
                 out_channels=64, dropout=0.3):
        super().__init__()

        # GCN message-passing layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

        # Batch normalisation stabilises training on imbalanced data
        self.bn1 = BatchNorm(hidden_channels)
        self.bn2 = BatchNorm(out_channels)

        # Classification head — single logit for BCEWithLogitsLoss
        self.classifier = nn.Linear(out_channels, 1)

        self.dropout = dropout

    def forward(self, x, edge_index):
        """
        Forward pass.

        Args:
            x:          Node features  [N, in_channels]
            edge_index: Edge list      [2, E]

        Returns:
            logits: Raw logits [N, 1] — apply sigmoid for probabilities
        """
        # Layer 1: aggregate → normalise → activate → drop
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 2
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Classifier
        logits = self.classifier(h)
        return logits.squeeze(-1)   # [N]

    def get_embeddings(self, x, edge_index):
        """Return intermediate node embeddings (for explainability)."""
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.relu(h)

        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.relu(h)
        return h  # [N, out_channels]
