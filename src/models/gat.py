"""
=============================================================================
FraudGraph-AI — Graph Attention Network (GAT)
=============================================================================

MENTOR NOTE — GCN vs GAT:

GCN treats ALL neighbors equally (weighted only by degree).
GAT learns ATTENTION WEIGHTS — "which neighbors matter more?"

This is critical for fraud detection because:
• A legitimate exchange node may have 1,000 connections, 999 licit + 1 illicit.
  GCN averages them all → the fraud signal is diluted.
• GAT can learn to ATTEND to the suspicious connection and ignore the rest.

The attention mechanism:
    α_ij = softmax( LeakyReLU( a^T [W·h_i || W·h_j] ) )
    h_i' = σ( Σ_j α_ij · W · h_j )

MULTI-HEAD ATTENTION:
We run K independent attention heads and concatenate (or average) results.
This gives the model K different "perspectives" on neighborhood importance.

ARCHITECTURE:
    Input (165 features)
    → GATConv(165 → 32, heads=4)  → ELU → Dropout(0.3)  [output: 128]
    → GATConv(128 → 64, heads=2, concat=False) → ELU → Dropout(0.3) [output: 64]
    → Linear(64 → 1)
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm


class GATFraudDetector(nn.Module):
    """
    2-layer multi-head GAT for binary node classification.
    Returns attention weights for explainability.
    """

    def __init__(self, in_channels, hidden_channels=32,
                 heads_1=4, heads_2=2, dropout=0.3):
        super().__init__()

        # Layer 1: multi-head attention, outputs concatenated
        # Output dim = hidden_channels * heads_1
        self.conv1 = GATConv(
            in_channels, hidden_channels,
            heads=heads_1, dropout=dropout, concat=True
        )
        self.bn1 = BatchNorm(hidden_channels * heads_1)

        # Layer 2: multi-head attention, output averaged (concat=False)
        # Input dim = hidden_channels * heads_1, output = hidden_channels * heads_2
        out_dim = hidden_channels * heads_2
        self.conv2 = GATConv(
            hidden_channels * heads_1, hidden_channels * heads_2,
            heads=1, dropout=dropout, concat=False
        )
        self.bn2 = BatchNorm(out_dim)

        # Classifier
        self.classifier = nn.Linear(out_dim, 1)
        self.dropout = dropout

        # Store attention weights for explainability
        self._attn_weights_1 = None
        self._attn_weights_2 = None

    def forward(self, x, edge_index, return_attention=False):
        """
        Forward pass.

        Args:
            x:                Node features [N, in_channels]
            edge_index:       Edge list [2, E]
            return_attention:  If True, also return attention weights

        Returns:
            logits: [N] raw logits
            (optional) attn_weights: tuple of attention weight tensors
        """
        # Layer 1
        h, attn1 = self.conv1(x, edge_index, return_attention_weights=True)
        self._attn_weights_1 = attn1
        h = self.bn1(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 2
        h, attn2 = self.conv2(h, edge_index, return_attention_weights=True)
        self._attn_weights_2 = attn2
        h = self.bn2(h)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        logits = self.classifier(h).squeeze(-1)

        if return_attention:
            return logits, (attn1, attn2)
        return logits

    def get_embeddings(self, x, edge_index):
        """Return intermediate node embeddings."""
        h = self.conv1(x, edge_index)
        h = self.bn1(h)
        h = F.elu(h)
        h = self.conv2(h, edge_index)
        h = self.bn2(h)
        h = F.elu(h)
        return h

    def get_attention_weights(self):
        """Return stored attention weights from last forward pass."""
        return self._attn_weights_1, self._attn_weights_2
