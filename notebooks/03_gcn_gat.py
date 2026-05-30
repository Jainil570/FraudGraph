# %% [markdown]
# # 🧠 FraudGraph-AI -- Steps 6-12: GNN Training & Evaluation
#
# This notebook covers the core of the project:
# - **Step 6:** Graph Construction & Theory
# - **Step 7:** GCN Model
# - **Step 8:** GAT Model
# - **Step 9:** Training Pipeline
# - **Step 10:** Evaluation & Comparison
# - **Step 11:** Explainable AI
# - **Step 12:** MLflow Experiment Tracking
#
# ---
#
# ## Step 6 -- Graph Construction Theory
#
# ### What is a Graph in this context?
#
# ```
# Transaction A ---> Transaction B ---> Transaction C
#       |                                    |
#       └---> Transaction D ---> Transaction E ┘
# ```
#
# Each **node** = one Bitcoin transaction (203,769 total)
# Each **edge** = a payment flow from one transaction to another (234,355 total)
# Each node has **165 features** (amount, time patterns, aggregated stats)
#
# ### edge_index Format (PyTorch Geometric)
#
# PyG stores edges as a `[2, E]` tensor:
# ```
# edge_index = [[src_0, src_1, src_2, ...],
#               [dst_0, dst_1, dst_2, ...]]
# ```
# Edge 0 goes from node `src_0` to node `dst_0`.
#
# ### Message Passing (How GNNs Work)
#
# Each GNN layer performs three steps:
# 1. **MESSAGE**: Each node sends its features to its neighbors
# 2. **AGGREGATE**: Each node collects messages from ALL neighbors
# 3. **UPDATE**: Aggregated messages are combined with the node's own features
#
# After 2 layers, each node has information from its **2-hop neighborhood**.
# This is how a GNN detects that "my neighbor's neighbors are mostly fraud".
#
# ### Transductive Learning
#
# Unlike typical ML where train/test are separate datasets, here ALL nodes
# exist in ONE graph. The model sees the test node features and edges
# during training -- it just doesn't see their LABELS. This is transductive
# learning, and it's natural for fraud detection where new transactions
# connect to existing ones.

# %%
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from src.utils.data_loader import prepare_data
from src.utils.config import GCN_PARAMS, GAT_PARAMS, DEVICE
from src.models.gcn import GCNFraudDetector
from src.models.gat import GATFraudDetector
from src.training.trainer import train_gnn
from src.evaluation.evaluator import evaluate_gnn, run_full_evaluation
from src.evaluation.explainability import (
    explain_xgboost, explain_gnn_embeddings, visualize_attention_weights,
)
from src.tracking.mlflow_tracker import (
    setup_experiment, log_gnn_run,
)
from src.utils.visualization import plot_training_history
from src.utils.metrics import format_comparison_table
import matplotlib.pyplot as plt

# %% [markdown]
# ## Load & Prepare Data

# %%
bundle = prepare_data()
data = bundle['data']
labels = bundle['binary_labels']
train_mask = bundle['train_mask']
val_mask = bundle['val_mask']
test_mask = bundle['test_mask']
pos_weight = bundle['pos_weight']

print(f"\nData on: {bundle['device']}")
print(f"Feature shape: {data.x.shape}")
print(f"Edge shape:    {data.edge_index.shape}")

# %% [markdown]
# ## Step 7 -- Build & Train GCN
#
# ### How GCN Works
#
# The Graph Convolutional Network aggregates neighbor features using a
# **symmetric normalised adjacency matrix**:
#
# ```
# H^(l+1) = σ( D̃^(-½) Ã D̃^(-½) H^(l) W^(l) )
# ```
#
# Where:
# - Ã = A + I (adjacency + self-loops)
# - D̃ = degree matrix of Ã
# - W = learnable weight matrix
# - σ = ReLU activation
#
# **Key limitation**: GCN weights ALL neighbors equally (normalised by degree).
# A node with 100 legit neighbors and 1 fraud neighbor will dilute the fraud
# signal through averaging.

# %%
gcn_model = GCNFraudDetector(
    in_channels=GCN_PARAMS['in_channels'],
    hidden_channels=GCN_PARAMS['hidden_channels'],
    out_channels=GCN_PARAMS['out_channels'],
    dropout=GCN_PARAMS['dropout'],
)
print(f"GCN parameters: {sum(p.numel() for p in gcn_model.parameters()):,}")
print(gcn_model)

# %%
gcn_model, gcn_history = train_gnn(
    model=gcn_model,
    data=data,
    binary_labels=labels,
    train_mask=train_mask,
    val_mask=val_mask,
    pos_weight=pos_weight,
    lr=GCN_PARAMS['lr'],
    weight_decay=GCN_PARAMS['weight_decay'],
    epochs=GCN_PARAMS['epochs'],
    patience=GCN_PARAMS['patience'],
    model_name="GCN",
)

# %%
fig = plot_training_history(gcn_history, "GCN")
plt.show()

# %% [markdown]
# ## Step 8 -- Build & Train GAT
#
# ### GCN vs GAT -- The Key Difference
#
# | Aspect | GCN | GAT |
# |--------|-----|-----|
# | Neighbor weighting | Fixed (degree-based) | Learned (attention) |
# | Fraud ring detection | Average dilutes signal | Attends to suspicious neighbors |
# | Interpretability | Black-box aggregation | Attention weights show "why" |
# | Computational cost | Cheaper | More expensive |
#
# ### Why GAT is Better for Fraud
#
# GAT computes an **attention coefficient** α_ij for each edge (i,j):
# ```
# α_ij = softmax_j( LeakyReLU( a^T [W·h_i || W·h_j] ) )
# ```
#
# This means: the model LEARNS which neighbors are important.
# For a node with 99 legit + 1 fraud neighbor, GAT can assign
# attention weight 0.8 to the fraud neighbor and 0.002 to each legit one.
#
# **Multi-head attention** runs K independent attention mechanisms and
# concatenates results -> K different "perspectives" on importance.

# %%
gat_model = GATFraudDetector(
    in_channels=GAT_PARAMS['in_channels'],
    hidden_channels=GAT_PARAMS['hidden_channels'],
    heads_1=GAT_PARAMS['heads_1'],
    heads_2=GAT_PARAMS['heads_2'],
    dropout=GAT_PARAMS['dropout'],
)
print(f"GAT parameters: {sum(p.numel() for p in gat_model.parameters()):,}")
print(gat_model)

# %%
gat_model, gat_history = train_gnn(
    model=gat_model,
    data=data,
    binary_labels=labels,
    train_mask=train_mask,
    val_mask=val_mask,
    pos_weight=pos_weight,
    lr=GAT_PARAMS['lr'],
    weight_decay=GAT_PARAMS['weight_decay'],
    epochs=GAT_PARAMS['epochs'],
    patience=GAT_PARAMS['patience'],
    model_name="GAT",
)

# %%
fig = plot_training_history(gat_history, "GAT")
plt.show()

# %% [markdown]
# ## Step 10 -- Evaluation & Comparison

# %%
print("\n" + "="*60)
print("  EVALUATING ALL MODELS ON TEST SET")
print("="*60 + "\n")

gcn_results = evaluate_gnn(gcn_model, data, labels, test_mask, "GCN")
gat_results = evaluate_gnn(gat_model, data, labels, test_mask, "GAT")

all_results = {'GCN': gcn_results, 'GAT': gat_results}
metrics_dict = {k: v['metrics'] for k, v in all_results.items()}
print("\n" + format_comparison_table(metrics_dict))

# %% [markdown]
# ## Step 11 -- Explainable AI
#
# ### Why Explainability Matters in Fintech
#
# Regulators (GDPR Article 22, US Equal Credit Opportunity Act) require
# that automated decisions affecting individuals be **explainable**.
# A black-box "this is fraud" prediction is not acceptable.
#
# We need to answer: **"WHY does the model think this is fraud?"**

# %%
# GAT Attention Weights -- which neighbors influenced each prediction?
print("\n[EXPLAIN] Visualising GAT attention weights...")
attn_results = visualize_attention_weights(
    gat_model, data, test_mask, labels, top_n=5
)

for r in attn_results[:3]:
    print(f"\n  Node {r['node_id']}: P(fraud)={r['predicted_prob']:.3f}, "
          f"actual={'FRAUD' if r['actual_label'] else 'LEGIT'}")
    for nid, attn in sorted(r['top_neighbors'], key=lambda x: -x[1])[:5]:
        print(f"    -> Neighbor {nid}: attention={attn:.4f}")

# %%
# SHAP on GNN embeddings (surrogate approach)
print("\n[EXPLAIN] SHAP on GNN embeddings...")
try:
    shap_values = explain_gnn_embeddings(
        gat_model, data, test_mask, labels, n_samples=100
    )
    print("[EXPLAIN] SHAP analysis complete. Check outputs/ for plots.")
except Exception as e:
    print(f"[WARN] SHAP failed (may need more samples): {e}")

# %% [markdown]
# ## Step 12 -- MLflow Experiment Tracking

# %%
setup_experiment("FraudGraph-AI")

# Log GCN run
log_gnn_run(
    model_name="GCN",
    params=GCN_PARAMS,
    metrics=gcn_results['metrics'],
    history=gcn_history,
    model_path=os.path.join("models", "gcn_best.pt"),
)

# Log GAT run
log_gnn_run(
    model_name="GAT",
    params=GAT_PARAMS,
    metrics=gat_results['metrics'],
    history=gat_history,
    model_path=os.path.join("models", "gat_best.pt"),
)

print("\n[MLflow] All runs logged. View dashboard with: mlflow ui")

# %% [markdown]
# ## Summary
#
# | What We Did | Why |
# |-------------|-----|
# | GCN (2-layer) | Baseline GNN -- uniform neighbor aggregation |
# | GAT (multi-head) | Learned attention -- better for fraud rings |
# | Weighted BCE | Handle 2% imbalance without oversampling |
# | Temporal split | Prevent data leakage from future transactions |
# | Early stopping on PR-AUC | Optimise for fraud ranking, not loss |
# | SHAP + Attention viz | Regulatory compliance & analyst trust |
# | MLflow tracking | Reproducibility & experiment comparison |
#
# **Next steps:** Deploy with `uvicorn api.main:app` and containerise with Docker.

# %%
print("\n✅ GNN training, evaluation, and tracking complete!")
print("   Run `mlflow ui` to view the experiment dashboard.")
print("   Run `uvicorn api.main:app` to start the prediction API.")
