# %% [markdown]
# # 📊 FraudGraph-AI — Step 1 & 3: Exploratory Data Analysis
#
# **Objective:** Understand the Elliptic Bitcoin Dataset before building models.
#
# ## Why This Matters (Business Context)
#
# Financial fraud costs the global economy **$5.4 trillion annually**. Traditional
# rule-based systems catch simple fraud (stolen cards, known patterns) but fail
# against **organized fraud rings** — networks of accounts that launder money
# through chains of seemingly legitimate transactions.
#
# ### Why Traditional ML Fails on Fraud Rings
#
# A single fraudulent transaction may look identical to a legitimate one in terms
# of amount, time, and frequency. The fraud signal lives in the **GRAPH STRUCTURE**:
# - Money flowing through many intermediary accounts before cashing out
# - Clusters of accounts created around the same time
# - Circular transaction patterns (A→B→C→A)
#
# Traditional ML (Random Forest, XGBoost) sees each transaction **independently**.
# Graph Neural Networks see the **neighborhood** — who sent money, who received
# it, and what THEIR transaction patterns look like.
#
# ### Why PR-AUC > Accuracy
#
# With 2% fraud rate, a model that ALWAYS predicts "legit" gets 98% accuracy.
# PR-AUC measures how well the model **ranks** fraud above legitimate transactions,
# which is what matters when an analyst reviews the "top 500 suspicious" list.

# %%
import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_loader import load_elliptic_dataset, prepare_labels, get_timesteps
from src.utils.visualization import (
    plot_class_distribution, plot_degree_distribution,
    plot_temporal_fraud_ratio, plot_feature_correlation,
)

sns.set_theme(style="darkgrid")

# %% [markdown]
# ## Load the Dataset

# %%
data, dataset = load_elliptic_dataset()

print(f"\n{'='*50}")
print(f"  DATASET OVERVIEW")
print(f"{'='*50}")
print(f"  Nodes (transactions): {data.num_nodes:,}")
print(f"  Edges (payment flows): {data.num_edges:,}")
print(f"  Node features:        {data.num_node_features}")
print(f"  Directed graph:       Yes")
print(f"  Average degree:       {data.num_edges / data.num_nodes:.2f}")
print(f"{'='*50}")

# %% [markdown]
# ## Understanding Nodes, Edges, and Labels
#
# | Element | Meaning |
# |---------|---------|
# | **Node** | A single Bitcoin transaction |
# | **Edge** | A payment flow (transaction A funded transaction B) |
# | **Label** | Illicit (fraud), Licit (legitimate), or Unknown |
# | **Features** | 165 per node: 94 local + 71 aggregated neighbor features |
# | **Timestep** | One of 49 temporal snapshots (~2 weeks each) |

# %%
# Prepare labels
binary_labels, known_mask = prepare_labels(data)

# %% [markdown]
# ## Class Distribution (The Imbalance Problem)

# %%
fig = plot_class_distribution(binary_labels, known_mask)
plt.show()

# Show exact numbers
n_known = known_mask.sum().item()
n_unknown = (~known_mask).sum().item()
n_fraud = binary_labels[known_mask].sum().item()
n_legit = n_known - n_fraud

print(f"\nLabeled:   {n_known:>8,} ({100*n_known/data.num_nodes:.1f}%)")
print(f"Unknown:   {n_unknown:>8,} ({100*n_unknown/data.num_nodes:.1f}%)")
print(f"Fraud:     {int(n_fraud):>8,} ({100*n_fraud/n_known:.2f}% of labeled)")
print(f"Legit:     {int(n_legit):>8,} ({100*n_legit/n_known:.2f}% of labeled)")

# %% [markdown]
# ## Feature Analysis

# %%
x_np = data.x.cpu().numpy()

# Basic statistics
print(f"\nFeature matrix shape: {x_np.shape}")
print(f"Missing values (NaN): {np.isnan(x_np).sum()}")
print(f"Infinite values:      {np.isinf(x_np).sum()}")
print(f"\nFeature statistics (first 10 features):")
stats = pd.DataFrame({
    'mean': x_np[:, :10].mean(axis=0),
    'std':  x_np[:, :10].std(axis=0),
    'min':  x_np[:, :10].min(axis=0),
    'max':  x_np[:, :10].max(axis=0),
}, index=[f'F{i}' for i in range(10)])
print(stats.round(3).to_string())

# %% [markdown]
# ## Feature Correlation Heatmap

# %%
fig = plot_feature_correlation(x_np, top_n=20)
plt.show()

# %% [markdown]
# ## Graph Statistics & Degree Distribution

# %%
from torch_geometric.utils import degree

row = data.edge_index[0]
deg = degree(row, num_nodes=data.num_nodes).cpu().numpy()

print(f"\nGraph Statistics:")
print(f"  Average degree:  {deg.mean():.2f}")
print(f"  Median degree:   {np.median(deg):.0f}")
print(f"  Max degree:      {deg.max():.0f}")
print(f"  Isolated nodes:  {(deg == 0).sum():,}")
print(f"  Density:         {data.num_edges / (data.num_nodes * (data.num_nodes-1)):.8f}")

fig = plot_degree_distribution(data.edge_index, data.num_nodes)
plt.show()

# %% [markdown]
# ## Temporal Analysis — Fraud Ratio Over Time

# %%
timesteps = get_timesteps(data)
fig = plot_temporal_fraud_ratio(binary_labels, timesteps, known_mask)
plt.show()

# Per-timestep stats
ts_np = timesteps.cpu().numpy()
lab_np = binary_labels.cpu().numpy()
km_np = known_mask.cpu().numpy()

print(f"\n{'Timestep':>10s} {'Total':>8s} {'Labeled':>8s} {'Fraud':>8s} {'Ratio':>8s}")
print("-" * 50)
for t in sorted(set(ts_np)):
    mask_t = ts_np == t
    total = mask_t.sum()
    labeled = (mask_t & km_np).sum()
    fraud = lab_np[mask_t & km_np].sum()
    ratio = fraud / labeled if labeled > 0 else 0
    print(f"{int(t):>10d} {total:>8d} {labeled:>8d} {int(fraud):>8d} {ratio:>8.3f}")

# %% [markdown]
# ## Key EDA Takeaways
#
# 1. **Severe class imbalance**: ~2% illicit — must use weighted loss & PR-AUC
# 2. **77% unlabeled**: Keep in graph for message passing, exclude from loss
# 3. **No missing values**: Dataset is clean
# 4. **Low average degree**: Sparse graph — GNNs must extract maximum signal
# 5. **Temporal variation**: Fraud ratio fluctuates across timesteps — temporal split is essential
# 6. **Feature correlation**: Some features are highly correlated — GNN can learn to weight them

# %%
print("\n✅ EDA complete. Proceed to 02_baseline.py for baseline models.")
