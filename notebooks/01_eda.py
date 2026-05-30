# %% [markdown]
# # FraudGraph-AI -- Step 1 & 3: Exploratory Data Analysis
#
# **Objective:** Understand the Elliptic Bitcoin Dataset before building models.
#
# ## Why This Matters (Business Context)
#
# Financial fraud costs the global economy **$5.4 trillion annually**. Traditional
# rule-based systems catch simple fraud (stolen cards, known patterns) but fail
# against **organized fraud rings** -- networks of accounts that launder money
# through chains of seemingly legitimate transactions.
#
# ### Why Traditional ML Fails on Fraud Rings
#
# A single fraudulent transaction may look identical to a legitimate one in terms
# of amount, time, and frequency. The fraud signal lives in the **GRAPH STRUCTURE**:
# - Money flowing through many intermediary accounts before cashing out
# - Clusters of accounts created around the same time
# - Circular transaction patterns (A->B->C->A)
#
# Traditional ML (Random Forest, XGBoost) sees each transaction **independently**.
# Graph Neural Networks see the **neighborhood** -- who sent money, who received
# it, and what THEIR transaction patterns look like.
#
# ### Why PR-AUC > Accuracy
#
# With 2% fraud rate (across ALL nodes), a model that ALWAYS predicts "legit"
# gets 80%+ accuracy. PR-AUC measures how well the model **ranks** fraud above
# legitimate transactions, which is what matters when an analyst reviews the
# "top 500 suspicious" list.

# %%
import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for script mode
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_loader import load_elliptic_dataset, prepare_labels, create_temporal_masks
from src.utils.visualization import (
    plot_class_distribution,
    plot_degree_distribution,
    plot_feature_correlation,
)
from src.utils.config import OUTPUTS_DIR

sns.set_theme(style="darkgrid")
print("[OK] Imports successful.")

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
# | **Label** | y=0 illicit (fraud), y=1 licit (legit), y=2 unknown |
# | **Features** | 165 per node: 94 local + 71 aggregated neighbor features |
# | **Train/Test** | PyG provides built-in temporal masks (earlier=train) |

# %%
binary_labels, known_mask = prepare_labels(data)
train_mask, val_mask, test_mask = create_temporal_masks(data, known_mask)

# %% [markdown]
# ## Class Distribution

# %%
fig = plot_class_distribution(binary_labels, known_mask)
plt.show()

n_known  = known_mask.sum().item()
n_unknown = (~known_mask).sum().item()
n_fraud  = binary_labels[known_mask].sum().item()
n_legit  = n_known - n_fraud

print(f"\nLabeled:   {n_known:>8,} ({100*n_known/data.num_nodes:.1f}%)")
print(f"Unknown:   {n_unknown:>8,} ({100*n_unknown/data.num_nodes:.1f}%)")
print(f"Fraud:     {int(n_fraud):>8,} ({100*n_fraud/n_known:.2f}% of labeled)")
print(f"Legit:     {int(n_legit):>8,} ({100*n_legit/n_known:.2f}% of labeled)")

# %% [markdown]
# ## Feature Analysis

# %%
x_np = data.x.cpu().numpy()

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
# ## Feature Correlation Heatmap (Top 20 by Variance)

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
# ## Temporal Split Summary
#
# PyG provides built-in temporal masks. The dataset spans 49 time steps,
# with earlier steps assigned to training and later steps to testing.

# %%
print(f"\nTemporal split summary:")
print(f"  Train nodes (labeled, earlier timesteps) : {train_mask.sum().item():,}")
print(f"  Val nodes   (labeled, tail of train)     : {val_mask.sum().item():,}")
print(f"  Test nodes  (labeled, later timesteps)   : {test_mask.sum().item():,}")
print(f"  Total labeled                            : {known_mask.sum().item():,}")

# Fraud ratio per split
for name, mask in [("Train", train_mask), ("Val", val_mask), ("Test", test_mask)]:
    n = mask.sum().item()
    if n > 0:
        nf = binary_labels[mask].sum().item()
        print(f"  {name} fraud ratio: {100*nf/n:.1f}%")

# %% [markdown]
# ## Key EDA Takeaways
#
# 1. **~77% unlabeled nodes**: Keep in graph for message passing, exclude from loss
# 2. **~90% illicit among labeled** (Elliptic dataset focuses on confirmed illicit clusters)
# 3. **~20% illicit across ALL nodes** (~42K/203K)
# 4. **No missing values**: Dataset is clean (already scaled by PyG)
# 5. **Low average degree (~1.15)**: Sparse graph
# 6. **PyG temporal masks**: Use directly for train/test split
# 7. **Weighted loss needed**: Large imbalance between fraud/legit in labeled set

# %%
print("\n[OK] EDA complete. Proceed to 02_baseline.py for baseline models.")
