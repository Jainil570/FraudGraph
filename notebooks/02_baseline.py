# %% [markdown]
# # 🏋️ FraudGraph-AI — Step 5: Baseline Models
#
# **Why baselines?** Before investing in complex GNN architectures, we must
# establish how well simple feature-based models perform. If XGBoost matches
# the GNN, then the graph structure adds no value, and the GNN is unnecessary
# complexity.
#
# We train three models, each progressively more powerful:
# 1. **Logistic Regression** — Linear boundary. Can fraud be separated linearly?
# 2. **Random Forest** — Non-linear ensemble. Captures feature interactions.
# 3. **XGBoost** — Gradient boosting. State-of-the-art for tabular data.
#
# All use **class_weight='balanced'** to handle the 2% fraud imbalance.
# All use the **same temporal split** as GNNs for fair comparison.

# %%
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.train_baselines import run_baselines
from src.utils.metrics import format_comparison_table
from src.utils.visualization import (
    plot_roc_curves, plot_pr_curves, plot_confusion_matrix,
    plot_feature_importance,
)
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# ## Train All Baselines

# %%
results = run_baselines()

# %% [markdown]
# ## Model Comparison Table

# %%
metrics_dict = {name: res['metrics'] for name, res in results.items()}
print(format_comparison_table(metrics_dict))

# %% [markdown]
# ## ROC Curves

# %%
fig = plot_roc_curves(results)
plt.show()

# %% [markdown]
# ## Precision-Recall Curves
#
# This is the MORE IMPORTANT curve for fraud detection. ROC-AUC is inflated
# by true negatives (of which there are many). PR-AUC directly measures
# the precision-recall tradeoff that analysts care about.

# %%
fig = plot_pr_curves(results)
plt.show()

# %% [markdown]
# ## Confusion Matrices

# %%
for name, res in results.items():
    cm = res['metrics']['confusion_matrix']
    fig = plot_confusion_matrix(cm, name)
    plt.show()

# %% [markdown]
# ## XGBoost Feature Importance

# %%
xgb_model = results['XGBoost']['model']
importances = xgb_model.feature_importances_
fig = plot_feature_importance(importances, top_n=20, model_name="XGBoost")
plt.show()

# %% [markdown]
# ## Baseline Takeaways
#
# 1. XGBoost is the strongest baseline — but its PR-AUC is still limited
# 2. All baselines treat each transaction INDEPENDENTLY
# 3. They cannot capture: "this node's neighbors are mostly fraudulent"
# 4. GNNs should improve on these results by leveraging graph structure
#
# **Next:** `03_gcn_gat.py` — Build and train Graph Neural Networks

# %%
print("\n✅ Baselines complete. Proceed to 03_gcn_gat.py for GNN models.")
