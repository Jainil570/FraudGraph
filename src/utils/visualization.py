"""
=============================================================================
FraudGraph-AI -- Visualization Utilities
=============================================================================
All plotting helpers for EDA, training curves, and evaluation reports.
Saves figures to outputs/ directory automatically.
=============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from src.utils.config import OUTPUTS_DIR

# Global style
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})


def _save(fig, name):
    path = os.path.join(OUTPUTS_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[PLOT] Saved -> {path}")
    plt.close(fig)


# -- EDA Plots ----------------------------------------------------------

def plot_class_distribution(labels, known_mask, save=True):
    """Bar chart of fraud vs legit among labeled nodes."""
    known = labels[known_mask]
    n_fraud = (known == 1).sum().item()
    n_legit = (known == 0).sum().item()

    fig, ax = plt.subplots()
    bars = ax.bar(['Licit', 'Illicit'], [n_legit, n_fraud],
                  color=['#2ecc71', '#e74c3c'], edgecolor='black')
    ax.set_title('Class Distribution (Labeled Nodes)')
    ax.set_ylabel('Count')
    for b in bars:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 200,
                f'{int(b.get_height()):,}', ha='center', fontweight='bold')
    if save:
        _save(fig, 'class_distribution')
    return fig


def plot_degree_distribution(edge_index, num_nodes, save=True):
    """Histogram of node degrees."""
    from torch_geometric.utils import degree
    import torch
    row = edge_index[0]
    deg = degree(row, num_nodes=num_nodes).cpu().numpy()

    fig, ax = plt.subplots()
    ax.hist(deg, bins=50, color='#3498db', edgecolor='black', alpha=0.8)
    ax.set_title('Node Degree Distribution')
    ax.set_xlabel('Degree')
    ax.set_ylabel('Frequency')
    ax.axvline(np.mean(deg), color='red', linestyle='--',
               label=f'Mean: {np.mean(deg):.1f}')
    ax.legend()
    if save:
        _save(fig, 'degree_distribution')
    return fig


def plot_temporal_fraud_ratio(labels, timesteps, known_mask, save=True):
    """Line chart: fraud ratio per timestep."""
    import torch
    ts_np = timesteps.cpu().numpy()
    lab_np = labels.cpu().numpy()
    km_np = known_mask.cpu().numpy()

    unique_ts = sorted(set(ts_np[km_np]))
    ratios = []
    counts = []
    for t in unique_ts:
        mask_t = (ts_np == t) & km_np
        n = mask_t.sum()
        n_f = lab_np[mask_t].sum()
        ratios.append(n_f / n if n > 0 else 0)
        counts.append(n)

    fig, ax1 = plt.subplots()
    color_r = '#e74c3c'
    color_c = '#3498db'
    ax1.plot(unique_ts, ratios, color=color_r, marker='o',
             markersize=4, linewidth=2, label='Fraud Ratio')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Fraud Ratio', color=color_r)
    ax1.tick_params(axis='y', labelcolor=color_r)
    ax1.set_title('Fraud Ratio & Transaction Volume Over Time')

    ax2 = ax1.twinx()
    ax2.bar(unique_ts, counts, alpha=0.25, color=color_c, label='Tx Count')
    ax2.set_ylabel('Transaction Count', color=color_c)
    ax2.tick_params(axis='y', labelcolor=color_c)

    fig.legend(loc='upper right', bbox_to_anchor=(0.88, 0.95))
    if save:
        _save(fig, 'temporal_fraud_ratio')
    return fig


def plot_feature_correlation(features_np, top_n=20, save=True):
    """Heatmap of top-N correlated features."""
    corr = np.corrcoef(features_np.T)
    # Pick top_n features by variance
    var = np.var(features_np, axis=0)
    top_idx = np.argsort(var)[::-1][:top_n]
    sub_corr = corr[np.ix_(top_idx, top_idx)]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(sub_corr, cmap='coolwarm', center=0, ax=ax,
                xticklabels=top_idx, yticklabels=top_idx)
    ax.set_title(f'Feature Correlation (Top {top_n} by Variance)')
    if save:
        _save(fig, 'feature_correlation')
    return fig


# -- Training Plots -----------------------------------------------------

def plot_training_history(history, model_name="Model", save=True):
    """Loss and metric curves over epochs."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0].set_title(f'{model_name} -- Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()

    if 'val_roc_auc' in history:
        axes[1].plot(history['val_roc_auc'], label='ROC-AUC',
                     color='#2ecc71', linewidth=2)
        axes[1].set_title(f'{model_name} -- ROC-AUC')
        axes[1].set_xlabel('Epoch')
        axes[1].legend()

    if 'val_pr_auc' in history:
        axes[2].plot(history['val_pr_auc'], label='PR-AUC',
                     color='#e74c3c', linewidth=2)
        axes[2].set_title(f'{model_name} -- PR-AUC')
        axes[2].set_xlabel('Epoch')
        axes[2].legend()

    fig.suptitle(f'{model_name} Training History', fontsize=16, y=1.02)
    fig.tight_layout()
    if save:
        _save(fig, f'training_history_{model_name.lower().replace(" ", "_")}')
    return fig


# -- Evaluation Plots ---------------------------------------------------

def plot_roc_curves(results_dict, save=True):
    """Overlay ROC curves for multiple models."""
    fig, ax = plt.subplots()
    for name, res in results_dict.items():
        fpr, tpr, _ = roc_curve(res['y_true'], res['y_prob'])
        auc = res['metrics'].get('roc_auc', 0)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve Comparison')
    ax.legend()
    if save:
        _save(fig, 'roc_curves')
    return fig


def plot_pr_curves(results_dict, save=True):
    """Overlay Precision-Recall curves for multiple models."""
    fig, ax = plt.subplots()
    for name, res in results_dict.items():
        prec, rec, _ = precision_recall_curve(res['y_true'], res['y_prob'])
        auc = res['metrics'].get('pr_auc', 0)
        ax.plot(rec, prec, linewidth=2, label=f"{name} (AP={auc:.3f})")
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve Comparison')
    ax.legend()
    if save:
        _save(fig, 'pr_curves')
    return fig


def plot_confusion_matrix(cm, model_name="Model", save=True):
    """Heatmap of confusion matrix."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Legit', 'Fraud'],
                yticklabels=['Legit', 'Fraud'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'{model_name} -- Confusion Matrix')
    if save:
        _save(fig, f'confusion_matrix_{model_name.lower().replace(" ", "_")}')
    return fig


def plot_feature_importance(importances, feature_names=None, top_n=20,
                            model_name="Model", save=True):
    """Horizontal bar chart of top feature importances."""
    if feature_names is None:
        feature_names = [f'F{i}' for i in range(len(importances))]
    idx = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(top_n), importances[idx][::-1],
            color='#3498db', edgecolor='black')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in idx][::-1])
    ax.set_xlabel('Importance')
    ax.set_title(f'{model_name} -- Top {top_n} Feature Importances')
    fig.tight_layout()
    if save:
        _save(fig, f'feature_importance_{model_name.lower().replace(" ", "_")}')
    return fig
