"""
=============================================================================
FraudGraph-AI -- Model Evaluator
=============================================================================
Loads all trained models, runs predictions on the test set, computes
metrics, and generates comparison reports & plots.
=============================================================================
"""

import os
import torch
import numpy as np
from src.utils.config import MODELS_DIR, DEVICE, GCN_PARAMS, GAT_PARAMS
from src.utils.metrics import (
    compute_all_metrics, print_metrics_report, format_comparison_table,
)
from src.utils.visualization import (
    plot_roc_curves, plot_pr_curves, plot_confusion_matrix,
)
from src.models.gcn import GCNFraudDetector
from src.models.gat import GATFraudDetector


def evaluate_gnn(model, data, binary_labels, test_mask, model_name="GNN"):
    """Evaluate a trained GNN model on the test set."""
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        y_prob = torch.sigmoid(logits[test_mask]).cpu().numpy()
        y_true = binary_labels[test_mask].cpu().numpy().astype(int)

    metrics = compute_all_metrics(y_true, y_prob)
    print_metrics_report(metrics, model_name)

    return {
        'metrics': metrics,
        'y_true': y_true,
        'y_prob': y_prob,
        'y_pred': (y_prob >= 0.5).astype(int),
    }


def load_gnn_model(model_class, params, model_name):
    """Load a saved GNN checkpoint."""
    path = os.path.join(
        MODELS_DIR,
        f"{model_name.lower().replace(' ', '_')}_best.pt"
    )
    if not os.path.exists(path):
        print(f"[WARN] Checkpoint not found: {path}")
        return None

    model = model_class(**params)
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    print(f"[LOAD] {model_name} loaded from {path}")
    return model


def run_full_evaluation(data, binary_labels, test_mask,
                        baseline_results=None):
    """
    Evaluate all models and produce comparison report.

    Args:
        data:             PyG Data on device
        binary_labels:    Labels on device
        test_mask:        Test mask on device
        baseline_results: Dict from train_baselines.run_baselines()

    Returns:
        all_results: Combined dict of all model results
    """
    all_results = {}

    # -- Baselines ----------------------------------------------
    if baseline_results is not None:
        for name, res in baseline_results.items():
            all_results[name] = res
            print(f"[EVAL] {name} -- from baseline cache")

    # -- GCN ----------------------------------------------------
    gcn_model = load_gnn_model(
        GCNFraudDetector,
        {
            'in_channels': GCN_PARAMS['in_channels'],
            'hidden_channels': GCN_PARAMS['hidden_channels'],
            'out_channels': GCN_PARAMS['out_channels'],
            'dropout': GCN_PARAMS['dropout'],
        },
        "GCN"
    )
    if gcn_model is not None:
        all_results['GCN'] = evaluate_gnn(
            gcn_model, data, binary_labels, test_mask, "GCN"
        )

    # -- GAT ----------------------------------------------------
    gat_model = load_gnn_model(
        GATFraudDetector,
        {
            'in_channels': GAT_PARAMS['in_channels'],
            'hidden_channels': GAT_PARAMS['hidden_channels'],
            'heads_1': GAT_PARAMS['heads_1'],
            'heads_2': GAT_PARAMS['heads_2'],
            'dropout': GAT_PARAMS['dropout'],
        },
        "GAT"
    )
    if gat_model is not None:
        all_results['GAT'] = evaluate_gnn(
            gat_model, data, binary_labels, test_mask, "GAT"
        )

    # -- Comparison ---------------------------------------------
    if all_results:
        metrics_dict = {k: v['metrics'] for k, v in all_results.items()}
        table = format_comparison_table(metrics_dict)
        print("\n" + table)

        # Plot curves
        plot_roc_curves(all_results)
        plot_pr_curves(all_results)

        # Plot confusion matrices for GNNs
        for name in ['GCN', 'GAT']:
            if name in all_results:
                cm = all_results[name]['metrics']['confusion_matrix']
                plot_confusion_matrix(cm, name)

    return all_results
