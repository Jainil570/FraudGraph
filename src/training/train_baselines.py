"""
=============================================================================
FraudGraph-AI — Baseline Model Training Script
=============================================================================
Trains Logistic Regression, Random Forest, and XGBoost on node features
(ignoring graph structure) to establish non-graph baselines.
=============================================================================
"""

import numpy as np
from src.utils.data_loader import prepare_data
from src.utils.metrics import compute_all_metrics, print_metrics_report
from src.models.baselines import (
    get_baseline_models, train_baseline,
    predict_baseline, save_baseline,
)


def run_baselines():
    """Train all baseline models and return results."""

    # Load and prepare data
    bundle = prepare_data()
    data = bundle['data']
    labels = bundle['binary_labels']
    train_mask = bundle['train_mask']
    test_mask = bundle['test_mask']

    # Extract feature matrices (CPU numpy)
    X_train = data.x[train_mask].cpu().numpy()
    y_train = labels[train_mask].cpu().numpy().astype(int)
    X_test = data.x[test_mask].cpu().numpy()
    y_test = labels[test_mask].cpu().numpy().astype(int)

    print(f"\n[INFO] Baseline training data: {X_train.shape}")
    print(f"[INFO] Baseline test data:     {X_test.shape}")
    print(f"[INFO] Train fraud ratio:      "
          f"{y_train.mean()*100:.2f}%")
    print(f"[INFO] Test fraud ratio:       "
          f"{y_test.mean()*100:.2f}%\n")

    models = get_baseline_models()
    results = {}

    for name, model in models.items():
        # Train
        model = train_baseline(model, X_train, y_train, name)

        # Predict
        y_pred, y_prob = predict_baseline(model, X_test)

        # Evaluate
        metrics = compute_all_metrics(y_test, y_prob)
        print_metrics_report(metrics, name)

        # Save model
        save_baseline(model, name)

        results[name] = {
            'model': model,
            'metrics': metrics,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob,
        }

    return results


if __name__ == "__main__":
    results = run_baselines()

    # Print comparison
    print("\n" + "="*70)
    print("  BASELINE COMPARISON")
    print("="*70)
    header = f"{'Model':<25s} {'ROC-AUC':>8s} {'PR-AUC':>8s} " \
             f"{'Prec':>8s} {'Recall':>8s} {'F1':>8s}"
    print(header)
    print("-" * 70)
    for name, res in results.items():
        m = res['metrics']
        print(f"{name:<25s} {m['roc_auc']:>8.4f} {m['pr_auc']:>8.4f} "
              f"{m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f}")
