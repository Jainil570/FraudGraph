"""
=============================================================================
FraudGraph-AI -- MLflow Experiment Tracking
=============================================================================

MENTOR NOTE:
MLflow tracks EVERYTHING about each experiment run:
- Parameters (learning rate, architecture, etc.)
- Metrics (loss, ROC-AUC, PR-AUC per epoch)
- Artifacts (saved model files, plots)

This lets you compare runs: "Did 4-head GAT beat 2-head GAT?"
Without tracking, you lose this information forever.

We use a LOCAL file-based backend (mlruns/ directory).
No server needed -- just run `mlflow ui` to see the dashboard.
=============================================================================
"""

import os
import mlflow
import mlflow.pytorch
from src.utils.config import MLRUNS_DIR

# Set tracking URI to local directory
mlflow.set_tracking_uri(f"file:///{MLRUNS_DIR.replace(os.sep, '/')}")


def setup_experiment(experiment_name="FraudGraph-AI"):
    """Create or get an MLflow experiment."""
    mlflow.set_experiment(experiment_name)
    print(f"[MLflow] Experiment: {experiment_name}")
    print(f"[MLflow] Tracking URI: {mlflow.get_tracking_uri()}")


def log_training_run(model_name, params, metrics, history=None,
                     model_path=None, tags=None):
    """
    Log a complete training run to MLflow.

    Args:
        model_name: Name for the run
        params:     Dict of hyperparameters
        metrics:    Dict of final evaluation metrics
        history:    Optional training history dict
        model_path: Optional path to saved model checkpoint
        tags:       Optional dict of tags
    """
    with mlflow.start_run(run_name=model_name):
        # Log parameters
        for key, val in params.items():
            mlflow.log_param(key, val)

        # Log final metrics
        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                mlflow.log_metric(key, val)

        # Log training history as step-metrics
        if history:
            for key in ['train_loss', 'val_loss', 'val_roc_auc',
                         'val_pr_auc', 'val_f1']:
                if key in history:
                    for step, val in enumerate(history[key]):
                        mlflow.log_metric(key, val, step=step)

        # Log model artifact
        if model_path and os.path.exists(model_path):
            mlflow.log_artifact(model_path)

        # Tags
        if tags:
            mlflow.set_tags(tags)

        mlflow.set_tag("model_type", model_name)
        print(f"[MLflow] Run '{model_name}' logged successfully.")


def log_baseline_run(model_name, params, metrics):
    """Convenience wrapper for baseline model logging."""
    log_training_run(
        model_name=model_name,
        params=params,
        metrics=metrics,
        tags={"category": "baseline"},
    )


def log_gnn_run(model_name, params, metrics, history, model_path):
    """Convenience wrapper for GNN model logging."""
    log_training_run(
        model_name=model_name,
        params=params,
        metrics=metrics,
        history=history,
        model_path=model_path,
        tags={"category": "gnn"},
    )
