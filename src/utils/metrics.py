"""
=============================================================================
FraudGraph-AI — Evaluation Metrics
=============================================================================

MENTOR NOTE — Why These Metrics Matter for Fraud Detection:

• ACCURACY is misleading with 2% fraud. A "predict all legit" model = 98%.
• PRECISION = of flagged transactions, how many are real fraud?
  Low precision → analysts waste time on false alarms.
• RECALL = of all actual fraud, how many did we catch?
  Low recall → fraud slips through, financial loss.
• PR-AUC (Precision-Recall AUC) is THE key metric for imbalanced binary
  classification. Unlike ROC-AUC, it doesn't get inflated by true negatives.
• PRECISION@K = of top K riskiest transactions, how many are fraud?
  Directly maps to analyst workflow: "investigate the top 500".
=============================================================================
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve,
)


def precision_at_k(y_true, y_prob, k=500):
    """Of the top-K highest-probability predictions, how many are fraud?"""
    if len(y_prob) < k:
        k = len(y_prob)
    top_k_idx = np.argsort(y_prob)[::-1][:k]
    return float(np.mean(y_true[top_k_idx]))


def compute_all_metrics(y_true, y_prob, threshold=0.5, k=500):
    """
    Compute every fraud-relevant metric in one call.

    Args:
        y_true: Ground truth binary labels (numpy array)
        y_prob: Predicted probabilities (numpy array)
        threshold: Classification threshold (default 0.5)
        k: K for Precision@K

    Returns:
        dict with all metric values
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
        'f1':        f1_score(y_true, y_pred, zero_division=0),
    }

    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics['roc_auc'] = 0.0

    try:
        metrics['pr_auc'] = average_precision_score(y_true, y_prob)
    except ValueError:
        metrics['pr_auc'] = 0.0

    metrics[f'precision_at_{k}'] = precision_at_k(y_true, y_prob, k)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics['confusion_matrix'] = cm
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics.update({
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        })

    return metrics


def get_curves(y_true, y_prob):
    """Return ROC and PR curve data for plotting."""
    fpr, tpr, roc_th = roc_curve(y_true, y_prob)
    prec, rec, pr_th = precision_recall_curve(y_true, y_prob)
    return {
        'roc': {'fpr': fpr, 'tpr': tpr, 'thresholds': roc_th},
        'pr':  {'precision': prec, 'recall': rec, 'thresholds': pr_th},
    }


def print_metrics_report(metrics, model_name="Model"):
    """Pretty-print a metrics summary."""
    print(f"\n{'='*55}")
    print(f"  {model_name} — Evaluation Report")
    print(f"{'='*55}")
    for key in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']:
        if key in metrics:
            print(f"  {key:<20s}: {metrics[key]:.4f}")
    for key in metrics:
        if key.startswith('precision_at_'):
            print(f"  {key:<20s}: {metrics[key]:.4f}")
    if 'false_positives' in metrics:
        print(f"  {'false_positives':<20s}: {metrics['false_positives']}")
        print(f"  {'false_negatives':<20s}: {metrics['false_negatives']}")
        print(f"  {'fpr':<20s}: {metrics['fpr']:.4f}")
    print(f"{'='*55}\n")


def format_comparison_table(results_dict):
    """
    Format a comparison table from {model_name: metrics_dict}.

    Returns:
        str: Markdown-formatted table
    """
    header = "| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | FP |"
    sep    = "|-------|---------|--------|-----------|--------|----|----|"
    rows = [header, sep]
    for name, m in results_dict.items():
        row = (f"| {name} | {m.get('roc_auc',0):.4f} | {m.get('pr_auc',0):.4f} "
               f"| {m.get('precision',0):.4f} | {m.get('recall',0):.4f} "
               f"| {m.get('f1',0):.4f} | {m.get('false_positives','N/A')} |")
        rows.append(row)
    return "\n".join(rows)
