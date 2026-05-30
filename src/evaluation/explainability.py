"""
=============================================================================
FraudGraph-AI — Explainable AI Module
=============================================================================

MENTOR NOTE — Why Explainability Matters in Fintech:

In regulated industries (banking, insurance), you CANNOT deploy a model
that says "this is fraud" without explaining WHY. Regulators (GDPR Art.22,
US ECOA) require that automated decisions be explainable.

Our approach:
1. SHAP TreeExplainer — For XGBoost baseline (exact, fast)
2. SHAP KernelExplainer — For GNN node embeddings (surrogate approach)
3. GAT Attention Weights — Which neighbors influenced the prediction?

The surrogate approach works as follows:
- Extract GNN node embeddings (64-dim vectors)
- Train a simple model (logistic regression) on embeddings → labels
- Run SHAP on THAT model
- Map SHAP values back to original features via the GNN weight matrices
=============================================================================
"""

import os
import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
from src.utils.config import OUTPUTS_DIR, DEVICE


def explain_xgboost(model, X_test, feature_names=None, max_display=20):
    """
    SHAP TreeExplainer for XGBoost.
    Fast and exact for tree-based models.
    """
    print("[SHAP] Running TreeExplainer on XGBoost...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    if feature_names is None:
        feature_names = [f'F{i}' for i in range(X_test.shape[1])]

    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_test,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    path = os.path.join(OUTPUTS_DIR, 'shap_xgboost_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SHAP] Summary plot → {path}")

    # Global feature importance from SHAP
    importance = np.abs(shap_values).mean(axis=0)

    return shap_values, importance


def explain_gnn_embeddings(gnn_model, data, test_mask, binary_labels,
                           feature_names=None, n_samples=200):
    """
    Surrogate SHAP explanation for GNN predictions.

    Steps:
    1. Extract GNN embeddings for test nodes
    2. Train a lightweight surrogate model on embeddings → labels
    3. Run SHAP KernelExplainer on the surrogate
    """
    from sklearn.linear_model import LogisticRegression

    print("[SHAP] Extracting GNN embeddings...")
    gnn_model.eval()
    with torch.no_grad():
        embeddings = gnn_model.get_embeddings(
            data.x, data.edge_index
        )
    emb_test = embeddings[test_mask].cpu().numpy()
    y_test = binary_labels[test_mask].cpu().numpy().astype(int)

    # Train surrogate
    print("[SHAP] Training surrogate model on embeddings...")
    surrogate = LogisticRegression(max_iter=1000, class_weight='balanced')
    surrogate.fit(emb_test, y_test)

    # SHAP on surrogate (sample for speed)
    n = min(n_samples, len(emb_test))
    bg = shap.sample(emb_test, min(50, n))
    explainer = shap.KernelExplainer(surrogate.predict_proba, bg)

    print(f"[SHAP] Computing SHAP values for {n} samples...")
    shap_values = explainer.shap_values(emb_test[:n])

    # Save summary
    fig, ax = plt.subplots(figsize=(10, 8))
    emb_names = [f'Emb_{i}' for i in range(emb_test.shape[1])]
    shap.summary_plot(
        shap_values[1] if isinstance(shap_values, list) else shap_values,
        emb_test[:n],
        feature_names=emb_names,
        max_display=20,
        show=False,
    )
    path = os.path.join(OUTPUTS_DIR, 'shap_gnn_embeddings.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SHAP] GNN embedding SHAP → {path}")

    return shap_values


def visualize_attention_weights(gat_model, data, test_mask,
                                binary_labels, top_n=10):
    """
    Visualise GAT attention weights for the most-fraudulent test nodes.

    Shows which neighbors the model "attended to" when making each
    fraud prediction — directly interpretable by analysts.
    """
    gat_model.eval()
    with torch.no_grad():
        logits, (attn1, attn2) = gat_model(
            data.x, data.edge_index, return_attention=True
        )
        probs = torch.sigmoid(logits)

    # Find top-N most-fraudulent test nodes
    test_indices = torch.where(test_mask)[0]
    test_probs = probs[test_indices]
    top_fraud_idx = test_indices[torch.argsort(test_probs, descending=True)[:top_n]]

    edge_idx_attn, attn_weights = attn1
    attn_weights = attn_weights.cpu().numpy()
    edge_np = edge_idx_attn.cpu().numpy()

    results = []
    for node_idx in top_fraud_idx:
        nid = node_idx.item()
        # Find edges where this node is the target
        incoming = edge_np[1] == nid
        if incoming.sum() == 0:
            continue

        neighbor_ids = edge_np[0][incoming]
        neighbor_attns = attn_weights[incoming].mean(axis=-1)  # avg over heads

        actual = int(binary_labels[nid].item())
        pred_prob = probs[nid].item()

        results.append({
            'node_id': nid,
            'predicted_prob': pred_prob,
            'actual_label': actual,
            'top_neighbors': list(zip(
                neighbor_ids.tolist(),
                neighbor_attns.tolist()
            )),
        })

    # Plot attention distribution for top fraud node
    if results:
        top = results[0]
        neighbors, attns = zip(*sorted(
            top['top_neighbors'], key=lambda x: -x[1]
        )[:15])

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(range(len(attns)), attns, color='#e74c3c', edgecolor='black')
        ax.set_yticks(range(len(attns)))
        ax.set_yticklabels([f'Node {n}' for n in neighbors])
        ax.set_xlabel('Attention Weight')
        ax.set_title(f"GAT Attention — Node {top['node_id']} "
                     f"(P(fraud)={top['predicted_prob']:.3f})")
        path = os.path.join(OUTPUTS_DIR, 'gat_attention_weights.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[ATTN] Attention plot → {path}")

    return results
