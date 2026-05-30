"""
=============================================================================
FraudGraph-AI -- GNN Training Pipeline
=============================================================================

MENTOR NOTE -- Training Strategy:

1. WEIGHTED BCE LOSS: BCEWithLogitsLoss with pos_weight ≈ 40.
   This tells the optimizer: "missing one fraud costs as much as
   40 false alarms." Without this, the model converges to "predict
   all legit" within 2 epochs.

2. ADAM + WEIGHT DECAY: Adam is the go-to optimizer for GNNs.
   Weight decay (L2 regularisation) prevents overfitting on the
   small labeled set.

3. ReduceLROnPlateau: If val PR-AUC plateaus for 5 epochs, reduce
   learning rate by 50%. This fine-tunes the model in later stages.

4. EARLY STOPPING on val PR-AUC (not loss!). We care about ranking
   fraud correctly, not minimising cross-entropy.

5. MODEL CHECKPOINTING: Save the model with the best val PR-AUC.
   Training may overfit after the optimal point -- we keep the best.
=============================================================================
"""

import os
import copy
import torch
import torch.nn as nn
import numpy as np
from src.utils.metrics import compute_all_metrics
from src.utils.config import MODELS_DIR, DEVICE


def train_gnn(model, data, binary_labels, train_mask, val_mask,
              pos_weight, lr=0.001, weight_decay=1e-4,
              epochs=100, patience=15, model_name="GNN"):
    """
    Full training loop for a GNN model.

    Args:
        model:         GCN or GAT nn.Module
        data:          PyG Data object (on device)
        binary_labels: Tensor of 0/1 labels (on device)
        train_mask:    Boolean mask for training nodes
        val_mask:      Boolean mask for validation nodes
        pos_weight:    Class weight tensor for BCE loss
        lr, weight_decay, epochs, patience: Hyperparameters
        model_name:    Name for logging/saving

    Returns:
        model:   Best model (by val PR-AUC)
        history: Dict of per-epoch metrics
    """
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # History tracking
    history = {
        'train_loss': [], 'val_loss': [],
        'val_roc_auc': [], 'val_pr_auc': [], 'val_f1': [],
        'lr': [],
    }

    best_val_pr_auc = 0.0
    best_model_state = None
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"  Training {model_name}")
    print(f"  LR={lr}, WD={weight_decay}, Epochs={epochs}, "
          f"Patience={patience}")
    print(f"  pos_weight={pos_weight.item():.2f}, Device={DEVICE}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        # -- Train ----------------------------------------------
        model.train()
        optimizer.zero_grad()

        logits = model(data.x, data.edge_index)
        loss = criterion(logits[train_mask], binary_labels[train_mask])
        loss.backward()
        optimizer.step()

        train_loss = loss.item()

        # -- Validate -------------------------------------------
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            val_loss = criterion(
                logits[val_mask], binary_labels[val_mask]
            ).item()

            # Compute metrics on validation set
            y_true = binary_labels[val_mask].cpu().numpy()
            y_prob = torch.sigmoid(logits[val_mask]).cpu().numpy()
            val_metrics = compute_all_metrics(y_true, y_prob)

        val_pr_auc = val_metrics['pr_auc']
        val_roc_auc = val_metrics['roc_auc']
        current_lr = optimizer.param_groups[0]['lr']

        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_roc_auc'].append(val_roc_auc)
        history['val_pr_auc'].append(val_pr_auc)
        history['val_f1'].append(val_metrics['f1'])
        history['lr'].append(current_lr)

        # Scheduler step
        scheduler.step(val_pr_auc)

        # Early stopping check
        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            marker = " * best"
        else:
            patience_counter += 1
            marker = ""

        if epoch % 5 == 0 or epoch == 1 or marker:
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                  f"ROC: {val_roc_auc:.4f} | "
                  f"PR: {val_pr_auc:.4f} | "
                  f"LR: {current_lr:.6f}{marker}")

        if patience_counter >= patience:
            print(f"\n  [STOP] Early stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs)")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Save checkpoint
    save_path = os.path.join(
        MODELS_DIR, f"{model_name.lower().replace(' ', '_')}_best.pt"
    )
    torch.save({
        'model_state_dict': model.state_dict(),
        'best_val_pr_auc': best_val_pr_auc,
        'history': history,
    }, save_path)
    print(f"\n  [SAVE] Best model -> {save_path}")
    print(f"  [BEST] Val PR-AUC = {best_val_pr_auc:.4f}")

    return model, history
