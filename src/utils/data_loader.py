"""
=============================================================================
FraudGraph-AI — Data Loading & Preparation Pipeline
=============================================================================

MENTOR NOTE:
This module handles the entire data lifecycle for the Elliptic Bitcoin Dataset.

Key decisions:
1. TEMPORAL SPLIT — Split by timestep, NOT randomly. In real fraud detection,
   you train on past data and predict future transactions. Random split leaks
   future info into training → unrealistically high metrics.

2. UNKNOWN LABEL HANDLING — ~77% of nodes are unlabeled. We mask them out
   during loss computation but keep them in the graph for message passing.
   Removing unlabeled nodes would destroy graph structure.

3. CLASS WEIGHTS — With ~2% fraud, a "predict all legit" model gets 98%
   accuracy but catches zero fraud. Inverse-frequency weights penalize
   missing fraud ~50x more.

4. FEATURE SCALING — Financial features have wildly different scales.
   StandardScaler ensures stable GNN convergence.
=============================================================================
"""

import torch
import numpy as np
from torch_geometric.datasets import EllipticBitcoinDataset
from sklearn.preprocessing import StandardScaler
from src.utils.config import (
    DATA_DIR, DEVICE, RANDOM_SEED,
    TRAIN_TIMESTEPS, VAL_TIMESTEPS, TEST_TIMESTEPS,
)


def load_elliptic_dataset():
    """
    Load the Elliptic Bitcoin Dataset via PyTorch Geometric.

    Returns:
        data: PyG Data object with the full graph
        dataset: The dataset object for metadata
    """
    print("[INFO] Loading Elliptic Bitcoin Dataset...")
    dataset = EllipticBitcoinDataset(root=DATA_DIR)
    data = dataset[0]

    print(f"[INFO] Dataset loaded successfully!")
    print(f"  Nodes: {data.num_nodes:,}")
    print(f"  Edges: {data.num_edges:,}")
    print(f"  Features per node: {data.num_node_features}")
    print(f"  Device target: {DEVICE}")

    return data, dataset


def prepare_labels(data):
    """
    Prepare binary labels and identify known/unknown nodes.

    PyG EllipticBitcoinDataset encoding:
      0 = illicit → we map to 1 (fraud positive)
      1 = licit   → we map to 0 (legit negative)
      2 = unknown → masked out of loss

    Returns:
        binary_labels: Tensor of 0.0/1.0 for each node
        known_mask:    Boolean mask, True for labeled nodes
    """
    y = data.y.clone()
    unique_labels = torch.unique(y)
    print(f"[INFO] Unique raw labels: {unique_labels.tolist()}")

    # Detect encoding and normalise
    if 2 in unique_labels:
        known_mask = (y != 2)
        binary_labels = torch.zeros_like(y, dtype=torch.float)
        binary_labels[y == 0] = 1.0   # illicit → fraud
        binary_labels[y == 1] = 0.0   # licit  → legit
    elif -1 in unique_labels:
        known_mask = (y != -1)
        binary_labels = y.float().clamp(0, 1)
    else:
        known_mask = torch.ones(y.shape[0], dtype=torch.bool)
        binary_labels = y.float()

    n_known = known_mask.sum().item()
    n_fraud = binary_labels[known_mask].sum().item()
    n_legit = n_known - n_fraud

    print(f"[INFO] Labels prepared:")
    print(f"  Known: {n_known:,}/{y.shape[0]:,} ({100*n_known/y.shape[0]:.1f}%)")
    print(f"  Fraud: {int(n_fraud):,} ({100*n_fraud/n_known:.2f}%)")
    print(f"  Legit: {int(n_legit):,} ({100*n_legit/n_known:.2f}%)")

    return binary_labels, known_mask


def get_timesteps(data):
    """Extract per-node timestep information."""
    if hasattr(data, 'timestep'):
        return data.timestep
    if hasattr(data, 'time_step'):
        return data.time_step
    # Fallback: first feature column is timestep in Elliptic raw format
    return data.x[:, 0].long()


def create_temporal_masks(data, binary_labels, known_mask):
    """
    Create train/val/test masks using TEMPORAL splitting.

    WHY TEMPORAL? In production, you never train on future data.
    - Train: timesteps  1–34 (historical)
    - Val:   timesteps 35–42 (recent past for tuning)
    - Test:  timesteps 43–49 (future for evaluation)
    """
    timesteps = get_timesteps(data)
    print(f"[INFO] Timestep range: {timesteps.min().item()} – {timesteps.max().item()}")

    ts_train = set(TRAIN_TIMESTEPS)
    ts_val   = set(VAL_TIMESTEPS)
    ts_test  = set(TEST_TIMESTEPS)

    ts_np = timesteps.cpu().numpy()
    train_mask = torch.tensor([int(t) in ts_train for t in ts_np], dtype=torch.bool)
    val_mask   = torch.tensor([int(t) in ts_val   for t in ts_np], dtype=torch.bool)
    test_mask  = torch.tensor([int(t) in ts_test  for t in ts_np], dtype=torch.bool)

    # Intersect with known_mask: only labeled nodes contribute to loss
    train_mask = train_mask & known_mask
    val_mask   = val_mask   & known_mask
    test_mask  = test_mask  & known_mask

    print(f"[INFO] Temporal split:")
    for name, mask in [("Train", train_mask), ("Val", val_mask), ("Test", test_mask)]:
        n = mask.sum().item()
        n_f = binary_labels[mask].sum().item() if n > 0 else 0
        pct = 100 * n_f / n if n > 0 else 0
        print(f"  {name}: {n:,} nodes, fraud ratio {pct:.2f}%")

    return train_mask, val_mask, test_mask, timesteps


def compute_class_weights(binary_labels, train_mask):
    """
    Compute inverse-frequency class weight for BCE pos_weight.

    pos_weight = n_legit / n_fraud  (≈30-50x for Elliptic)
    This makes each fraud sample count as much as ~40 legit samples.
    """
    train_labels = binary_labels[train_mask]
    n_fraud = (train_labels == 1).sum().item()
    n_legit = (train_labels == 0).sum().item()

    pos_weight = torch.tensor(
        [n_legit / max(n_fraud, 1)], dtype=torch.float
    )
    print(f"[INFO] Class weights — pos_weight: {pos_weight.item():.2f} "
          f"(fraud={int(n_fraud)}, legit={int(n_legit)})")
    return pos_weight


def scale_features(data, train_mask):
    """Standardise features. Scaler fit on TRAIN ONLY to prevent leakage."""
    x_np = data.x.cpu().numpy()
    scaler = StandardScaler()
    scaler.fit(x_np[train_mask.cpu().numpy()])
    data.x = torch.tensor(scaler.transform(x_np), dtype=torch.float32)
    print("[INFO] Features scaled using training-set statistics.")
    return data, scaler


def prepare_data():
    """
    Master function — loads, labels, splits, scales, and returns everything.

    Returns dict with keys:
        data, binary_labels, known_mask, train_mask, val_mask, test_mask,
        timesteps, pos_weight, scaler, device
    """
    data, dataset = load_elliptic_dataset()
    binary_labels, known_mask = prepare_labels(data)
    train_mask, val_mask, test_mask, timesteps = create_temporal_masks(
        data, binary_labels, known_mask
    )
    data, scaler = scale_features(data, train_mask)
    pos_weight = compute_class_weights(binary_labels, train_mask)

    # Move tensors to device
    data = data.to(DEVICE)
    binary_labels = binary_labels.to(DEVICE)
    train_mask = train_mask.to(DEVICE)
    val_mask = val_mask.to(DEVICE)
    test_mask = test_mask.to(DEVICE)
    pos_weight = pos_weight.to(DEVICE)

    print(f"\n[OK] Data ready on {DEVICE}.")

    return {
        'data': data,
        'binary_labels': binary_labels,
        'known_mask': known_mask,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'timesteps': timesteps,
        'pos_weight': pos_weight,
        'scaler': scaler,
        'device': DEVICE,
    }
