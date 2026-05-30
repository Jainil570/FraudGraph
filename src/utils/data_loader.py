"""
=============================================================================
FraudGraph-AI -- Data Loading & Preparation Pipeline
=============================================================================

MENTOR NOTE:
This module handles the entire data lifecycle for the Elliptic Bitcoin Dataset.

Key decisions:
1. TEMPORAL SPLIT -- PyG's EllipticBitcoinDataset already provides built-in
   train_mask and test_mask that respect temporal ordering (earlier timesteps
   train, later timesteps test). We use these directly + carve a val set
   from the train split for hyperparameter tuning.

   WHY NOT RANDOM SPLIT? In real fraud detection, you train on past data
   and predict future transactions. Random split leaks future info into
   training, producing unrealistically high metrics.

2. LABEL ENCODING -- PyG encodes labels as:
     y=0  ->  raw class=1 (illicit)  ->  binary label 1 (FRAUD)
     y=1  ->  raw class=2 (licit)    ->  binary label 0 (LEGIT)
     y=2  ->  unknown                ->  masked out of loss
   ~77% of nodes are unlabeled. They stay in the graph for message passing
   but are excluded from loss computation.

3. CLASS WEIGHTS -- Among labeled nodes: ~90% illicit, ~10% licit.
   Among ALL nodes: ~20% illicit. Both indicate the need for weighted loss.

4. FEATURE SCALING -- PyG's version already applies scaling. We re-apply
   StandardScaler fit-only on train nodes as a best-practice insurance.
=============================================================================
"""

import torch
import numpy as np
from torch_geometric.datasets import EllipticBitcoinDataset
from sklearn.preprocessing import StandardScaler
from src.utils.config import DATA_DIR, DEVICE


def load_elliptic_dataset():
    """
    Load the Elliptic Bitcoin Dataset via PyTorch Geometric.

    Returns:
        data:    PyG Data object (single graph)
        dataset: The dataset wrapper object
    """
    print("[INFO] Loading Elliptic Bitcoin Dataset...")
    dataset = EllipticBitcoinDataset(root=DATA_DIR)
    data = dataset[0]

    print(f"[INFO] Dataset loaded successfully!")
    print(f"  Nodes          : {data.num_nodes:,}")
    print(f"  Edges          : {data.num_edges:,}")
    print(f"  Node features  : {data.num_node_features}")
    print(f"  Device target  : {DEVICE}")
    return data, dataset


def prepare_labels(data):
    """
    Prepare binary fraud labels and known-node mask.

    PyG EllipticBitcoinDataset label encoding:
      y=0 -> illicit (raw class 1) -> binary 1 (FRAUD)
      y=1 -> licit   (raw class 2) -> binary 0 (LEGIT)
      y=2 -> unknown               -> excluded from loss

    Returns:
        binary_labels : FloatTensor [N]  -- 1=fraud, 0=legit
        known_mask    : BoolTensor  [N]  -- True for labeled nodes
    """
    y = data.y.clone()
    print(f"[INFO] Label values in dataset: {torch.unique(y).tolist()}")
    print(f"[INFO]   y=0 -> illicit (FRAUD), y=1 -> licit (LEGIT), y=2 -> unknown")

    known_mask    = (y != 2)
    binary_labels = torch.zeros(y.shape[0], dtype=torch.float)
    binary_labels[y == 0] = 1.0   # illicit -> fraud
    binary_labels[y == 1] = 0.0   # licit   -> legit

    n_all   = y.shape[0]
    n_known = known_mask.sum().item()
    n_fraud = binary_labels[known_mask].sum().item()
    n_legit = n_known - n_fraud

    print(f"[INFO] Label breakdown:")
    print(f"  Total nodes      : {n_all:,}")
    print(f"  Labeled          : {n_known:,} ({100*n_known/n_all:.1f}%)")
    print(f"  Unknown          : {n_all-n_known:,} ({100*(n_all-n_known)/n_all:.1f}%)")
    print(f"  Illicit [FRAUD]  : {int(n_fraud):,} ({100*n_fraud/n_known:.1f}% of labeled)")
    print(f"  Licit   [LEGIT]  : {int(n_legit):,} ({100*n_legit/n_known:.1f}% of labeled)")
    return binary_labels, known_mask


def create_temporal_masks(data, known_mask, val_fraction=0.15):
    """
    Create train / val / test masks using PyG's built-in temporal split.

    PyG provides:
        data.train_mask -- labeled nodes from earlier timesteps (~29K)
        data.test_mask  -- labeled nodes from later timesteps  (~16K)

    We further carve a validation set from the TAIL of the train split
    (not random, to preserve temporal ordering).

    Args:
        data:          PyG Data object
        known_mask:    Boolean mask of all labeled nodes
        val_fraction:  Fraction of train nodes to use for validation

    Returns:
        train_mask, val_mask, test_mask  -- all BoolTensor [N]
    """
    # PyG masks (already temporal and intersection with known labels)
    pyg_train = data.train_mask.clone()
    test_mask  = data.test_mask.clone()

    # Carve validation from the END of training mask
    train_indices = torch.where(pyg_train)[0]
    n_val = int(len(train_indices) * val_fraction)
    n_train = len(train_indices) - n_val

    val_indices   = train_indices[n_train:]   # last n_val (temporally later)
    train_indices = train_indices[:n_train]   # first n_train (earlier)

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask   = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[train_indices] = True
    val_mask[val_indices]     = True

    print(f"[INFO] Temporal split (using PyG built-in masks):")
    for name, mask in [("Train", train_mask), ("Val", val_mask), ("Test", test_mask)]:
        n = mask.sum().item()
        n_f = 0
        if n > 0:
            try:
                binary_labels_cpu = torch.zeros(data.num_nodes)
                binary_labels_cpu[data.y == 0] = 1.0
                n_f = binary_labels_cpu[mask].sum().item()
            except Exception:
                pass
        print(f"  {name:<6}: {n:>6,} nodes")

    return train_mask, val_mask, test_mask


def compute_class_weights(binary_labels, train_mask):
    """
    Compute pos_weight for BCEWithLogitsLoss.

    pos_weight = n_legit / n_fraud  in training set.
    This penalises missing a fraud sample proportionally.
    """
    train_labels = binary_labels[train_mask]
    n_fraud = (train_labels == 1).sum().item()
    n_legit = (train_labels == 0).sum().item()
    pos_weight = torch.tensor([n_legit / max(n_fraud, 1)], dtype=torch.float)
    print(f"[INFO] pos_weight = {pos_weight.item():.3f}  "
          f"(fraud={int(n_fraud):,}, legit={int(n_legit):,} in train set)")
    return pos_weight


def scale_features(data, train_mask):
    """
    StandardScaler fitted ONLY on training nodes to prevent leakage.
    Returns scaled data and the fitted scaler.
    """
    x_np = data.x.cpu().numpy()
    scaler = StandardScaler()
    scaler.fit(x_np[train_mask.cpu().numpy()])
    data.x = torch.tensor(scaler.transform(x_np), dtype=torch.float32)
    print("[INFO] Features re-scaled using training-set statistics.")
    return data, scaler


def prepare_data():
    """
    Master function -- loads, labels, splits, scales.

    Returns dict with:
        data, binary_labels, known_mask,
        train_mask, val_mask, test_mask,
        pos_weight, scaler, device
    """
    data, dataset = load_elliptic_dataset()
    binary_labels, known_mask = prepare_labels(data)
    train_mask, val_mask, test_mask = create_temporal_masks(data, known_mask)
    data, scaler = scale_features(data, train_mask)
    pos_weight = compute_class_weights(binary_labels, train_mask)

    # Move to device
    data          = data.to(DEVICE)
    binary_labels = binary_labels.to(DEVICE)
    train_mask    = train_mask.to(DEVICE)
    val_mask      = val_mask.to(DEVICE)
    test_mask     = test_mask.to(DEVICE)
    pos_weight    = pos_weight.to(DEVICE)

    print(f"\n[OK] Data ready on {DEVICE}.")
    return {
        'data': data,
        'binary_labels': binary_labels,
        'known_mask': known_mask,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'pos_weight': pos_weight,
        'scaler': scaler,
        'device': DEVICE,
    }
