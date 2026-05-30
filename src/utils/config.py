"""
=============================================================================
FraudGraph-AI — Central Configuration
=============================================================================
All paths, hyperparameters, and constants in one place.
=============================================================================
"""

import os
import torch

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")

for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Device ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Reproducibility ───────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Temporal Split ─────────────────────────────────────────────────────
TRAIN_TIMESTEPS = list(range(1, 35))   # 1–34
VAL_TIMESTEPS   = list(range(35, 43))  # 35–42
TEST_TIMESTEPS  = list(range(43, 50))  # 43–49

# ── Labels ─────────────────────────────────────────────────────────────
LABEL_ILLICIT = 1
LABEL_LICIT = 2

# ── GCN Hyperparameters ───────────────────────────────────────────────
GCN_PARAMS = {
    "in_channels": 165,
    "hidden_channels": 128,
    "out_channels": 64,
    "dropout": 0.3,
    "lr": 0.001,
    "weight_decay": 1e-4,
    "epochs": 100,
    "patience": 15,
}

# ── GAT Hyperparameters ───────────────────────────────────────────────
GAT_PARAMS = {
    "in_channels": 165,
    "hidden_channels": 32,
    "heads_1": 4,
    "heads_2": 2,
    "dropout": 0.3,
    "lr": 0.0005,
    "weight_decay": 1e-4,
    "epochs": 100,
    "patience": 15,
}

# ── XGBoost ────────────────────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "eval_metric": "logloss",
}

# ── Random Forest ──────────────────────────────────────────────────────
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": RANDOM_SEED,
    "class_weight": "balanced",
    "n_jobs": -1,
}
