"""
=============================================================================
FraudGraph-AI — Baseline Models (Non-Graph)
=============================================================================

MENTOR NOTE — Why Baselines Matter:

Before building a complex GNN, you MUST establish baselines. If a simple
Random Forest matches your GNN, the graph structure isn't helping — and
you've wasted engineering effort on unnecessary complexity.

Baselines answer: "Does the graph structure actually add value?"

We use three progressively stronger baselines:
1. Logistic Regression — linear decision boundary (sanity check)
2. Random Forest       — ensemble of trees, handles non-linearity
3. XGBoost             — gradient boosting, typically strongest non-graph model

All use CLASS WEIGHTING to handle the 2% fraud imbalance.
All use the SAME temporal split as GNNs for fair comparison.
=============================================================================
"""

import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from src.utils.config import MODELS_DIR, RANDOM_SEED, RF_PARAMS, XGB_PARAMS


def get_baseline_models():
    """
    Return a dict of baseline models with tuned hyperparameters.

    All models use class_weight='balanced' or scale_pos_weight
    to handle the severe imbalance.
    """
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=RANDOM_SEED,
            solver='lbfgs',
            C=1.0,
        ),
        'Random Forest': RandomForestClassifier(**RF_PARAMS),
        'XGBoost': XGBClassifier(
            **XGB_PARAMS,
            use_label_encoder=False,
            # scale_pos_weight is set dynamically in train_baselines
        ),
    }
    return models


def train_baseline(model, X_train, y_train, model_name="model"):
    """Train a single baseline model."""
    print(f"[TRAIN] Training {model_name}...")

    # For XGBoost, set scale_pos_weight dynamically
    if isinstance(model, XGBClassifier):
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        model.set_params(scale_pos_weight=n_neg / max(n_pos, 1))

    model.fit(X_train, y_train)
    print(f"[TRAIN] {model_name} training complete.")
    return model


def predict_baseline(model, X):
    """Get predictions and probabilities from a baseline model."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return y_pred, y_prob


def save_baseline(model, model_name):
    """Save model to disk."""
    fname = model_name.lower().replace(' ', '_') + '.joblib'
    path = os.path.join(MODELS_DIR, fname)
    joblib.dump(model, path)
    print(f"[SAVE] {model_name} → {path}")
    return path


def load_baseline(model_name):
    """Load model from disk."""
    fname = model_name.lower().replace(' ', '_') + '.joblib'
    path = os.path.join(MODELS_DIR, fname)
    return joblib.load(path)
