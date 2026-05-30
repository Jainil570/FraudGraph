"""
=============================================================================
FraudGraph-AI — FastAPI Prediction Service
=============================================================================

MENTOR NOTE — API Design for ML Models:

Production ML APIs follow these principles:
1. STATELESS: Each request contains everything needed for prediction.
   No session state — enables horizontal scaling.
2. MODEL AT STARTUP: Load the model once when the server starts,
   not per-request. Model loading takes seconds; inference takes ms.
3. INPUT VALIDATION: Pydantic enforces that features are exactly
   165-dimensional floats. Bad input → 422 error, not a crash.
4. RISK LEVELS: Raw probabilities are useless to analysts.
   Map to LOW/MEDIUM/HIGH/CRITICAL for actionable decisions.

Run with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
=============================================================================
"""

import os
import torch
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    PredictionRequest, PredictionResponse,
    HealthResponse, ModelInfoResponse,
)

# ── Globals ────────────────────────────────────────────────────────────

MODEL = None
MODEL_NAME = "GAT"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_PR_AUC = None


def _load_model():
    """Load the best GAT model at startup. Falls back to GCN."""
    global MODEL, MODEL_NAME, BEST_PR_AUC

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base, "models")

    # Try GAT first, then GCN
    for name, cls_name in [("gat", "GAT"), ("gcn", "GCN")]:
        path = os.path.join(models_dir, f"{name}_best.pt")
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=DEVICE,
                                    weights_only=False)

            if name == "gat":
                from src.models.gat import GATFraudDetector
                from src.utils.config import GAT_PARAMS as P
                model = GATFraudDetector(
                    in_channels=P['in_channels'],
                    hidden_channels=P['hidden_channels'],
                    heads_1=P['heads_1'],
                    heads_2=P['heads_2'],
                    dropout=P['dropout'],
                )
            else:
                from src.models.gcn import GCNFraudDetector
                from src.utils.config import GCN_PARAMS as P
                model = GCNFraudDetector(
                    in_channels=P['in_channels'],
                    hidden_channels=P['hidden_channels'],
                    out_channels=P['out_channels'],
                    dropout=P['dropout'],
                )

            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(DEVICE)
            model.eval()

            MODEL = model
            MODEL_NAME = cls_name
            BEST_PR_AUC = checkpoint.get('best_val_pr_auc')
            print(f"[API] Loaded {cls_name} model from {path}")
            return

    print("[API] WARNING: No trained model found. /predict will fail.")


# ── Lifespan ───────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


# ── App ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="FraudGraph-AI",
    description="Graph Neural Network-based Bitcoin fraud detection API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _risk_level(prob: float) -> str:
    if prob >= 0.85:
        return "CRITICAL"
    if prob >= 0.60:
        return "HIGH"
    if prob >= 0.30:
        return "MEDIUM"
    return "LOW"


# ── Endpoints ──────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=MODEL is not None,
        device=str(DEVICE),
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    n_params = sum(p.numel() for p in MODEL.parameters()) if MODEL else 0
    arch = "GATFraudDetector" if MODEL_NAME == "GAT" else "GCNFraudDetector"
    return ModelInfoResponse(
        model_name=MODEL_NAME,
        architecture=arch,
        num_parameters=n_params,
        training_pr_auc=BEST_PR_AUC,
        device=str(DEVICE),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict fraud probability for a single transaction.

    NOTE: In a real system, you would also need the transaction's
    neighborhood (edges) for proper GNN inference. This endpoint
    uses a simplified approach where we create a single-node graph
    and pass it through the model's classifier layer directly
    (using pre-computed embeddings in production).
    """
    if MODEL is None:
        return PredictionResponse(
            fraud_probability=0.0,
            is_fraud=False,
            risk_level="UNKNOWN",
            threshold_used=request.threshold,
            model_used="none",
        )

    features = torch.tensor(
        [request.features], dtype=torch.float32
    ).to(DEVICE)

    # For single-node inference without graph structure,
    # we pass through the classifier with a self-loop edge
    with torch.no_grad():
        edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(DEVICE)
        logit = MODEL(features, edge_index)
        prob = torch.sigmoid(logit).item()

    is_fraud = prob >= request.threshold

    return PredictionResponse(
        fraud_probability=round(prob, 6),
        is_fraud=is_fraud,
        risk_level=_risk_level(prob),
        threshold_used=request.threshold,
        model_used=MODEL_NAME,
    )
