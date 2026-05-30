"""
FraudGraph-AI — API Request/Response Schemas (Pydantic v2)
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class PredictionRequest(BaseModel):
    """
    Request body for /predict endpoint.

    The client sends 165 node features representing a Bitcoin transaction.
    In production, these would come from a feature store or real-time
    feature pipeline.
    """
    features: List[float] = Field(
        ...,
        min_length=165,
        max_length=165,
        description="165-dimensional feature vector for the transaction node",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Classification threshold (lower = more sensitive)",
    )


class PredictionResponse(BaseModel):
    """Response from /predict endpoint."""
    fraud_probability: float = Field(
        ..., description="Probability that the transaction is fraudulent"
    )
    is_fraud: bool = Field(
        ..., description="Binary fraud prediction based on threshold"
    )
    risk_level: str = Field(
        ..., description="Risk category: LOW / MEDIUM / HIGH / CRITICAL"
    )
    threshold_used: float
    model_used: str


class HealthResponse(BaseModel):
    """Response from /health endpoint."""
    status: str = "healthy"
    model_loaded: bool
    device: str


class ModelInfoResponse(BaseModel):
    """Response from /model-info endpoint."""
    model_name: str
    architecture: str
    num_parameters: int
    training_pr_auc: Optional[float] = None
    device: str
