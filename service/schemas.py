"""
schemas.py — Pydantic request/response models for the inference API.

Strict validation on all inputs and outputs. Every response includes
model_version for lineage tracing back to the MLflow run.
"""

from pydantic import BaseModel, Field


# ── Prediction ────────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    """Response for a single prediction."""
    label: str = Field(..., description="Predicted class: 'Fractured' or 'Non-Fractured'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence (0–1)")
    fractured_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of fracture"
    )
    model_version: str = Field(..., description="Model identifier (MLflow run ID or git SHA)")
    model_name: str = Field(..., description="Architecture name (e.g. ModelFinal)")
    inference_time_ms: float = Field(..., ge=0.0, description="Wall-clock inference time in ms")
    uncertain: bool = Field(
        False, description="True if confidence is below threshold (flagged for review)"
    )


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    predictions: list[PredictionResponse]
    total: int
    failed: int = Field(0, description="Number of images that failed to process")


# ── Health / Readiness ────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    """Liveness probe: is the process alive?"""
    status: str = Field("ok", description="Always 'ok' if the process is running")
    uptime_seconds: float = Field(..., description="Seconds since server start")


class ReadyResponse(BaseModel):
    """Readiness probe: is the model loaded and ready to serve?"""
    status: str = Field(..., description="'ready' or 'not_ready'")
    model_loaded: bool = Field(..., description="Whether the model is in memory")
    model_name: str = Field("", description="Name of the loaded model architecture")
    device: str = Field("", description="Inference device (cpu/cuda/mps)")


# ── Error ─────────────────────────────────────────────────────────────────────
class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: str = ""
