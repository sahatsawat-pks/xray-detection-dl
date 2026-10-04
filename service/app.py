"""
app.py — FastAPI inference service for bone fracture detection.

Endpoints:
    GET  /health        — Liveness probe (process alive)
    GET  /ready         — Readiness probe (model loaded)
    POST /predict       — Single image prediction
    POST /predict/batch — Batch image predictions
    GET  /metrics       — Monitoring dashboard snapshot
    GET  /drift         — Latest drift check result
    GET  /slo           — SLO status

The model loads ONCE at startup via the lifespan handler.
"""

import io
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from monitoring.alerts import AlertManager
from monitoring.dashboard import MetricsCollector
from monitoring.drift import DriftMonitor
from service import model_loader
from service.schemas import (
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    PredictionResponse,
    ReadyResponse,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
logger = logging.getLogger(__name__)

# ── Server start time ─────────────────────────────────────────────────────────
_start_time: float = time.time()

# ── Monitoring singletons ─────────────────────────────────────────────────────
metrics_collector = MetricsCollector(window_seconds=300)
drift_monitor = DriftMonitor(window_size=200, check_interval=50)
alert_manager = AlertManager(cooldown_seconds=300)


# ── Lifespan: load model at startup ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup, clean up on shutdown."""
    global _start_time
    _start_time = time.time()
    try:
        model_loader.load_model(
            model_name="ModelFinal",
            version="local-dev",
        )
        logger.info("Model loaded successfully")
    except FileNotFoundError as e:
        logger.warning(f"Model checkpoint not found: {e}. /ready will return 503.")
    yield
    logger.info("Shutting down — model unloaded")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Bone Fracture X-Ray Detection API",
    description="AI-assisted triage: upload an X-ray → get fracture/no-fracture classification.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── GET /health — Liveness ────────────────────────────────────────────────────
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["probes"],
    summary="Liveness probe — is the process alive?",
)
async def health():
    """
    Always returns 200 if the process is running.
    Distinct from /ready: a healthy but unready service means the model
    is still loading. Kubernetes uses this to decide whether to kill the pod.
    """
    return HealthResponse(
        status="ok",
        uptime_seconds=round(time.time() - _start_time, 2),
    )


# ── GET /ready — Readiness ────────────────────────────────────────────────────
@app.get(
    "/ready",
    response_model=ReadyResponse,
    tags=["probes"],
    summary="Readiness probe — is the model loaded and ready?",
    responses={503: {"model": ErrorResponse}},
)
async def ready():
    """
    Returns 200 if the model is loaded and ready to serve predictions.
    Returns 503 if the model is not yet loaded.

    Distinct from /health: the process can be alive but not ready
    (e.g. model still downloading or loading into GPU memory).
    """
    if model_loader.is_ready():
        return ReadyResponse(
            status="ready",
            model_loaded=True,
            model_name=model_loader.get_model_name(),
            device=model_loader.get_device(),
        )
    return JSONResponse(
        status_code=503,
        content=ReadyResponse(
            status="not_ready",
            model_loaded=False,
            model_name="",
            device="",
        ).model_dump(),
    )


# ── POST /predict — Single image ─────────────────────────────────────────────
@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["inference"],
    summary="Predict fracture from a single X-ray image",
    responses={
        422: {"model": ErrorResponse, "description": "Invalid or corrupted image"},
        503: {"model": ErrorResponse, "description": "Model not ready"},
    },
)
async def predict(file: UploadFile = File(..., description="X-ray image (JPEG/PNG)")):
    """
    Upload a single X-ray image and receive a fracture/no-fracture prediction.

    The response includes:
    - **label**: Predicted class
    - **confidence**: How confident the model is (0–1)
    - **uncertain**: True if confidence < 0.60 (flagged for human review)
    - **model_version**: For lineage tracing
    - **inference_time_ms**: Wall-clock inference time
    """
    # Check model readiness
    if not model_loader.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded. Service is not ready.")

    # Read and validate image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
    except Exception as e:
        logger.warning(f"Failed to open image: {e}")
        raise HTTPException(
            status_code=422,
            detail=f"Cannot open image file: {e!s}",
        ) from e

    # Validate image quality
    is_valid, error_msg = model_loader.validate_image(image)
    if not is_valid:
        logger.warning(f"Image validation failed: {error_msg}")
        raise HTTPException(status_code=422, detail=error_msg)

    # Run inference
    try:
        result = model_loader.predict(image)
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {e!s}") from e

    # Log prediction (structured JSON for monitoring)
    logger.info(
        f"prediction: label={result['label']} confidence={result['confidence']:.4f} "
        f"uncertain={result['uncertain']} time_ms={result['inference_time_ms']:.1f}"
    )

    # ── Monitoring instrumentation ────────────────────────────────────────
    metrics_collector.record_request(
        latency_ms=result["inference_time_ms"],
        label=result["label"],
        confidence=result["confidence"],
        uncertain=result["uncertain"],
    )

    # Drift tracking
    predicted_label = 1 if result["label"] == "Fractured" else 0
    drift_result = drift_monitor.record(result["fractured_probability"], predicted_label)
    if drift_result:
        pred_drift = drift_result.get("prediction_drift", {})
        psi = pred_drift.get("psi", 0.0)
        severity = pred_drift.get("severity", "none")
        metrics_collector.update_drift(psi, severity)
        alert_manager.check_drift(psi)

    return PredictionResponse(**result)


# ── POST /predict/batch — Multiple images ────────────────────────────────────
@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["inference"],
    summary="Predict fractures for multiple X-ray images",
    responses={503: {"model": ErrorResponse}},
)
async def predict_batch(
    files: list[UploadFile] = File(..., description="X-ray images (JPEG/PNG)")
):
    """
    Upload multiple X-ray images for batch prediction.
    Each image is processed independently. Failed images are counted
    but do not prevent other predictions from completing.
    """
    if not model_loader.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded.")

    predictions = []
    failed = 0

    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))

            is_valid, error_msg = model_loader.validate_image(image)
            if not is_valid:
                failed += 1
                logger.warning(f"Batch: skipped {file.filename} — {error_msg}")
                continue

            result = model_loader.predict(image)
            predictions.append(PredictionResponse(**result))
        except Exception as e:
            failed += 1
            logger.warning(f"Batch: failed {file.filename} — {e}")

    return BatchPredictionResponse(
        predictions=predictions,
        total=len(files),
        failed=failed,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Monitoring Endpoints (R3)
# ═══════════════════════════════════════════════════════════════════════════════

# ── GET /metrics — Dashboard snapshot ─────────────────────────────────────────
@app.get(
    "/metrics",
    tags=["monitoring"],
    summary="Dashboard metrics snapshot (5 required metrics)",
)
async def get_metrics():
    """
    Returns the 5 required monitoring metrics:
    1. Request rate (predictions/min)
    2. Error rate by predicted class
    3. Latency percentiles (p50, p95, p99)
    4. PSI drift score
    5. Active model version
    """
    return metrics_collector.snapshot().to_dict()


# ── GET /drift — Latest drift check ──────────────────────────────────────────
@app.get(
    "/drift",
    tags=["monitoring"],
    summary="Latest prediction drift check result",
)
async def get_drift():
    """
    Returns the latest PSI drift check result, or forces a check
    if enough predictions have been recorded.
    """
    if drift_monitor.last_result:
        return drift_monitor.last_result
    return {"message": "No drift check has been triggered yet", "predictions_recorded": drift_monitor.count}


# ── GET /slo — SLO status ────────────────────────────────────────────────────
@app.get(
    "/slo",
    tags=["monitoring"],
    summary="Service Level Objective status",
)
async def get_slo():
    """
    Check current SLO status:
    - Availability >= 99%
    - p95 latency < 200ms
    - No critical drift
    """
    return metrics_collector.get_slo_status()


# ── GET /alerts — Alert history ───────────────────────────────────────────────
@app.get(
    "/alerts",
    tags=["monitoring"],
    summary="Active and recent alerts",
)
async def get_alerts():
    """Return active alerts and recent alert history."""
    return {
        "active": alert_manager.get_active_alerts(),
        "history": alert_manager.get_alert_history(),
    }
