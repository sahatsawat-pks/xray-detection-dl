"""
test_service.py — Integration tests for the FastAPI inference service.

Tests all endpoints using FastAPI's TestClient. Verifies correct response
schemas, health/ready distinction, valid/invalid input handling, and batch
predictions. No cloud SDK imports.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Check if model checkpoint exists
MODEL_DIR = PROJECT_ROOT / "models"
HAS_MODEL = (MODEL_DIR / "ModelFinal_best_vloss.pth").exists()

skip_no_model = pytest.mark.skipif(
    not HAS_MODEL, reason="Model checkpoint not found (run `make train` first)"
)


@pytest.fixture(scope="module")
def client():
    """Create a FastAPI TestClient with the model loaded."""
    from fastapi.testclient import TestClient

    from service.app import app
    with TestClient(app) as c:
        yield c


def _make_jpeg_bytes(width=224, height=224, color=True) -> bytes:
    """Generate a valid JPEG image as bytes."""
    channels = 3 if color else 1
    mode = "RGB" if color else "L"
    arr = np.random.randint(0, 255, (height, width, channels) if color else (height, width),
                            dtype=np.uint8)
    img = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


def _make_blank_jpeg(width=224, height=224) -> bytes:
    """Generate an all-black JPEG image (degenerate input)."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


# ═══════════════════════════════════════════════════════════════════════════════
# Health & Readiness Probes
# ═══════════════════════════════════════════════════════════════════════════════
class TestHealthEndpoint:
    """GET /health — liveness probe."""

    def test_health_returns_200(self, client):
        """Health should always return 200 if the process is alive."""
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_schema(self, client):
        """Health response should have status and uptime."""
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "ok"
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0


class TestReadyEndpoint:
    """GET /ready — readiness probe (distinct from health)."""

    @skip_no_model
    def test_ready_returns_200_when_model_loaded(self, client):
        """Ready should return 200 when the model is in memory."""
        resp = client.get("/ready")
        assert resp.status_code == 200

    @skip_no_model
    def test_ready_response_schema(self, client):
        """Ready response should include model_loaded and model_name."""
        resp = client.get("/ready")
        data = resp.json()
        assert data["status"] == "ready"
        assert data["model_loaded"] is True
        assert data["model_name"] != ""
        assert data["device"] in ("cpu", "cuda", "mps")


# ═══════════════════════════════════════════════════════════════════════════════
# Single Prediction
# ═══════════════════════════════════════════════════════════════════════════════
@skip_no_model
class TestPredictEndpoint:
    """POST /predict — single image prediction."""

    def test_valid_image_returns_200(self, client):
        """A valid JPEG should return a 200 with prediction."""
        jpeg = _make_jpeg_bytes()
        resp = client.post("/predict", files={"file": ("test.jpg", jpeg, "image/jpeg")})
        assert resp.status_code == 200

    def test_response_schema(self, client):
        """Prediction response should match PredictionResponse schema."""
        jpeg = _make_jpeg_bytes()
        resp = client.post("/predict", files={"file": ("test.jpg", jpeg, "image/jpeg")})
        data = resp.json()
        assert "label" in data
        assert data["label"] in ("Fractured", "Non-Fractured")
        assert 0 <= data["confidence"] <= 1
        assert 0 <= data["fractured_probability"] <= 1
        assert "model_version" in data
        assert "model_name" in data
        assert "inference_time_ms" in data
        assert isinstance(data["uncertain"], bool)

    def test_corrupted_image_returns_422(self, client):
        """Corrupted/invalid file should return 422, not 500."""
        garbage = b"\xff\xd8\xff\xe0" + b"\x00" * 10  # Truncated JPEG
        resp = client.post("/predict", files={"file": ("bad.jpg", garbage, "image/jpeg")})
        assert resp.status_code == 422

    def test_blank_image_returns_422(self, client):
        """An all-black image should be rejected (degenerate input)."""
        blank = _make_blank_jpeg()
        resp = client.post("/predict", files={"file": ("blank.jpg", blank, "image/jpeg")})
        assert resp.status_code == 422

    def test_non_image_file_returns_422(self, client):
        """A text file uploaded as image should return 422."""
        text = b"This is not an image at all"
        resp = client.post("/predict", files={"file": ("text.txt", text, "text/plain")})
        assert resp.status_code == 422

    def test_no_file_returns_422(self, client):
        """Missing file should return 422."""
        resp = client.post("/predict")
        assert resp.status_code == 422

    def test_gradcam_field_in_prediction(self, client):
        """Prediction response should contain heatmap_base64 string or None."""
        jpeg = _make_jpeg_bytes()
        resp = client.post("/predict", files={"file": ("test.jpg", jpeg, "image/jpeg")})
        assert resp.status_code == 200
        data = resp.json()
        assert "heatmap_base64" in data
        if data["heatmap_base64"] is not None:
            assert isinstance(data["heatmap_base64"], str)
            assert len(data["heatmap_base64"]) > 100


# ═══════════════════════════════════════════════════════════════════════════════
# Batch Prediction
# ═══════════════════════════════════════════════════════════════════════════════
@skip_no_model
class TestBatchEndpoint:
    """POST /predict/batch — multiple image predictions."""

    def test_batch_returns_200(self, client):
        """Batch with valid images should return 200."""
        files = [
            ("files", ("img1.jpg", _make_jpeg_bytes(), "image/jpeg")),
            ("files", ("img2.jpg", _make_jpeg_bytes(), "image/jpeg")),
        ]
        resp = client.post("/predict/batch", files=files)
        assert resp.status_code == 200

    def test_batch_response_schema(self, client):
        """Batch response should have predictions list and counts."""
        files = [
            ("files", ("img1.jpg", _make_jpeg_bytes(), "image/jpeg")),
            ("files", ("img2.jpg", _make_jpeg_bytes(), "image/jpeg")),
        ]
        resp = client.post("/predict/batch", files=files)
        data = resp.json()
        assert "predictions" in data
        assert data["total"] == 2
        assert len(data["predictions"]) == 2

    def test_batch_partial_failure(self, client):
        """Batch should process valid images even if some fail."""
        files = [
            ("files", ("good.jpg", _make_jpeg_bytes(), "image/jpeg")),
            ("files", ("bad.jpg", b"not-an-image", "image/jpeg")),
        ]
        resp = client.post("/predict/batch", files=files)
        data = resp.json()
        assert data["total"] == 2
        assert data["failed"] == 1
        assert len(data["predictions"]) == 1
