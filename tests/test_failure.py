"""
test_failure.py — Engineered failure tests (R4 rubric).

Planned failure mode: Corrupted Image Injection
Simulates a faulty PACS feed sending bad images to the prediction service.

These tests demonstrate:
    1. WITHOUT protection — model silently predicts garbage (dangerous)
    2. WITH protection — service rejects/flags bad inputs (safe)

Each test class maps to one attack vector from the failure engineering plan.
"""

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

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


def _make_jpeg_bytes(arr: np.ndarray, mode: str = "RGB") -> bytes:
    """Convert numpy array to JPEG bytes."""
    img = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


def _make_valid_xray(width=224, height=224) -> bytes:
    """Generate a realistic-looking grayscale X-ray image (random)."""
    arr = np.random.randint(30, 220, (height, width, 3), dtype=np.uint8)
    return _make_jpeg_bytes(arr)


# ═══════════════════════════════════════════════════════════════════════════════
# Attack Vector 1: Truncated JPEG
# Simulates: PACS network timeout cutting off the image transfer
# ═══════════════════════════════════════════════════════════════════════════════
@skip_no_model
class TestTruncatedJPEG:
    """Truncated JPEG files — simulates interrupted PACS transfer."""

    def test_truncated_header_only(self, client):
        """JPEG with only SOI marker should be rejected."""
        # Just the JPEG Start-Of-Image marker
        payload = b"\xff\xd8"
        resp = client.post("/predict", files={"file": ("trunc.jpg", payload, "image/jpeg")})
        assert resp.status_code == 422, "Truncated JPEG should be rejected"

    def test_truncated_partial_header(self, client):
        """Partial JPEG header (first 100 bytes) should be rejected."""
        payload = b"\xff\xd8\xff\xe0" + b"\x00" * 96
        resp = client.post("/predict", files={"file": ("partial.jpg", payload, "image/jpeg")})
        assert resp.status_code == 422

    def test_truncated_half_image(self, client):
        """A real JPEG cut in half should be rejected or flagged."""
        # Generate a valid JPEG, then truncate it
        valid = _make_valid_xray()
        truncated = valid[:len(valid) // 2]
        resp = client.post("/predict", files={"file": ("half.jpg", truncated, "image/jpeg")})
        # Should either reject (422) or the image loader may still handle it
        # (PIL has LOAD_TRUNCATED_IMAGES=True in dataset.py, but service validates)
        assert resp.status_code in (200, 422), f"Unexpected status: {resp.status_code}"


# ═══════════════════════════════════════════════════════════════════════════════
# Attack Vector 2: Degenerate Images
# Simulates: Malfunctioning X-ray detector producing blank frames
# ═══════════════════════════════════════════════════════════════════════════════
@skip_no_model
class TestDegenerateImages:
    """Blank, solid-color, and degenerate images."""

    def test_all_black(self, client):
        """All-black image (dead detector) should be rejected."""
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        jpeg = _make_jpeg_bytes(arr)
        resp = client.post("/predict", files={"file": ("black.jpg", jpeg, "image/jpeg")})
        assert resp.status_code == 422, "All-black image should be rejected (low entropy)"

    def test_all_white(self, client):
        """All-white image (overexposed) should be rejected."""
        arr = np.ones((224, 224, 3), dtype=np.uint8) * 255
        jpeg = _make_jpeg_bytes(arr)
        resp = client.post("/predict", files={"file": ("white.jpg", jpeg, "image/jpeg")})
        assert resp.status_code == 422, "All-white image should be rejected (low entropy)"

    def test_single_pixel(self, client):
        """1×1 pixel image should be rejected (too small)."""
        arr = np.array([[[128, 128, 128]]], dtype=np.uint8)
        jpeg = _make_jpeg_bytes(arr)
        resp = client.post("/predict", files={"file": ("tiny.jpg", jpeg, "image/jpeg")})
        assert resp.status_code == 422, "1×1 image should be rejected"

    def test_solid_gray(self, client):
        """Solid gray image should be rejected (low entropy)."""
        arr = np.ones((224, 224, 3), dtype=np.uint8) * 128
        jpeg = _make_jpeg_bytes(arr)
        resp = client.post("/predict", files={"file": ("gray.jpg", jpeg, "image/jpeg")})
        assert resp.status_code == 422, "Solid gray should be rejected"


# ═══════════════════════════════════════════════════════════════════════════════
# Attack Vector 3: Non-Medical Images
# Simulates: Document photo accidentally uploaded via PACS interface
# ═══════════════════════════════════════════════════════════════════════════════
@skip_no_model
class TestNonMedicalImages:
    """Non-X-ray images fed to the model — should still produce output
    but the confidence thresholding should flag as 'uncertain'."""

    def test_random_noise(self, client):
        """Pure random noise should not be rejected but may flag uncertain."""
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        jpeg = _make_jpeg_bytes(arr)
        resp = client.post("/predict", files={"file": ("noise.jpg", jpeg, "image/jpeg")})
        # Random noise has high entropy, so it passes validation
        # But model confidence should be unpredictable
        assert resp.status_code == 200
        data = resp.json()
        # The prediction should exist regardless
        assert data["label"] in ("Fractured", "Non-Fractured")
        assert "uncertain" in data

    def test_gradient_image(self, client):
        """A smooth gradient image should produce a prediction (passes entropy)."""
        arr = np.tile(np.linspace(0, 255, 224, dtype=np.uint8), (224, 3, 1)).T
        arr = arr.reshape(224, 224, 3)
        jpeg = _make_jpeg_bytes(arr)
        resp = client.post("/predict", files={"file": ("gradient.jpg", jpeg, "image/jpeg")})
        assert resp.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# Attack Vector 4: Non-Image Files
# Simulates: Corrupted file system sending wrong file types
# ═══════════════════════════════════════════════════════════════════════════════
@skip_no_model
class TestNonImageFiles:
    """Files that aren't images at all."""

    def test_text_file(self, client):
        """Plain text should be rejected."""
        payload = b"Patient: John Doe\nDiagnosis: Pending\n"
        resp = client.post("/predict", files={"file": ("report.txt", payload, "text/plain")})
        assert resp.status_code == 422

    def test_pdf_header(self, client):
        """PDF file should be rejected."""
        payload = b"%PDF-1.4 fake pdf content here"
        resp = client.post("/predict", files={"file": ("report.pdf", payload, "application/pdf")})
        assert resp.status_code == 422

    def test_empty_file(self, client):
        """Empty file should be rejected."""
        resp = client.post("/predict", files={"file": ("empty.jpg", b"", "image/jpeg")})
        assert resp.status_code == 422

    def test_binary_garbage(self, client):
        """Random binary data should be rejected."""
        payload = np.random.bytes(1024)
        resp = client.post("/predict", files={"file": ("garbage.bin", payload, "application/octet-stream")})
        assert resp.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════════
# Attack Vector 5: Confidence Thresholding
# Simulates: Model encountering out-of-distribution inputs
# ═══════════════════════════════════════════════════════════════════════════════
@skip_no_model
class TestConfidenceThresholding:
    """Verify that low-confidence predictions are flagged as 'uncertain'."""

    def test_response_has_uncertain_field(self, client):
        """Every prediction should include the 'uncertain' boolean."""
        jpeg = _make_valid_xray()
        resp = client.post("/predict", files={"file": ("test.jpg", jpeg, "image/jpeg")})
        assert resp.status_code == 200
        data = resp.json()
        assert "uncertain" in data
        assert isinstance(data["uncertain"], bool)

    def test_response_has_fractured_probability(self, client):
        """Every prediction should include raw fracture probability."""
        jpeg = _make_valid_xray()
        resp = client.post("/predict", files={"file": ("test.jpg", jpeg, "image/jpeg")})
        data = resp.json()
        assert "fractured_probability" in data
        assert 0.0 <= data["fractured_probability"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# Attack Vector 6: Batch Failure Resilience
# Simulates: Mixed batch from PACS with some corrupted transfers
# ═══════════════════════════════════════════════════════════════════════════════
@skip_no_model
class TestBatchResilience:
    """Batch endpoint should handle mixed valid/invalid inputs gracefully."""

    def test_mixed_batch(self, client):
        """Batch with valid + invalid images should process valid ones."""
        valid = _make_valid_xray()
        black = _make_jpeg_bytes(np.zeros((224, 224, 3), dtype=np.uint8))
        garbage = b"not an image"

        files = [
            ("files", ("valid.jpg", valid, "image/jpeg")),
            ("files", ("black.jpg", black, "image/jpeg")),
            ("files", ("garbage.bin", garbage, "image/jpeg")),
        ]
        resp = client.post("/predict/batch", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3
        assert data["failed"] >= 2, "Black and garbage should fail"
        # At least the valid image should succeed
        assert len(data["predictions"]) >= 1

    def test_all_invalid_batch(self, client):
        """Batch with all invalid images should return 200 with 0 predictions."""
        files = [
            ("files", ("bad1.jpg", b"garbage1", "image/jpeg")),
            ("files", ("bad2.jpg", b"garbage2", "image/jpeg")),
        ]
        resp = client.post("/predict/batch", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert data["failed"] == 2
        assert len(data["predictions"]) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Meta: Verify Protection Summary
# ═══════════════════════════════════════════════════════════════════════════════
@skip_no_model
class TestProtectionSummary:
    """Verify that the full protection chain is wired correctly."""

    def test_valid_input_succeeds(self, client):
        """A valid X-ray image should produce a successful prediction."""
        jpeg = _make_valid_xray()
        resp = client.post("/predict", files={"file": ("xray.jpg", jpeg, "image/jpeg")})
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] in ("Fractured", "Non-Fractured")
        assert data["confidence"] > 0

    def test_monitoring_endpoints_available(self, client):
        """Monitoring endpoints should be accessible for demo."""
        for endpoint in ["/metrics", "/drift", "/slo", "/alerts"]:
            resp = client.get(endpoint)
            assert resp.status_code == 200, f"{endpoint} should return 200"
