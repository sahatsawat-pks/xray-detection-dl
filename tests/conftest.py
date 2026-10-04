"""
conftest.py — Shared pytest fixtures for bone fracture detection tests.

Provides reusable fixtures for models, datasets, sample images,
and device configuration. No cloud SDK imports.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Device ────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def device() -> str:
    """Return the best available device for testing."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Sample tensors ────────────────────────────────────────────────────────────
@pytest.fixture
def sample_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """A small batch of random image tensors and labels for testing."""
    images = torch.randn(4, 3, 224, 224)
    labels = torch.tensor([0, 1, 0, 1])
    return images, labels


@pytest.fixture
def single_image_tensor() -> torch.Tensor:
    """A single random image tensor matching input spec (1, 3, 224, 224)."""
    return torch.randn(1, 3, 224, 224)


# ── Sample PIL image ─────────────────────────────────────────────────────────
@pytest.fixture
def sample_pil_image(tmp_path) -> Path:
    """Create a temporary valid JPEG image and return its path."""
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    path = tmp_path / "test_xray.jpg"
    img.save(path, "JPEG")
    return path


@pytest.fixture
def corrupted_image(tmp_path) -> Path:
    """Create a truncated/corrupted JPEG file."""
    path = tmp_path / "corrupted.jpg"
    # Write partial JPEG header followed by garbage
    path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 50)
    return path


# ── Model fixtures ────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def model_baseline():
    """Instantiate ModelBaseline for testing."""
    from src.models import ModelBaseline
    return ModelBaseline(num_classes=2)


@pytest.fixture(scope="session")
def model_improved():
    """Instantiate ModelImproved for testing."""
    from src.models import ModelImproved
    return ModelImproved(num_classes=2, freeze_layers=6)


@pytest.fixture(scope="session")
def model_final():
    """Instantiate ModelFinal for testing."""
    from src.models import ModelFinal
    return ModelFinal(num_classes=2)


# ── Paths ─────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def data_csv(project_root) -> Path:
    return project_root / "dataset" / "dataset.csv"


@pytest.fixture(scope="session")
def image_root(project_root) -> Path:
    return project_root / "dataset" / "images"
