"""
config.py — Single point of environment knowledge.

Every path, hyperparameter, and cloud setting is read here.
No other module reads environment variables directly.
"""

import os
from pathlib import Path

import torch
from dotenv import load_dotenv

# Load cloud.env if present (never committed)
_env_file = Path(__file__).resolve().parent.parent / "cloud.env"
if _env_file.exists():
    load_dotenv(_env_file)


# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CSV      = str(PROJECT_ROOT / "dataset" / "dataset.csv")
IMAGE_ROOT    = str(PROJECT_ROOT / "dataset" / "images")
MODEL_DIR     = str(PROJECT_ROOT / "models")
RESULTS_DIR   = str(PROJECT_ROOT / "results")


# ── Hyperparameters ──────────────────────────────────────────────────────────
SEED         = int(os.getenv("SEED", "42"))
BATCH_SIZE   = int(os.getenv("BATCH_SIZE", "32"))
NUM_EPOCHS   = int(os.getenv("NUM_EPOCHS", "30"))
PATIENCE     = int(os.getenv("PATIENCE", "8"))
IMG_SIZE     = int(os.getenv("IMG_SIZE", "224"))
NUM_WORKERS  = int(os.getenv("NUM_WORKERS", "0"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "1e-4"))

# Per-model learning rates
LEARNING_RATES = {
    "ModelBaseline": float(os.getenv("LR_BASELINE", "1e-3")),
    "ModelImproved": float(os.getenv("LR_IMPROVED", "3e-4")),
    "ModelFinal":    float(os.getenv("LR_FINAL",    "1e-4")),
}


# ── Cloud / MLflow ───────────────────────────────────────────────────────────
CLOUD_PROVIDER        = os.getenv("CLOUD_PROVIDER", "gcp")
PROJECT_ID            = os.getenv("PROJECT_ID", "")
REGION                = os.getenv("REGION", "asia-southeast1")
BLOB_URI              = os.getenv("BLOB_URI", "")
CONTAINER_REGISTRY    = os.getenv("CONTAINER_REGISTRY", "")
MLFLOW_TRACKING_URI   = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_REGISTRY_NAME   = os.getenv("MODEL_REGISTRY_NAME", "fracture-detector")
ENDPOINT_NAME         = os.getenv("ENDPOINT_NAME", "xray-fracture-predict")
METRICS_NAMESPACE     = os.getenv("METRICS_NAMESPACE", "itcs355")
BUDGET_LIMIT_THB      = int(os.getenv("BUDGET_LIMIT_THB", "800"))


# ── Device ───────────────────────────────────────────────────────────────────
def get_device() -> str:
    """Return the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Reproducibility ──────────────────────────────────────────────────────────
def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducibility."""
    import random

    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
