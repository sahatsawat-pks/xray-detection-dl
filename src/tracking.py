"""
tracking.py — MLflow tracking utilities.

Provides consistent experiment tracking, lineage tagging,
and model registration for every training run.
No cloud SDK imports — uses only mlflow and standard library.
"""

import hashlib
import os
import subprocess
from pathlib import Path

import mlflow


def get_git_sha() -> str:
    """Return the current git commit SHA, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_data_fingerprint(data_dir: str | None = None) -> str:
    """Return a DVC hash of the dataset, or a manual hash if DVC is not set up."""
    if data_dir is None:
        data_dir = str(Path(__file__).resolve().parent.parent / "dataset" / "images")

    # Try DVC first
    dvc_file = data_dir + ".dvc"
    if os.path.exists(dvc_file):
        with open(dvc_file) as f:
            for line in f:
                if "md5:" in line or "- md5:" in line:
                    return line.split(":")[-1].strip()

    # Fallback: hash the directory listing (not file contents — too slow)
    file_list = sorted(str(p) for p in Path(data_dir).rglob("*") if p.is_file())
    listing = "\n".join(file_list)
    return hashlib.md5(listing.encode()).hexdigest()[:12]


def get_image_digest() -> str:
    """Return the Docker image digest, if available."""
    return os.environ.get("IMAGE_DIGEST", "not-built")


def setup_mlflow(experiment_name: str = "fracture-detection") -> None:
    """Configure MLflow tracking URI and experiment."""
    from src.config import MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)


def log_lineage_tags(model_name: str, seed: int) -> None:
    """Log standard lineage tags to the active MLflow run."""
    mlflow.set_tags({
        "git_sha": get_git_sha(),
        "dvc_data_hash": get_data_fingerprint(),
        "image_digest": get_image_digest(),
        "model_name": model_name,
        "seed": str(seed),
    })


def log_training_params(model_name: str, config: dict) -> None:
    """Log hyperparameters to the active MLflow run."""
    mlflow.log_params({
        "model_name": model_name,
        "batch_size": config.get("batch_size"),
        "num_epochs": config.get("num_epochs"),
        "learning_rate": config.get("learning_rate"),
        "weight_decay": config.get("weight_decay"),
        "patience": config.get("patience"),
        "img_size": config.get("img_size"),
        "seed": config.get("seed"),
        "optimizer": config.get("optimizer", "AdamW"),
        "scheduler": config.get("scheduler", "CosineAnnealingLR"),
    })


def log_epoch_metrics(epoch: int, train_loss: float, val_loss: float,
                      train_perf: dict, val_perf: dict) -> None:
    """Log per-epoch metrics to the active MLflow run."""
    mlflow.log_metrics({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_accuracy": train_perf["accuracy"],
        "val_accuracy": val_perf["accuracy"],
        "train_f1": train_perf["f1"],
        "val_f1": val_perf["f1"],
        "val_precision": val_perf["precision"],
        "val_recall": val_perf["recall"],
        "val_auc": val_perf.get("auc", 0.0),
    }, step=epoch)


def log_test_metrics(test_perf: dict) -> None:
    """Log final test set metrics to the active MLflow run."""
    mlflow.log_metrics({
        "test_accuracy": test_perf["accuracy"],
        "test_f1": test_perf["f1"],
        "test_precision": test_perf["precision"],
        "test_recall": test_perf["recall"],
        "test_auc": test_perf.get("auc", 0.0),
    })


def log_model_artifact(checkpoint_path: str, plots_dir: str | None = None) -> None:
    """Log the trained model checkpoint and plots as MLflow artifacts."""
    if os.path.exists(checkpoint_path):
        mlflow.log_artifact(checkpoint_path, artifact_path="model")
    if plots_dir and os.path.isdir(plots_dir):
        mlflow.log_artifacts(plots_dir, artifact_path="plots")


def register_best_model(model_name: str, run_id: str) -> str:
    """Register the best model from a run to the MLflow Model Registry."""
    from src.config import MODEL_REGISTRY_NAME
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, MODEL_REGISTRY_NAME)
    print(f"[tracking] Registered {model_name} as {MODEL_REGISTRY_NAME} "
          f"version {result.version}")
    return str(result.version)
