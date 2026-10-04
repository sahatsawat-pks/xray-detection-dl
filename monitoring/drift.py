"""
drift.py — Data and prediction drift detection.

Computes PSI (Population Stability Index) and Kolmogorov–Smirnov
statistics to detect distribution shifts in model predictions.

PSI thresholds (industry standard):
    PSI < 0.10  — no significant drift
    PSI 0.10–0.20 — moderate drift, investigate
    PSI >= 0.20 — significant drift, action required

For image classification, drift is detected on:
    1. Prediction confidence distribution (primary)
    2. Predicted class ratio (secondary)

No cloud SDK imports — uses only numpy and standard library.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
PSI_THRESHOLD_WARN = 0.10
PSI_THRESHOLD_ALERT = 0.20


def compute_psi(
    baseline: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
    eps: float = 1e-4,
) -> float:
    """
    Compute Population Stability Index between two distributions.

    PSI = Σ (P_i - Q_i) × ln(P_i / Q_i)
    where P = current distribution, Q = baseline distribution.

    Args:
        baseline: Reference distribution (e.g. validation set predictions).
        current: New distribution to compare.
        bins: Number of histogram bins.
        eps: Small constant to avoid division by zero.

    Returns:
        PSI value (float). >= 0.20 indicates significant drift.
    """
    # Create bins from baseline distribution
    bin_edges = np.linspace(0, 1, bins + 1)

    # Compute proportions in each bin
    baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
    current_counts, _ = np.histogram(current, bins=bin_edges)

    # Normalize to proportions
    baseline_prop = baseline_counts / (len(baseline) + eps) + eps
    current_prop = current_counts / (len(current) + eps) + eps

    # PSI formula
    psi = np.sum((current_prop - baseline_prop) * np.log(current_prop / baseline_prop))

    return float(psi)


def compute_ks_statistic(
    baseline: np.ndarray,
    current: np.ndarray,
) -> tuple[float, float]:
    """
    Compute the Kolmogorov–Smirnov statistic between two distributions.

    Returns:
        (ks_statistic, p_value) — large KS with small p-value indicates drift.
    """
    from scipy import stats
    ks_stat, p_value = stats.ks_2samp(baseline, current)
    return float(ks_stat), float(p_value)


def check_prediction_drift(
    baseline_probs: np.ndarray,
    current_probs: np.ndarray,
    bins: int = 10,
) -> dict:
    """
    Check for drift in prediction confidence distributions.

    Args:
        baseline_probs: Baseline fracture probabilities (from validation set).
        current_probs: Current prediction probabilities (from production).

    Returns:
        dict with: psi, ks_statistic, p_value, drifted, severity, timestamp.
    """
    psi = compute_psi(baseline_probs, current_probs, bins=bins)

    # KS test (optional — requires scipy)
    try:
        ks_stat, p_value = compute_ks_statistic(baseline_probs, current_probs)
    except ImportError:
        ks_stat, p_value = -1.0, -1.0

    # Severity classification
    if psi >= PSI_THRESHOLD_ALERT:
        severity = "critical"
        drifted = True
    elif psi >= PSI_THRESHOLD_WARN:
        severity = "warning"
        drifted = True
    else:
        severity = "none"
        drifted = False

    result = {
        "psi": round(psi, 6),
        "ks_statistic": round(ks_stat, 6),
        "p_value": round(p_value, 6),
        "drifted": drifted,
        "severity": severity,
        "threshold": PSI_THRESHOLD_ALERT,
        "baseline_size": len(baseline_probs),
        "current_size": len(current_probs),
        "baseline_mean": round(float(np.mean(baseline_probs)), 4),
        "current_mean": round(float(np.mean(current_probs)), 4),
        "timestamp": datetime.now(UTC).isoformat(),
    }

    if drifted:
        logger.warning(f"Prediction drift detected: PSI={psi:.4f} ({severity})")
    else:
        logger.info(f"No drift: PSI={psi:.4f}")

    return result


def check_class_ratio_drift(
    baseline_labels: np.ndarray,
    current_labels: np.ndarray,
    tolerance: float = 0.10,
) -> dict:
    """
    Check for drift in predicted class ratios.

    If the ratio of Fractured predictions shifts significantly from the
    baseline, something may have changed in the input distribution.

    Args:
        baseline_labels: Baseline predicted labels (0/1).
        current_labels: Current predicted labels (0/1).
        tolerance: Maximum acceptable deviation in positive rate.

    Returns:
        dict with: baseline_rate, current_rate, deviation, drifted.
    """
    baseline_rate = float(np.mean(baseline_labels))
    current_rate = float(np.mean(current_labels))
    deviation = abs(current_rate - baseline_rate)

    return {
        "baseline_positive_rate": round(baseline_rate, 4),
        "current_positive_rate": round(current_rate, 4),
        "deviation": round(deviation, 4),
        "tolerance": tolerance,
        "drifted": deviation > tolerance,
        "timestamp": datetime.now(UTC).isoformat(),
    }


# ── Baseline management ──────────────────────────────────────────────────────
class DriftMonitor:
    """
    Stateful drift monitor that accumulates predictions and periodically
    checks for drift against a stored baseline.
    """

    def __init__(
        self,
        baseline_path: str | None = None,
        window_size: int = 100,
        check_interval: int = 50,
    ):
        """
        Args:
            baseline_path: Path to JSON file with baseline probabilities.
            window_size: Number of recent predictions to keep in the window.
            check_interval: Run drift check every N predictions.
        """
        self.window_size = window_size
        self.check_interval = check_interval
        self.predictions: list[float] = []
        self.labels: list[int] = []
        self.count = 0
        self.last_result: dict | None = None

        # Load baseline
        self.baseline_probs: np.ndarray | None = None
        self.baseline_labels: np.ndarray | None = None
        if baseline_path and Path(baseline_path).exists():
            self._load_baseline(baseline_path)

    def _load_baseline(self, path: str) -> None:
        """Load baseline distribution from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        self.baseline_probs = np.array(data.get("probs", []))
        self.baseline_labels = np.array(data.get("labels", []))
        logger.info(
            f"Loaded baseline: {len(self.baseline_probs)} samples, "
            f"positive rate={np.mean(self.baseline_labels):.3f}"
        )

    def save_baseline(self, probs: np.ndarray, labels: np.ndarray, path: str) -> None:
        """Save current predictions as a new baseline."""
        data = {
            "probs": probs.tolist(),
            "labels": labels.tolist(),
            "created_at": datetime.now(UTC).isoformat(),
            "size": len(probs),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved baseline ({len(probs)} samples) to {path}")

    def record(self, fractured_probability: float, predicted_label: int) -> dict | None:
        """
        Record a new prediction. Returns drift result if check was triggered.

        Args:
            fractured_probability: Model output probability for fracture class.
            predicted_label: Predicted class (0 or 1).

        Returns:
            Drift result dict if a check was triggered, None otherwise.
        """
        self.predictions.append(fractured_probability)
        self.labels.append(predicted_label)
        self.count += 1

        # Trim to window size
        if len(self.predictions) > self.window_size:
            self.predictions = self.predictions[-self.window_size:]
            self.labels = self.labels[-self.window_size:]

        # Check at interval
        if self.count % self.check_interval == 0 and self.baseline_probs is not None:
            return self.check_now()

        return None

    def check_now(self) -> dict:
        """Force a drift check against the baseline."""
        if self.baseline_probs is None:
            return {"error": "No baseline loaded"}

        current_probs = np.array(self.predictions)
        current_labels = np.array(self.labels)

        pred_drift = check_prediction_drift(self.baseline_probs, current_probs)
        class_drift = check_class_ratio_drift(
            self.baseline_labels if self.baseline_labels is not None
            else (self.baseline_probs >= 0.5).astype(int),
            current_labels,
        )

        result = {
            "prediction_drift": pred_drift,
            "class_ratio_drift": class_drift,
            "window_size": len(self.predictions),
            "total_predictions": self.count,
        }

        self.last_result = result
        return result
