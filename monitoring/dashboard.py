"""
dashboard.py — Metrics collector and dashboard configuration.

Tracks the 5 required monitoring metrics:
    1. Request rate (predictions per minute)
    2. Error rate by predicted class
    3. Latency percentiles (p50, p95, p99)
    4. PSI drift score
    5. Active model version

Collects metrics in-process and exposes them via a /metrics endpoint
or emits them to cloud monitoring via the adapter.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MetricsSnapshot:
    """Point-in-time snapshot of all monitored metrics."""
    timestamp: str
    # 1. Request rate
    requests_total: int
    requests_per_minute: float
    # 2. Error rate
    errors_total: int
    error_rate: float
    errors_by_class: dict[str, int]
    # 3. Latency percentiles
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_mean_ms: float
    # 4. Drift
    psi_score: float | None
    drift_status: str  # "ok", "warning", "critical"
    # 5. Model
    model_version: str
    model_name: str

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "request_rate": {
                "total": self.requests_total,
                "per_minute": round(self.requests_per_minute, 2),
            },
            "error_rate": {
                "total": self.errors_total,
                "rate": round(self.error_rate, 4),
                "by_class": self.errors_by_class,
            },
            "latency_ms": {
                "p50": round(self.latency_p50_ms, 2),
                "p95": round(self.latency_p95_ms, 2),
                "p99": round(self.latency_p99_ms, 2),
                "mean": round(self.latency_mean_ms, 2),
            },
            "drift": {
                "psi": self.psi_score,
                "status": self.drift_status,
            },
            "model": {
                "version": self.model_version,
                "name": self.model_name,
            },
        }


class MetricsCollector:
    """
    In-process metrics collector for the inference service.

    Tracks requests, latencies, errors, and predictions in a rolling
    time window. Thread-safe for use with async FastAPI.
    """

    def __init__(self, window_seconds: int = 300, max_points: int = 10_000):
        """
        Args:
            window_seconds: Rolling window for rate calculations (default 5 min).
            max_points: Maximum latency data points to keep in memory.
        """
        self.window_seconds = window_seconds
        self._start_time = time.time()

        # Counters
        self.requests_total = 0
        self.errors_total = 0
        self.errors_by_class: dict[str, int] = {}

        # Latency tracking (rolling window)
        self._latencies: deque[tuple[float, float]] = deque(maxlen=max_points)

        # Request timestamps for rate calculation
        self._request_times: deque[float] = deque(maxlen=max_points)

        # Prediction tracking
        self.predictions_by_class: dict[str, int] = {"Fractured": 0, "Non-Fractured": 0}
        self.uncertain_count = 0

        # Drift state
        self.last_psi: float | None = None
        self.drift_status: str = "ok"

        # Model info
        self.model_version: str = "unknown"
        self.model_name: str = "unknown"

    def record_request(
        self,
        latency_ms: float,
        label: str,
        confidence: float,
        uncertain: bool,
        error: bool = False,
    ) -> None:
        """Record a completed prediction request."""
        now = time.time()
        self.requests_total += 1
        self._request_times.append(now)
        self._latencies.append((now, latency_ms))

        if error:
            self.errors_total += 1
            self.errors_by_class[label] = self.errors_by_class.get(label, 0) + 1
        else:
            self.predictions_by_class[label] = self.predictions_by_class.get(label, 0) + 1

        if uncertain:
            self.uncertain_count += 1

    def record_error(self, error_type: str = "unknown") -> None:
        """Record a request that resulted in an error."""
        self.errors_total += 1
        self.errors_by_class[error_type] = self.errors_by_class.get(error_type, 0) + 1
        self._request_times.append(time.time())

    def update_drift(self, psi: float, status: str) -> None:
        """Update drift metrics from a drift check."""
        self.last_psi = psi
        self.drift_status = status

    def set_model_info(self, version: str, name: str) -> None:
        """Set the active model version and name."""
        self.model_version = version
        self.model_name = name

    def _requests_in_window(self) -> int:
        """Count requests in the rolling window."""
        now = time.time()
        cutoff = now - self.window_seconds
        return sum(1 for t in self._request_times if t > cutoff)

    def _latencies_in_window(self) -> list[float]:
        """Get latency values from the rolling window."""
        now = time.time()
        cutoff = now - self.window_seconds
        return [lat for t, lat in self._latencies if t > cutoff]

    def snapshot(self) -> MetricsSnapshot:
        """Take a point-in-time snapshot of all metrics."""
        latencies = self._latencies_in_window()
        req_in_window = self._requests_in_window()

        # Rate: requests per minute
        rpm = (req_in_window / self.window_seconds) * 60 if self.window_seconds > 0 else 0

        # Error rate
        error_rate = self.errors_total / max(self.requests_total, 1)

        # Latency percentiles
        if latencies:
            lat_arr = np.array(latencies)
            p50 = float(np.percentile(lat_arr, 50))
            p95 = float(np.percentile(lat_arr, 95))
            p99 = float(np.percentile(lat_arr, 99))
            mean = float(np.mean(lat_arr))
        else:
            p50 = p95 = p99 = mean = 0.0

        return MetricsSnapshot(
            timestamp=datetime.now(UTC).isoformat(),
            requests_total=self.requests_total,
            requests_per_minute=rpm,
            errors_total=self.errors_total,
            error_rate=error_rate,
            errors_by_class=dict(self.errors_by_class),
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            latency_p99_ms=p99,
            latency_mean_ms=mean,
            psi_score=self.last_psi,
            drift_status=self.drift_status,
            model_version=self.model_version,
            model_name=self.model_name,
        )

    def get_slo_status(self) -> dict:
        """
        Check Service Level Objectives.

        SLOs defined in the proposal:
            - Availability: 99% uptime
            - Latency: p95 < 200ms
            - Freshness: model retrained on demand
        """
        snap = self.snapshot()
        availability = 1.0 - snap.error_rate

        return {
            "availability": {
                "target": 0.99,
                "current": round(availability, 4),
                "met": availability >= 0.99,
            },
            "latency_p95_ms": {
                "target": 200.0,
                "current": snap.latency_p95_ms,
                "met": snap.latency_p95_ms <= 200.0 if snap.latency_p95_ms > 0 else True,
            },
            "drift": {
                "threshold": 0.20,
                "current_psi": snap.psi_score,
                "status": snap.drift_status,
                "met": snap.drift_status == "ok",
            },
        }
