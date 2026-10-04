"""
alerts.py — Alert rules and emission for production monitoring.

Defines alert conditions and fires them when thresholds are breached.
Alerts are emitted via structured logging (always) and optionally
via the cloud adapter's emit_metric() for cloud monitoring dashboards.

At least one working alert is required for R3 (ITCS355 capstone rubric).
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

logger = logging.getLogger(__name__)


class AlertSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(StrEnum):
    DRIFT = "drift"
    ERROR_RATE = "error_rate"
    LATENCY = "latency"
    AVAILABILITY = "availability"


@dataclass
class Alert:
    """A triggered alert."""
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    value: float
    threshold: float
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    resolved: bool = False

    def to_dict(self) -> dict:
        return {
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
            "resolved": self.resolved,
        }


class AlertManager:
    """
    Manages alert rules, tracks fired alerts, and emits them.

    Keeps a history of alerts and supports de-duplication to avoid
    alert fatigue (same alert type won't fire more than once per cooldown period).
    """

    def __init__(self, cooldown_seconds: float = 300.0):
        """
        Args:
            cooldown_seconds: Minimum time between repeated alerts of the same type.
        """
        self.cooldown_seconds = cooldown_seconds
        self.alerts: list[Alert] = []
        self._last_fired: dict[str, float] = {}
        self._cloud_adapter = None

    def set_cloud_adapter(self, adapter) -> None:
        """Optionally attach a cloud adapter for emit_metric()."""
        self._cloud_adapter = adapter

    def _should_fire(self, alert_type: str) -> bool:
        """Check cooldown to prevent alert fatigue."""
        last = self._last_fired.get(alert_type, 0)
        return (time.time() - last) > self.cooldown_seconds

    def _emit(self, alert: Alert) -> None:
        """Emit an alert via logging and optionally cloud monitoring."""
        # Always log (structured JSON)
        log_fn = logger.warning if alert.severity == AlertSeverity.WARNING else logger.critical
        log_fn(
            f"ALERT [{alert.severity.value}] {alert.alert_type.value}: "
            f"{alert.message} (value={alert.value}, threshold={alert.threshold})"
        )

        # Cloud emission (optional)
        if self._cloud_adapter:
            try:
                self._cloud_adapter.emit_metric(
                    name=f"alert/{alert.alert_type.value}",
                    value=alert.value,
                    unit="1",
                )
            except Exception as e:
                logger.error(f"Failed to emit metric to cloud: {e}")

        self.alerts.append(alert)
        self._last_fired[alert.alert_type.value] = time.time()

    # ── Alert Rules ───────────────────────────────────────────────────────────

    def check_drift(self, psi: float, threshold: float = 0.20) -> Alert | None:
        """
        Fire alert if PSI drift score exceeds threshold.

        This is the PRIMARY alert for the capstone — demonstrates
        detection of distribution shift in production.
        """
        if psi >= threshold and self._should_fire("drift"):
            alert = Alert(
                alert_type=AlertType.DRIFT,
                severity=AlertSeverity.CRITICAL if psi >= 0.30 else AlertSeverity.WARNING,
                message=f"Prediction drift detected: PSI={psi:.4f} >= {threshold}",
                value=round(psi, 4),
                threshold=threshold,
            )
            self._emit(alert)
            return alert
        return None

    def check_error_rate(
        self,
        errors: int,
        total: int,
        threshold: float = 0.05,
    ) -> Alert | None:
        """Fire alert if error rate exceeds threshold (default 5%)."""
        if total == 0:
            return None
        rate = errors / total
        if rate >= threshold and self._should_fire("error_rate"):
            alert = Alert(
                alert_type=AlertType.ERROR_RATE,
                severity=AlertSeverity.CRITICAL if rate >= 0.10 else AlertSeverity.WARNING,
                message=f"Error rate {rate:.1%} exceeds {threshold:.1%} ({errors}/{total})",
                value=round(rate, 4),
                threshold=threshold,
            )
            self._emit(alert)
            return alert
        return None

    def check_latency(
        self,
        p95_ms: float,
        threshold_ms: float = 200.0,
    ) -> Alert | None:
        """Fire alert if p95 latency exceeds target SLO."""
        if p95_ms >= threshold_ms and self._should_fire("latency"):
            alert = Alert(
                alert_type=AlertType.LATENCY,
                severity=AlertSeverity.WARNING,
                message=f"p95 latency {p95_ms:.1f}ms exceeds SLO {threshold_ms:.0f}ms",
                value=round(p95_ms, 1),
                threshold=threshold_ms,
            )
            self._emit(alert)
            return alert
        return None

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_active_alerts(self) -> list[dict]:
        """Return all unresolved alerts."""
        return [a.to_dict() for a in self.alerts if not a.resolved]

    def get_alert_history(self, limit: int = 50) -> list[dict]:
        """Return recent alert history."""
        return [a.to_dict() for a in self.alerts[-limit:]]

    def resolve_all(self) -> int:
        """Mark all alerts as resolved. Returns count resolved."""
        count = 0
        for a in self.alerts:
            if not a.resolved:
                a.resolved = True
                count += 1
        return count
