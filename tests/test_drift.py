"""
test_drift.py — Tests for drift detection and alert system (R3).

Verifies PSI computation, drift classification, alert triggering,
and the DriftMonitor state machine. No cloud SDK imports.
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from monitoring.alerts import AlertManager, AlertType
from monitoring.dashboard import MetricsCollector
from monitoring.drift import (
    PSI_THRESHOLD_ALERT,
    PSI_THRESHOLD_WARN,
    DriftMonitor,
    check_class_ratio_drift,
    check_prediction_drift,
    compute_psi,
)


# ═══════════════════════════════════════════════════════════════════════════════
# PSI Computation
# ═══════════════════════════════════════════════════════════════════════════════
class TestPSI:
    """Tests for Population Stability Index calculation."""

    def test_identical_distributions(self):
        """PSI of identical distributions should be ~0."""
        np.random.seed(42)
        baseline = np.random.uniform(0, 1, size=1000)
        psi = compute_psi(baseline, baseline)
        assert psi < 0.01, f"PSI should be near 0 for identical, got {psi}"

    def test_similar_distributions(self):
        """Slightly different distributions should have low PSI."""
        np.random.seed(42)
        baseline = np.random.beta(2, 5, size=1000)
        current = np.random.beta(2.1, 5.1, size=1000)
        psi = compute_psi(baseline, current)
        assert psi < PSI_THRESHOLD_WARN, f"Similar distributions should have PSI < 0.10, got {psi}"

    def test_shifted_distribution(self):
        """A clearly shifted distribution should have high PSI."""
        np.random.seed(42)
        baseline = np.random.beta(2, 8, size=1000)  # skew low
        current = np.random.beta(8, 2, size=1000)   # skew high (opposite)
        psi = compute_psi(baseline, current)
        assert psi > PSI_THRESHOLD_ALERT, f"Shifted distribution should have PSI >= 0.20, got {psi}"

    def test_psi_is_non_negative(self):
        """PSI should always be >= 0."""
        np.random.seed(42)
        for _ in range(10):
            a = np.random.uniform(0, 1, size=100)
            b = np.random.uniform(0, 1, size=100)
            psi = compute_psi(a, b)
            assert psi >= 0, f"PSI should be non-negative, got {psi}"

    def test_small_sample_handling(self):
        """PSI should handle small samples without crashing."""
        baseline = np.array([0.1, 0.2, 0.3])
        current = np.array([0.7, 0.8, 0.9])
        psi = compute_psi(baseline, current)
        assert isinstance(psi, float)


# ═══════════════════════════════════════════════════════════════════════════════
# Drift Detection
# ═══════════════════════════════════════════════════════════════════════════════
class TestDriftDetection:
    """Tests for the drift check functions."""

    def test_no_drift_detected(self):
        """Stable distribution should report no drift."""
        np.random.seed(42)
        baseline = np.random.beta(2, 5, size=500)
        current = np.random.beta(2, 5, size=500)
        result = check_prediction_drift(baseline, current)
        assert result["drifted"] is False
        assert result["severity"] == "none"

    def test_drift_detected(self):
        """Shifted distribution should report drift."""
        np.random.seed(42)
        baseline = np.random.beta(2, 8, size=500)
        current = np.random.beta(8, 2, size=500)
        result = check_prediction_drift(baseline, current)
        assert result["drifted"] is True
        assert result["severity"] in ("warning", "critical")

    def test_drift_result_schema(self):
        """Drift result should have all required fields."""
        np.random.seed(42)
        baseline = np.random.uniform(0, 1, size=100)
        current = np.random.uniform(0, 1, size=100)
        result = check_prediction_drift(baseline, current)
        required_keys = [
            "psi", "drifted", "severity", "threshold",
            "baseline_size", "current_size", "timestamp",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_class_ratio_drift_stable(self):
        """Stable class ratio should report no drift."""
        baseline = np.array([0, 0, 0, 0, 1])  # 20% positive
        current = np.array([0, 0, 0, 0, 1])   # 20% positive
        result = check_class_ratio_drift(baseline, current)
        assert result["drifted"] is False

    def test_class_ratio_drift_shifted(self):
        """Shifted class ratio should report drift."""
        baseline = np.array([0, 0, 0, 0, 1])  # 20% positive
        current = np.array([1, 1, 1, 1, 0])   # 80% positive
        result = check_class_ratio_drift(baseline, current, tolerance=0.10)
        assert result["drifted"] is True
        assert result["deviation"] > 0.10


# ═══════════════════════════════════════════════════════════════════════════════
# DriftMonitor (stateful)
# ═══════════════════════════════════════════════════════════════════════════════
class TestDriftMonitor:
    """Tests for the stateful DriftMonitor."""

    def test_record_accumulates(self):
        """Recording predictions should increment the counter."""
        monitor = DriftMonitor(window_size=50, check_interval=100)
        for i in range(10):
            monitor.record(0.3, 0)
        assert monitor.count == 10

    def test_check_triggers_at_interval(self):
        """Drift check should trigger at the configured interval."""
        np.random.seed(42)
        monitor = DriftMonitor(window_size=50, check_interval=10)
        # Manually set baseline
        monitor.baseline_probs = np.random.beta(2, 5, size=100)
        monitor.baseline_labels = (monitor.baseline_probs >= 0.5).astype(int)

        results = []
        for i in range(20):
            result = monitor.record(np.random.beta(2, 5), 0)
            if result is not None:
                results.append(result)

        assert len(results) >= 1, "Drift check should trigger at interval"

    def test_save_and_load_baseline(self, tmp_path):
        """Baseline should save and reload correctly."""
        np.random.seed(42)
        probs = np.random.uniform(0, 1, size=50)
        labels = (probs >= 0.5).astype(int)

        path = str(tmp_path / "baseline.json")
        monitor = DriftMonitor()
        monitor.save_baseline(probs, labels, path)

        monitor2 = DriftMonitor(baseline_path=path)
        assert monitor2.baseline_probs is not None
        assert len(monitor2.baseline_probs) == 50


# ═══════════════════════════════════════════════════════════════════════════════
# Alert Manager
# ═══════════════════════════════════════════════════════════════════════════════
class TestAlertManager:
    """Tests for the AlertManager."""

    def test_drift_alert_fires(self):
        """Alert should fire when PSI exceeds threshold."""
        mgr = AlertManager(cooldown_seconds=0)
        alert = mgr.check_drift(psi=0.30, threshold=0.20)
        assert alert is not None
        assert alert.alert_type == AlertType.DRIFT
        assert alert.value == 0.30

    def test_drift_alert_below_threshold(self):
        """No alert should fire when PSI is below threshold."""
        mgr = AlertManager(cooldown_seconds=0)
        alert = mgr.check_drift(psi=0.05, threshold=0.20)
        assert alert is None

    def test_error_rate_alert(self):
        """Alert should fire when error rate exceeds threshold."""
        mgr = AlertManager(cooldown_seconds=0)
        alert = mgr.check_error_rate(errors=10, total=100, threshold=0.05)
        assert alert is not None
        assert alert.alert_type == AlertType.ERROR_RATE

    def test_latency_alert(self):
        """Alert should fire when p95 latency exceeds SLO."""
        mgr = AlertManager(cooldown_seconds=0)
        alert = mgr.check_latency(p95_ms=250.0, threshold_ms=200.0)
        assert alert is not None
        assert alert.alert_type == AlertType.LATENCY

    def test_cooldown_prevents_spam(self):
        """Same alert type should not fire within cooldown period."""
        mgr = AlertManager(cooldown_seconds=9999)
        alert1 = mgr.check_drift(psi=0.30)
        alert2 = mgr.check_drift(psi=0.35)
        assert alert1 is not None
        assert alert2 is None, "Second alert should be suppressed by cooldown"

    def test_alert_history(self):
        """Alert history should track all fired alerts."""
        mgr = AlertManager(cooldown_seconds=0)
        mgr.check_drift(psi=0.30)
        mgr.check_error_rate(errors=10, total=100)
        history = mgr.get_alert_history()
        assert len(history) == 2

    def test_resolve_all(self):
        """resolve_all should mark alerts as resolved."""
        mgr = AlertManager(cooldown_seconds=0)
        mgr.check_drift(psi=0.30)
        count = mgr.resolve_all()
        assert count == 1
        assert len(mgr.get_active_alerts()) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics Collector
# ═══════════════════════════════════════════════════════════════════════════════
class TestMetricsCollector:
    """Tests for the MetricsCollector."""

    def test_record_and_snapshot(self):
        """Recording requests should be reflected in the snapshot."""
        mc = MetricsCollector(window_seconds=60)
        mc.record_request(latency_ms=50.0, label="Fractured", confidence=0.85, uncertain=False)
        mc.record_request(latency_ms=60.0, label="Non-Fractured", confidence=0.92, uncertain=False)

        snap = mc.snapshot()
        assert snap.requests_total == 2
        assert snap.errors_total == 0

    def test_latency_percentiles(self):
        """Latency percentiles should be computed correctly."""
        mc = MetricsCollector(window_seconds=60)
        for i in range(100):
            mc.record_request(latency_ms=float(i), label="Fractured", confidence=0.8, uncertain=False)

        snap = mc.snapshot()
        assert snap.latency_p50_ms > 0
        assert snap.latency_p95_ms > snap.latency_p50_ms
        assert snap.latency_p99_ms >= snap.latency_p95_ms

    def test_slo_check(self):
        """SLO status should report met/not-met for each objective."""
        mc = MetricsCollector(window_seconds=60)
        for _ in range(10):
            mc.record_request(latency_ms=50.0, label="Fractured", confidence=0.9, uncertain=False)

        slo = mc.get_slo_status()
        assert "availability" in slo
        assert "latency_p95_ms" in slo
        assert "drift" in slo
        assert slo["availability"]["met"] is True

    def test_error_rate_tracking(self):
        """Error rate should be computed from recorded errors."""
        mc = MetricsCollector(window_seconds=60)
        for _ in range(9):
            mc.record_request(latency_ms=50.0, label="Fractured", confidence=0.9, uncertain=False)
        mc.record_error("validation_error")

        snap = mc.snapshot()
        assert snap.errors_total == 1
        assert snap.requests_total == 9  # record_error doesn't count as request
        assert snap.error_rate > 0.0  # Some errors recorded
