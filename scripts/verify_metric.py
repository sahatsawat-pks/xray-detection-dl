"""
verify_metric.py — Compare trained model metrics against claims in README.md.

This is what `make verify` runs (and what the grading rubric checks).
Verifies that the reproduced run matches the claimed metrics within tolerance.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "results" / "ModelFinal_results.json"
README = ROOT / "README.md"

CLAIM_PATTERN = re.compile(
    r"expected\s+(?P<metric>[a-zA-Z0-9_]+)\s*[:=]\s*(?P<value>[0-9.]+)\s*(?:±|\+/-)\s*(?P<tol>[0-9.]+)",
    re.IGNORECASE,
)


def main() -> int:
    if not METRICS.exists():
        print(f"❌ FAIL: {METRICS} missing — run 'make train' or 'make reproduce' first")
        return 1

    if not README.exists():
        print(f"❌ FAIL: {README} missing")
        return 1

    readme_text = README.read_text(encoding="utf-8")
    matches = list(CLAIM_PATTERN.finditer(readme_text))

    if not matches:
        print("❌ FAIL: README.md has no claim line.")
        print("         Expected format in README:")
        print("           expected test_auc: 0.913 ± 0.020")
        print("           expected test_f1: 0.687 ± 0.030")
        return 1

    actual_metrics = json.loads(METRICS.read_text(encoding="utf-8"))

    # Map possible metric name variants
    key_mapping = {
        "test_auc": "AUC",
        "auc": "AUC",
        "test_f1": "F1",
        "f1": "F1",
        "test_accuracy": "Accuracy",
        "accuracy": "Accuracy",
        "test_precision": "Precision",
        "precision": "Precision",
        "test_recall": "Recall",
        "recall": "Recall",
    }

    print("=== Metric Verification (README vs Actual) ===")
    all_passed = True

    for m in matches:
        metric_name = m.group("metric").lower()
        claimed = float(m.group("value"))
        tol = float(m.group("tol"))

        actual_key = key_mapping.get(metric_name)
        if not actual_key or actual_key not in actual_metrics:
            print(f"⚠️  Metric '{metric_name}' not found in actual results (available: {list(actual_metrics.keys())})")
            continue

        actual = float(actual_metrics[actual_key])
        delta = abs(actual - claimed)

        status = "✅ PASS" if delta <= tol else "❌ FAIL"
        print(f"  {metric_name.upper():12s}: Claimed {claimed:.4f} ± {tol:.4f} | Actual {actual:.4f} | Delta {delta:.4f}  {status}")

        if delta > tol:
            all_passed = False

    if all_passed:
        print("\n✅ Verification PASSED — actual metrics match claims within tolerance.")
        return 0
    else:
        print("\n❌ Verification FAILED — metrics outside claimed tolerance.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
