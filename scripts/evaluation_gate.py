"""
evaluation_gate.py — Registration & deployment evaluation gate.

Compares candidate evaluation metrics against an absolute floor and an
optional incumbent score. Exits with non-zero code when the candidate
should NOT be registered or promoted.

Usage:
    python scripts/evaluation_gate.py --metrics results/ModelFinal_results.json
    python scripts/evaluation_gate.py --metrics results/ModelFinal_results.json --metric F1 --floor 0.85
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Default thresholds
DEFAULT_MIN_ABSOLUTE = 0.80  # Minimum acceptable score for production
DEFAULT_MIN_IMPROVEMENT = 0.005  # Margin required to replace incumbent


def main() -> int:
    parser = argparse.ArgumentParser(description="Model evaluation gate for promotion")
    parser.add_argument(
        "--metrics",
        type=Path,
        required=True,
        help="Path to evaluation metrics JSON file (e.g., results/ModelFinal_results.json)",
    )
    parser.add_argument(
        "--metric",
        default="F1",
        help="Metric key to evaluate (e.g., F1, Accuracy, AUC)",
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=DEFAULT_MIN_ABSOLUTE,
        help=f"Absolute minimum metric score required (default: {DEFAULT_MIN_ABSOLUTE})",
    )
    parser.add_argument(
        "--incumbent",
        type=float,
        default=None,
        help="Incumbent model score; if omitted, treated as first registration",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=DEFAULT_MIN_IMPROVEMENT,
        help=f"Minimum improvement required over incumbent (default: {DEFAULT_MIN_IMPROVEMENT})",
    )
    args = parser.parse_args()

    if not args.metrics.exists():
        print(f"❌ GATE FAIL: Metrics file not found at {args.metrics}")
        return 1

    try:
        metrics_dict = json.loads(args.metrics.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"❌ GATE FAIL: Failed to parse metrics JSON: {e}")
        return 1

    if args.metric not in metrics_dict:
        # Fallback case-insensitive check
        matched_keys = [k for k in metrics_dict if k.lower() == args.metric.lower()]
        if not matched_keys:
            print(f"❌ GATE FAIL: Metric '{args.metric}' not in metrics file (available: {list(metrics_dict.keys())})")
            return 1
        metric_key = matched_keys[0]
    else:
        metric_key = args.metric

    candidate = float(metrics_dict[metric_key])
    print(f"Candidate {metric_key}: {candidate:.4f} (Required floor: {args.floor:.4f})")

    if candidate < args.floor:
        print(f"❌ GATE FAIL: Candidate {metric_key} ({candidate:.4f}) is below absolute floor ({args.floor:.4f})")
        return 1

    if args.incumbent is not None:
        delta = candidate - args.incumbent
        print(f"Incumbent: {args.incumbent:.4f} | Delta: {delta:+.4f} | Min improvement: {args.min_improvement:+.4f}")
        if delta < args.min_improvement:
            print(f"❌ GATE FAIL: Improvement {delta:+.4f} below required margin {args.min_improvement:+.4f}")
            return 1

    print("✅ GATE PASS: Candidate meets all quality requirements for promotion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
