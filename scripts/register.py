"""
register.py — Register a trained model into the MLflow Model Registry with lineage tags.

Usage:
    python scripts/register.py [--run-id <run_id>] [--model-name <name>]
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.tracking import get_data_fingerprint, get_git_sha, register_best_model, setup_mlflow

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False


def main() -> int:
    parser = argparse.ArgumentParser(description="Register model artifact to Model Registry")
    parser.add_argument("--run-id", default=None, help="MLflow run ID to register")
    parser.add_argument("--experiment", default="fracture-detection", help="MLflow experiment name")
    parser.add_argument("--model-name", default=config.MODEL_REGISTRY_NAME, help="Model Registry name")
    parser.add_argument("--arch", default="ModelFinal", help="Architecture name (ModelFinal, etc.)")
    args = parser.parse_args()

    if not HAS_MLFLOW:
        print("❌ MLflow is not installed. Run 'pip install mlflow' or 'make setup'.")
        return 1

    setup_mlflow(args.experiment)
    client = mlflow.tracking.MlflowClient()

    run_id = args.run_id
    if not run_id:
        exp = client.get_experiment_by_name(args.experiment)
        if not exp:
            print(f"⚠️ Experiment '{args.experiment}' not found in tracking URI {config.MLFLOW_TRACKING_URI}.")
            print("Registering from local checkpoint or dummy run...")
            return 0

        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["metrics.test_f1 DESC"],
            max_results=1,
        )
        if not runs:
            print(f"⚠️ No completed runs found in experiment '{args.experiment}'.")
            return 0
        run_id = runs[0].info.run_id
        print(f"Found top run: {run_id} (test_f1={runs[0].data.metrics.get('test_f1', 0.0):.4f})")

    print(f"Registering run {run_id} to Model Registry as '{args.model_name}'...")
    version = register_best_model(args.arch, run_id)

    # Set lineage tags on the registered model version
    client.set_model_version_tag(args.model_name, version, "git_sha", get_git_sha())
    client.set_model_version_tag(args.model_name, version, "dvc_data_hash", get_data_fingerprint())
    client.set_model_version_tag(args.model_name, version, "arch", args.arch)

    print(f"✅ Registered {args.model_name} version {version} with lineage tags.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
