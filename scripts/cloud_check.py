"""
cloud_check.py — Resolve environment contract capability slots and report status.

Checks the standard ITCS355 cloud slots and CLI tools.
Usage:
    python scripts/cloud_check.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

# Load cloud.env if present
env_file = Path(__file__).resolve().parent.parent / "cloud.env"
if env_file.exists():
    load_dotenv(env_file)

CAPABILITY_SLOTS = [
    "CLOUD_PROVIDER",
    "PROJECT_ID",
    "REGION",
    "BLOB_URI",
    "CONTAINER_REGISTRY",
    "MLFLOW_TRACKING_URI",
    "ENDPOINT_NAME",
    "METRICS_NAMESPACE",
]

CLI_FOR = {
    "gcp": "gcloud",
    "aws": "aws",
    "azure": "az",
    "local": None,
}

IDENTITY_CMD = {
    "gcp": ["gcloud", "auth", "list"],
    "aws": ["aws", "sts", "get-caller-identity"],
    "azure": ["az", "account", "show"],
}


def line(name: str, ok: bool, detail: str = "") -> bool:
    tag = "PASS" if ok else "FAIL"
    icon = "✅" if ok else "❌"
    print(f"  {icon} [{tag}] {name:<24} {detail}")
    return ok


def main() -> int:
    print("=== ITCS355 Cloud Environment Check ===\n")
    results = []

    provider = os.environ.get("CLOUD_PROVIDER", "").lower()
    valid_provider = provider in CLI_FOR
    results.append(line("CLOUD_PROVIDER", valid_provider, provider or "unset — expected gcp | aws | azure"))

    for slot in CAPABILITY_SLOTS:
        if slot == "CLOUD_PROVIDER":
            continue
        val = os.environ.get(slot, "")
        shown = val if len(val) < 40 else val[:37] + "..."
        results.append(line(slot, bool(val), shown or "unset (check cloud.env)"))

    print("\n--- Tooling & Identity ---")
    cli = CLI_FOR.get(provider)
    if cli:
        found = shutil.which(cli) is not None
        results.append(line(f"{cli} on PATH", found, "" if found else f"install {cli}"))
        if found:
            try:
                subprocess.run(IDENTITY_CMD[provider], capture_output=True, check=True, timeout=15)
                results.append(line("Cloud credentials", True, "identity resolved"))
            except Exception as e:
                results.append(line("Cloud credentials", False, f"{type(e).__name__} (run '{cli} auth login')"))
    else:
        line("Cloud CLI", True, "local provider")

    docker_ok = shutil.which("docker") is not None
    results.append(line("docker on PATH", docker_ok))

    passed = sum(results)
    total = len(results)
    print(f"\nSummary: {passed}/{total} checks passed.")

    if passed == total:
        print("✅ All cloud environment checks passed!")
        return 0
    else:
        print("⚠️ Some checks failed. Fill in missing variables in cloud.env.")
        return 0  # Do not block makefile execution when running locally


if __name__ == "__main__":
    raise SystemExit(main())
