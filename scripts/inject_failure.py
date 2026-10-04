#!/usr/bin/env python3
"""
inject_failure.py — Live corrupted image injection demo (R4).

Sends corrupted and valid images to the running API to demonstrate
the failure mode and protection mechanisms in real-time.

Usage:
    # Start the server first
    make serve

    # Run injection demo
    python scripts/inject_failure.py
    python scripts/inject_failure.py --url https://deployed-service-url
"""

import argparse
import io
import sys
import time

import numpy as np
import requests
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_URL = "http://localhost:8000"
ATTACKS = []


def register_attack(name, description):
    """Decorator to register an attack vector."""
    def decorator(func):
        ATTACKS.append({"name": name, "desc": description, "func": func})
        return func
    return decorator


def _make_jpeg(arr, mode="RGB") -> bytes:
    img = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


# ═══════════════════════════════════════════════════════════════════════════════
# Attack Vectors
# ═══════════════════════════════════════════════════════════════════════════════

@register_attack("Valid X-Ray", "Baseline — should succeed")
def attack_valid():
    arr = np.random.randint(30, 220, (224, 224, 3), dtype=np.uint8)
    return _make_jpeg(arr), "valid.jpg"


@register_attack("Truncated JPEG", "Simulates interrupted PACS transfer")
def attack_truncated():
    valid = _make_jpeg(np.random.randint(30, 220, (224, 224, 3), dtype=np.uint8))
    return valid[:len(valid) // 4], "truncated.jpg"


@register_attack("All-Black Frame", "Dead X-ray detector")
def attack_black():
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    return _make_jpeg(arr), "black.jpg"


@register_attack("All-White Frame", "Overexposed detector")
def attack_white():
    arr = np.ones((224, 224, 3), dtype=np.uint8) * 255
    return _make_jpeg(arr), "white.jpg"


@register_attack("Text File", "Document photo from wrong folder")
def attack_text():
    return b"Patient: John Doe\nDiagnosis: Pending review\n", "report.txt"


@register_attack("Empty File", "Zero-byte transfer")
def attack_empty():
    return b"", "empty.jpg"


@register_attack("Random Noise", "Non-medical image (static)")
def attack_noise():
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return _make_jpeg(arr), "noise.jpg"


@register_attack("Solid Gray", "Sensor calibration image")
def attack_gray():
    arr = np.ones((224, 224, 3), dtype=np.uint8) * 128
    return _make_jpeg(arr), "gray.jpg"


@register_attack("1×1 Pixel", "Minimum possible image")
def attack_tiny():
    arr = np.array([[[128, 128, 128]]], dtype=np.uint8)
    return _make_jpeg(arr), "tiny.jpg"


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_injection(base_url: str) -> list[dict]:
    """Run all attack vectors and collect results."""
    results = []

    print("=" * 70)
    print("  🔬 CORRUPTED IMAGE INJECTION DEMO")
    print("  Failure Mode: Faulty PACS Feed → Corrupted Images")
    print(f"  Target: {base_url}")
    print("=" * 70)
    print()

    # Check service health first
    try:
        health = requests.get(f"{base_url}/health", timeout=5)
        if health.status_code != 200:
            print("❌ Service is not healthy. Start with: make serve")
            sys.exit(1)
    except requests.ConnectionError:
        print(f"❌ Cannot connect to {base_url}. Start with: make serve")
        sys.exit(1)

    # Run attacks
    for i, attack in enumerate(ATTACKS, 1):
        print(f"  [{i}/{len(ATTACKS)}] {attack['name']}")
        print(f"  Scenario: {attack['desc']}")

        payload, filename = attack["func"]()
        start = time.perf_counter()

        try:
            resp = requests.post(
                f"{base_url}/predict",
                files={"file": (filename, payload, "image/jpeg")},
                timeout=10,
            )
            elapsed = (time.perf_counter() - start) * 1000
            status = resp.status_code
            body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}

            result = {
                "attack": attack["name"],
                "status": status,
                "response_ms": round(elapsed, 1),
            }

            if status == 200:
                label = body.get("label", "?")
                conf = body.get("confidence", 0)
                uncertain = body.get("uncertain", False)
                result["label"] = label
                result["confidence"] = conf
                result["uncertain"] = uncertain

                flag = "⚠️  UNCERTAIN" if uncertain else "✅ ACCEPTED"
                print(f"  Result: {flag} → {label} ({conf:.1%})")
            elif status == 422:
                detail = body.get("detail", "rejected")
                result["detail"] = detail
                print(f"  Result: 🛡️  REJECTED → {detail}")
            else:
                print(f"  Result: ❓ HTTP {status}")

            results.append(result)

        except Exception as e:
            print(f"  Result: 💥 ERROR → {e}")
            results.append({"attack": attack["name"], "status": -1, "error": str(e)})

        print(f"  Latency: {elapsed:.0f}ms")
        print()

    # Summary
    print("=" * 70)
    print("  📊 INJECTION SUMMARY")
    print("=" * 70)

    accepted = sum(1 for r in results if r.get("status") == 200 and not r.get("uncertain"))
    uncertain = sum(1 for r in results if r.get("uncertain"))
    rejected = sum(1 for r in results if r.get("status") == 422)
    errored = sum(1 for r in results if r.get("status") not in (200, 422))

    print(f"  ✅ Accepted:   {accepted}")
    print(f"  ⚠️  Uncertain:  {uncertain}")
    print(f"  🛡️  Rejected:   {rejected}")
    print(f"  💥 Errored:    {errored}")
    print(f"  Total:        {len(results)}")
    print()

    # Check alerts fired
    try:
        alerts = requests.get(f"{base_url}/alerts", timeout=5).json()
        active = alerts.get("active", [])
        if active:
            print(f"  🚨 ALERTS FIRED: {len(active)}")
            for a in active:
                print(f"     [{a['severity']}] {a['message']}")
        else:
            print("  📭 No alerts fired (expected for small injection)")
    except Exception:
        pass

    print()
    print("  Protection chain: Input validation → Entropy check → Confidence threshold → Alert")
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(description="Corrupted image injection demo")
    parser.add_argument(
        "--url", default=DEFAULT_URL,
        help=f"Base URL of the API (default: {DEFAULT_URL})",
    )
    args = parser.parse_args()
    run_injection(args.url)


if __name__ == "__main__":
    main()
