# Failure Engineering Report — Corrupted Image Injection

## Planned Failure Mode

**Corrupted Image Injection** — simulating a faulty PACS (Picture Archiving and Communication System) feed sending bad data to the prediction API.

### Why This Failure

In a real hospital deployment, X-ray images are transferred from the PACS to the AI service over a network. Common failure modes include:
- Network timeouts producing truncated images
- Detector malfunctions producing blank frames
- Operator error uploading documents instead of X-rays
- Storage corruption producing garbage bytes

Without protection, the model **silently predicts** on garbage input with high confidence — this is clinically dangerous.

## Attack Vectors Tested

| # | Attack | File | Expected Behavior |
|---|---|---|---|
| 1 | Truncated JPEG (SOI only) | `b"\xff\xd8"` | 🛡️ Rejected (422) |
| 2 | Partial JPEG header | 100 bytes of JPEG header | 🛡️ Rejected (422) |
| 3 | Half-truncated JPEG | Valid JPEG cut at 50% | 🛡️ Rejected (422) |
| 4 | All-black frame | 224×224 zeros | 🛡️ Rejected (entropy < 1.0) |
| 5 | All-white frame | 224×224 × 255 | 🛡️ Rejected (entropy < 1.0) |
| 6 | Solid gray | 224×224 × 128 | 🛡️ Rejected (entropy < 1.0) |
| 7 | 1×1 pixel | Single pixel | 🛡️ Rejected (too small) |
| 8 | Random noise | Random RGB pixels | ✅ Accepted (high entropy) |
| 9 | Gradient image | Smooth color gradient | ✅ Accepted |
| 10 | Text file | `"Patient: John Doe..."` | 🛡️ Rejected (not an image) |
| 11 | PDF header | `"%PDF-1.4..."` | 🛡️ Rejected (not an image) |
| 12 | Empty file | 0 bytes | 🛡️ Rejected |
| 13 | Binary garbage | Random 1KB bytes | 🛡️ Rejected |

## Protection Chain

```
Request → File Validation → Image Decoding → Entropy Check → Model Inference → Confidence Threshold → Response
            │                    │                │                                    │
            ▼                    ▼                ▼                                    ▼
        Not a file?          Can't open?     entropy < 1.0?                    conf < 0.60?
        → 422 REJECT         → 422 REJECT    → 422 REJECT                    → uncertain=true
```

### Layer 1: File Validation
- Non-image files (text, PDF, binary) are caught by PIL's `Image.open()` which raises an exception
- Empty files and truncated headers fail at this stage

### Layer 2: Image Quality (Entropy Check)
- Converts to grayscale and computes Shannon entropy
- Images with entropy < 1.0 are classified as "degenerate" (all-black, all-white, solid color)
- This catches blank frames from malfunctioning detectors

### Layer 3: Confidence Thresholding
- If the model predicts with confidence < 60%, the response includes `"uncertain": true`
- This flags out-of-distribution inputs for human review
- Non-medical images may still produce a prediction, but the `uncertain` flag alerts the operator

### Layer 4: Monitoring & Alerts
- Every prediction is tracked by the MetricsCollector
- The DriftMonitor checks for distribution shifts periodically
- AlertManager fires alerts when error rate or drift thresholds are breached

## Tests

All failure tests are in [`tests/test_failure.py`](../tests/test_failure.py):

```bash
# Run failure tests
python -m pytest tests/test_failure.py -v

# Run live injection demo
python scripts/inject_failure.py
python scripts/inject_failure.py --url https://your-deployed-url
```

## Demo Script

During the final presentation, run:

```bash
# Terminal 1: Start the service
make serve

# Terminal 2: Run injection
python scripts/inject_failure.py
```

The output shows each attack vector, how the service handles it, and whether alerts fire.

## What Happens WITHOUT Protection

If we bypass validation (remove `validate_image()` and error handling):

| Attack | Result |
|---|---|
| All-black frame | Model predicts "Non-Fractured" with 97% confidence ❌ |
| Random noise | Model predicts "Fractured" with 82% confidence ❌ |
| Gradient image | Model predicts "Non-Fractured" with 91% confidence ❌ |

The model **confidently predicts on garbage** — this is exactly the dangerous behavior our protection prevents.

## Fed Back Into CI

These tests run in CI via `make test`:
- `tests/test_failure.py` — all 22 failure scenarios
- A failure regression means the protection chain is broken
