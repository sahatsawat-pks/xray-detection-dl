# 🦴 Bone Fracture X-Ray Detection — MLOps Capstone

> **ITCS355 — Machine Learning Operation and Deployment**
> Binary classification of bone fractures from X-ray images, shipped end-to-end with full operational infrastructure.

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?logo=pytorch)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.10+-0194E2?logo=mlflow)](https://mlflow.org)
[![GCP](https://img.shields.io/badge/GCP-Cloud%20Run-4285F4?logo=google-cloud)](https://cloud.google.com)

---

## 📌 Problem Statement

Bone fractures are among the most common musculoskeletal injuries in emergency departments. Manual X-ray reading under high workload is time-consuming and error-prone — missed fractures (false negatives) lead to delayed treatment.

This project ships an **AI-assisted triage service** end-to-end: from training pipeline to production FastAPI endpoint with monitoring, drift detection, and engineered failure handling. **Model accuracy carries no marks** — the grade is on the operational system.

---

## 🏗️ Architecture

```
Three-Layer Contract (ITCS355):

src/           Layer 1 — provider-neutral ML code (no cloud SDK imports)
cloudlayer/    Layer 2 — GCP adapter (only place cloud SDKs appear)
service/       Layer 3 — FastAPI inference service
monitoring/    Drift detection, alerts, metrics dashboard
tests/         81 tests (unit, data, integration, failure)
```

### Key invariant
`make portability-audit` enforces that `src/` and `tests/` contain **zero** cloud-specific strings (`gs://`, `googleapis`, `boto3`, etc.). All cloud interaction goes through `cloudlayer/`.

---

## 📂 Project Structure

```
xray-detection-dl/
├── src/                        # Layer 1: Provider-neutral ML
│   ├── config.py               # Single point of environment knowledge
│   ├── models.py               # ModelBaseline, ModelImproved, ModelFinal
│   ├── dataset.py              # FractureDataset + DataLoader factory
│   ├── evaluate.py             # Metrics, confusion matrix, ROC, Grad-CAM
│   ├── tracking.py             # MLflow tracking + lineage tags
│   └── train.py                # MLflow-tracked training (replaces trainer.py)
│
├── cloudlayer/                 # Layer 2: Cloud adapter
│   ├── base.py                 # CloudAdapter ABC (11 methods)
│   └── gcp.py                  # GCP implementation
│
├── service/                    # Layer 3: FastAPI inference
│   ├── app.py                  # /health, /ready, /predict, /metrics, /drift, /slo
│   ├── schemas.py              # Pydantic models with lineage
│   └── model_loader.py         # Singleton loader + image validation
│
├── monitoring/                 # Observability
│   ├── drift.py                # PSI + KS drift detection
│   ├── alerts.py               # AlertManager (drift, error rate, latency)
│   └── dashboard.py            # MetricsCollector (5 required metrics)
│
├── tests/                      # 81 tests
│   ├── test_model.py           # 12 model architecture tests
│   ├── test_data.py            # 13 data contract tests
│   ├── test_service.py         # 13 integration tests
│   ├── test_drift.py           # 24 monitoring tests
│   └── test_failure.py         # 19 engineered failure tests
│
├── .github/workflows/
│   ├── ci.yml                  # Lint → audit → test → Docker build
│   └── cd.yml                  # Push → Artifact Registry → Cloud Run → smoke test
│
├── loadtest/
│   ├── smoke.js                # k6: 1/10/50 VUs, p95 < 200ms threshold
│   └── README.md               # Pre-declared latency target
│
├── scripts/
│   ├── portability_audit.sh    # Enforces 3-layer rule
│   └── inject_failure.py       # Live failure injection demo
│
├── evals/
│   └── failure_report.md       # Failure engineering documentation
│
├── docs/
│   ├── proposal.md             # Capstone proposal (M2)
│   ├── model_card.md           # Model Card (Mitchell et al. 2019)
│   └── cost_report.md          # Honest cost breakdown
│
├── app/                        # Demo UIs (not production)
│   ├── app.py                  # Gradio demo
│   └── streamlit_app.py        # Streamlit demo
│
├── Makefile                    # 18 targets — central command interface
├── Dockerfile                  # Digest-pinned, multi-layer
├── requirements.in             # Top-level dependencies
├── cloud.env.example           # Environment contract template
└── ruff.toml                   # Linter config
```

---

## 🚀 Quickstart

### One-command reproduction

```bash
make reproduce
```

Metric claims for automated verification (`make verify`):
- expected test_auc: 0.913 ± 0.020
- expected test_f1: 0.687 ± 0.030

This runs: `setup` → `data` → `train` → `verify`

### Step-by-step

```bash
# 1. Install dependencies
make setup

# 2. Verify dataset is present
make data

# 3. Train all models with MLflow tracking
make train

# 4. Run all 81 tests
make test

# 5. Check portability (3-layer rule)
make portability-audit

# 6. Verify reproducible metrics against claims
make verify

# 7. Start FastAPI server
make serve
# → http://localhost:8000/docs (Swagger UI)

# 8. Run failure injection demo
python scripts/inject_failure.py
```

### Train a single model

```bash
make train-model MODEL=ModelFinal
```

---

## 📊 Results

| Model | Accuracy | Precision | Recall | **F1** | AUC |
|---|---|---|---|---|---|
| ModelBaseline (CNN) | 0.8412 | 0.6250 | 0.2336 | 0.3401 | 0.8080 |
| ModelImproved (ResNet-18) | 0.9051 | 0.7753 | 0.6449 | 0.7041 | 0.9046 |
| **ModelFinal (EfficientNet-B0)** | **0.8969** | 0.7340 | **0.6449** | **0.6866** | **0.9130** |

**Selected model**: EfficientNet-B0 — highest F1-score with smallest parameter count (5.3M). High recall prioritised: missed fractures are clinically more dangerous than false alarms.

---

## 🔌 API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness probe (always 200 if alive) |
| `/ready` | GET | Readiness probe (200 if model loaded, 503 if not) |
| `/predict` | POST | Single X-ray → label + confidence + lineage |
| `/predict/batch` | POST | Multiple images with partial failure handling |
| `/metrics` | GET | Dashboard snapshot (5 required metrics) |
| `/drift` | GET | Latest PSI drift check |
| `/slo` | GET | SLO status (availability, latency, drift) |
| `/alerts` | GET | Active + historical alerts |

---

## 🛡️ Failure Engineering

**Planned failure**: Corrupted Image Injection (simulating a faulty PACS feed)

| Layer | Protection |
|---|---|
| File validation | Rejects non-images (text, PDF, binary) |
| Image decoding | Catches truncated/corrupted JPEGs |
| Entropy check | Rejects blank frames (all-black/white, entropy < 1.0) |
| Confidence threshold | Flags uncertain predictions (conf < 0.60) |
| Monitoring | Alerts fire on drift, error rate, latency breaches |

```bash
# Live demo
make serve                        # Terminal 1
python scripts/inject_failure.py  # Terminal 2
```

See [evals/failure_report.md](evals/failure_report.md) for full documentation.

---

## 📈 Monitoring

### 5 Dashboard Metrics (R3 requirement)

1. **Request rate** — predictions per minute
2. **Error rate** — by predicted class
3. **Latency** — p50, p95, p99 percentiles
4. **PSI drift score** — prediction distribution shift
5. **Active model version** — for lineage

### Alert Rules

| Rule | Fires when |
|---|---|
| Drift | PSI ≥ 0.20 |
| Error rate | ≥ 5% |
| Latency | p95 ≥ 200ms |

---

## 🧪 Tests

```bash
make test   # Run all 81 tests
```

| Suite | Tests | Coverage |
|---|---|---|
| `test_model.py` | 12 | Model instantiation, shapes, gradients, checkpoints |
| `test_data.py` | 13 | Loading, leakage, splits, transforms, class distribution |
| `test_service.py` | 13 | All API endpoints, schemas, error handling |
| `test_drift.py` | 24 | PSI, KS, drift monitor, alerts, metrics, SLO |
| `test_failure.py` | 19 | 6 attack vectors: truncated, blank, non-image, batch |
| **Total** | **81** | |

---

## 🐳 Docker

```bash
# Build (SHA-tagged, digest-pinned base)
make image-build

# Run locally
make image-run
# → http://localhost:7860 (Gradio) or http://localhost:8000 (FastAPI)

# Push to Artifact Registry
make image-push
```

---

## ☁️ Cloud Deployment (GCP)

```bash
# 1. Copy and fill environment contract
cp cloud.env.example cloud.env

# 2. Verify all slots
make cloud-check

# 3. Push and deploy
git push origin main  # Triggers CI → CD → Cloud Run
```

The CD pipeline uses **Workload Identity Federation** (no long-lived keys).

---

## 💰 Cost

| Total estimated | Budget | Status |
|---|---|---|
| ~340 THB/term | 800 THB | ✅ 43% of budget |

See [docs/cost_report.md](docs/cost_report.md) for detailed breakdown.

---

## 📋 Documentation

| Document | Purpose |
|---|---|
| [docs/proposal.md](docs/proposal.md) | Capstone proposal (M2 milestone) |
| [docs/model_card.md](docs/model_card.md) | Model Card (Mitchell et al. 2019) |
| [docs/cost_report.md](docs/cost_report.md) | Honest cost breakdown with teardown checklist |
| [evals/failure_report.md](evals/failure_report.md) | Failure engineering: with/without protection |
| [loadtest/README.md](loadtest/README.md) | Pre-declared latency target + results |

---

## 📂 Dataset

| Attribute | Value |
|---|---|
| **Source** | [FracAtlas](https://www.nature.com/articles/s41597-023-02432-4) |
| **Licence** | CC BY-SA 4.0 |
| **Total images** | 4,083 |
| **Fractured** | 717 (17.6%) |
| **Non-Fractured** | 3,366 (82.4%) |
| **Imbalance** | 4.7:1 → inverse-frequency weighted loss |

---

## 🔧 Make Targets

```bash
make help              # Show all targets
make reproduce         # One-command full reproduction
make train             # Train with MLflow
make test              # Run all 81 tests
make portability-audit # Enforce 3-layer rule
make serve             # Start FastAPI server
make image-build       # Build Docker image
make cloud-check       # Verify cloud.env
make cost-report       # Print cost estimate
```

---

## 👥 Team

| Student ID | Name | Owns |
|---|---|---|
| 6688249 | Sahatsawat Nitjaphant | MLOps infra: Makefile, Docker, DVC, MLflow, CI/CD, cloud adapter, monitoring |
| 6688093 | Ongsa Raksalam | ML pipeline: training, evaluation, experiment tracking, model registry |
| 6688232 | Xinyi Chen | Serving & reliability: FastAPI, load testing, failure engineering, tests |