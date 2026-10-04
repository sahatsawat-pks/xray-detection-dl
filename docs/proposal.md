# ITCS355 Capstone Proposal — Bone Fracture X-Ray Detection

**Team**: Sahatsawat Nitjaphant (6688249), Ongsa Raksalam (6688093), Thanadon Yindeesuk (6688152)

---

## Problem & Who Uses It

Bone fractures are among the most common musculoskeletal injuries seen in emergency departments. Manual X-ray reading under high workload is time-consuming and error-prone — missed fractures (false negatives) lead to delayed treatment and worse patient outcomes.

We build an **AI-assisted triage service** that accepts an X-ray image and returns a fracture/no-fracture classification with confidence score. The target user is a radiologist or ED physician who uses the prediction as a second opinion, **not** a replacement for clinical judgment. The service prioritises **high recall** — a missed fracture is more dangerous than a false alarm.

## Dataset & Licence

| Attribute | Value |
|---|---|
| **Name** | [FracAtlas](https://www.nature.com/articles/s41597-023-02432-4) |
| **Source** | Kaggle / Nature Scientific Data |
| **Licence** | CC BY-SA 4.0 |
| **Total images** | 4,083 X-rays |
| **Fractured** | 717 (17.6%) |
| **Non-Fractured** | 3,366 (82.4%) |
| **Body parts** | Leg (2,273), Hand (1,538), Shoulder (349), Hip (338) |
| **Challenge** | 4.7:1 class imbalance → addressed with inverse-frequency weighted loss |

## Latency & Freshness Requirements

| Requirement | Target |
|---|---|
| **Serving pattern** | Online (real-time, single image per request) |
| **Latency SLO** | p95 < 200 ms at 10 concurrent users |
| **Freshness** | Model retrained on demand (no scheduled retraining — fracture patterns don't drift seasonally) |
| **Availability SLO** | 99% uptime during operating hours |

## Serving Pattern & Justification

**Online inference via FastAPI** (containerised on Cloud Run / Vertex AI Endpoint).

Rationale: Clinical triage is inherently real-time — a radiologist uploads one X-ray and needs an answer within seconds. Batch scoring would add unacceptable delay. The 224×224 input and single EfficientNet-B0 forward pass is lightweight enough for CPU-only inference, keeping costs low and avoiding GPU quota issues.

The existing Gradio/Streamlit demo apps are kept as patient-facing UI prototypes but are **not** the production serving path.

## Planned Failure Mode

**Corrupted Image Injection** — simulating a faulty PACS (Picture Archiving and Communication System) feed.

| Phase | What happens |
|---|---|
| **Inject** | Send truncated JPEGs, non-medical images (document photos), and all-black frames to `/predict` |
| **Without protection** | Model silently predicts with high confidence on garbage — clinically dangerous |
| **With protection** | Input validation (file integrity, entropy check), confidence thresholding (< 0.6 → "uncertain"), anomaly alerting |
| **Fed back into tests** | `tests/test_failure.py` asserts corrupted inputs are rejected or flagged, runs in CI |
| **Demo** | Live injection during presentation → service rejects gracefully → alert fires in dashboard |

## Rough Cost Estimate

| Resource | Est. Monthly Cost |
|---|---|
| Cloud Run (CPU-only, scale-to-zero) | ~50 THB (pay per request) |
| Cloud Storage (DVC data + model artifacts) | ~10 THB |
| Artifact Registry (Docker images) | ~5 THB |
| MLflow server (Cloud Run, minimal) | ~20 THB |
| Cloud Monitoring (metrics + 1 alert) | Free tier |
| **Total estimated** | **~85 THB/month** |
| **Cost per 1,000 predictions** | ~0.18 THB (Cloud Run 256 MiB, ~200 ms/req) |

Well under the 800 THB term budget.

## Team Ownership

| Member | Owns |
|---|---|
| **Sahatsawat (6688249)** | MLOps infrastructure: Makefile, Docker, DVC, MLflow tracking, CI/CD, cloud adapter, monitoring/drift, cost report |
| **Ongsa (6688093)** | ML pipeline: model training, evaluation, experiment tracking, model registry, hyperparameter study |
| **Thanadon (6688152)** | Serving & reliability: FastAPI service, load testing, failure engineering, input validation, tests, model card |
