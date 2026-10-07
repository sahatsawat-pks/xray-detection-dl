# ITCS355 Capstone — Slide Presentation Deck (M4 Defense)

> **Presentation Duration**: 8 minutes presentation + 5 minutes Q&A  
> **Target Score**: 15 Marks (Capstone Demo & Defense M4)  
> **Team**: Sahatsawat Nitjaphant (6688249), Ongsa Raksalam (6688093), Xinyi Chen (6688232)  
> **Live Endpoint**: `https://xray-fracture-predict-24jrf436va-as.a.run.app`

---

## Slide 1: Title & Capstone Overview
- **Visual**: Dark clinical UI background, FracNet AI logo, architectural diagram badge, live Cloud Run status badge.
- **Header**: FracNet AI — Automated Bone Fracture Triage System
- **Subtitle**: End-to-End MLOps Pipeline, Reliable Serving, and Engineered Failure Defense
- **Team**: Sahatsawat Nitjaphant (6688249) · Ongsa Raksalam (6688093) · Xinyi Chen (6688232)
- **Course**: ITCS355 Machine Learning Operation and Deployment
- **Bullets**:
  - Production-ready musculoskeletal radiograph triage service.
  - Strict 3-layer portability architecture with 0 cloud SDK leakage.
  - Scale-to-zero serverless deployment on Google Cloud Run.
  - Explainable AI with Grad-CAM visual fracture localization.
- **Speaker Notes (0:00 – 0:30, Xinyi)**:
  > *"Good morning instructor and classmates. We are Team FracNet AI. Today we present our end-to-end MLOps Capstone system for automated bone fracture triage. Our goal was not just to train a neural network, but to build an operationally complete, portable, and failure-resilient cloud service that clinicians can actually trust."*

---

## Slide 2: The Problem in 60s
- **Visual**: Hospital emergency room workflow diagram contrasting standard delayed radiologist queue vs. FracNet automated priority queue. FracAtlas dataset breakdown chart (4,083 scans, 4.7:1 imbalance).
- **Header**: Emergency Triage Bottleneck & Clinical Reliability
- **Bullets**:
  - **The Bottleneck**: Emergency departments experience severe delays during peak hours waiting for expert radiologist review.
  - **Asymmetric Risk**: A false negative (missing a hairline fracture) leads to bone malunion, disability, and malpractice risk.
  - **Data Reality**: FracAtlas dataset contains 4,083 images with severe 4.7:1 class imbalance (only 17.5% fractures).
  - **Operational Requirement**: We need high recall triage, strict latency under 500 ms, zero crashes on corrupt inputs, and visual explainability.
- **Speaker Notes (0:30 – 1:15, Xinyi)**:
  > *"In an emergency room, radiologist queues can take hours. Missing a fracture causes lifelong patient disability. However, existing academic AI implementations fail in practice: when hospital networks drop packets, models either crash or emit high-confidence garbage. Our system solves this by combining high-recall triage with robust input validation, ensuring clean and corrupted scans are handled safely."*

---

## Slide 3: 3-Layer Architecture & Portability Audit
- **Visual**: Clear 3-tier architectural diagram highlighting Layer 1 (`src/`), Layer 2 (`cloudlayer/`), and Layer 3 (`service/`). Terminal screenshot showing `make portability-audit` passing with 0 cloud references.
- **Header**: The Three-Layer Software Engineering Contract
- **Bullets**:
  - **Layer 1 (`src/`)**: Pure PyTorch ML logic, preprocessing, and MLflow lineage tracking. **Zero cloud provider SDKs.**
  - **Layer 2 (`cloudlayer/`)**: Provider-neutral `CloudAdapter` ABC with pluggable implementations (`gcp.py`). The *only* location where cloud SDKs exist.
  - **Layer 3 (`service/` & `web/`)**: FastAPI serving with decoupled `/health` & `/ready` probes, plus modern web client.
  - **Automated Audit**: Enforced via `scripts/portability_audit.sh` across all CI commits.
- **Speaker Notes (1:15 – 2:15, Sahatsawat)**:
  > *"To ensure portability, we strictly separated our code into three distinct layers. Layer 1 contains our model and training code and has zero dependencies on Google Cloud. Layer 2 is our cloud adapter, implementing 11 standard operations. Layer 3 is our FastAPI service. Running 'make portability-audit' proves that not a single AWS, Azure, or GCP SDK is imported in Layer 1 or our test suite. We can swap cloud providers by changing one adapter file."*

---

## Slide 4: Automated ML Pipeline & Lineage Tracking
- **Visual**: MLflow run dashboard screenshot showing logged parameters, artifacts, DVC hash, git SHA, and metric plots. Model progression comparison table.
- **Header**: Reproducible Training Loop & Honest Metrics
- **Bullets**:
  - **Complete Lineage**: Every training run automatically records Git Commit SHA, DVC data MD5, Docker digest, and random seed (42).
  - **Model Progression**:
    - *ModelBaseline*: 4-block CNN from scratch (Val Loss: 0.542, Test AUC: 0.824).
    - *ModelImproved*: ResNet-18 fine-tuned (Val Loss: 0.412, Test AUC: 0.887).
    - *ModelFinal*: EfficientNet-B0 compound scaled (Val Loss: 0.315, Test AUC: 0.913).
  - **Automated Verification**: `make verify` guarantees claimed metrics in README match actual test evaluations ($\Delta = 0.0000$).
  - **Deployment Gate**: `make gate` halts CD if candidate AUC falls below the 0.8500 quality floor.
- **Speaker Notes (2:15 – 3:15, Xinyi)**:
  > *"Every artifact is fully tracked in MLflow. We don't just commit weights; we record the exact Git commit, dataset fingerprint, and container digest. We trained three iterations, ultimately selecting EfficientNet-B0 for its compound scaling. Most importantly, our claimed metrics are completely honest: running 'make verify' programmatically proves our claimed AUC of 0.9130 matches ground truth with zero discrepancy, passing our automated 0.85 evaluation gate."*

---

## Slide 5: Cloud Deployment & CI/CD Automation
- **Visual**: GitHub Actions pipeline diagram (Lint $\rightarrow$ Audit $\rightarrow$ Pytest $\rightarrow$ Build $\rightarrow$ WIF Auth $\rightarrow$ Artifact Registry $\rightarrow$ Cloud Run $\rightarrow$ Smoke Test). WIF diagram showing passwordless token exchange.
- **Header**: Zero-Secret CI/CD to Serverless Cloud Run
- **Bullets**:
  - **Digest-Pinned Docker**: Base image pinned by SHA256 digest (`python:3.11-slim@sha256:6f31...`) preventing upstream silent breaking changes.
  - **Workload Identity Federation (WIF)**: Automated deployment without committed service account JSON keys. GitHub OIDC tokens exchange directly with GCP STS.
  - **Cloud Run Configuration**:
    - Region: `asia-southeast1` (Singapore).
    - 1 vCPU, 1 GiB Memory, 120s timeout.
    - Scale-to-zero (0 to 3 instances) — **zero idle cost**.
  - **Quality Gates in CI**: 82 automated pytest tests, Ruff linting, and smoke tests run before traffic is served.
- **Speaker Notes (3:15 – 4:15, Sahatsawat)**:
  > *"For deployment, we used GitHub Actions with Google Workload Identity Federation. We never committed a private key or password to Git. On every merge to main, our pipeline audits code portability, runs 82 pytest tests, builds a digest-pinned Docker container, and deploys it to Cloud Run in Singapore. The service scales to zero when idle, meaning zero wasted budget."*

---

## Slide 6: Live System Demo & Explainability
- **Visual**: Split screen: Left side shows live Web Dashboard with drag-and-drop X-ray and Grad-CAM heatmap toggle; Right side shows terminal running `curl` and Swagger UI at `/docs`.
- **Header**: Live Serving & Grad-CAM Fracture Localization
- **Bullets**:
  - **Liveness vs Readiness**: `/health` confirms process uptime; `/ready` verifies model weights loaded in memory.
  - **Sub-500ms Online Inference**: Rapid binary prediction with confidence score and lineage version tag.
  - **Explainable AI (Grad-CAM)**:
    - Backpropagates class gradients to the final convolutional block (`features[-1]`).
    - Generates a visual heat attention map overlaid on the patient radiograph.
    - Clinicians can toggle between the raw X-ray and the localized fracture heatmap.
- **Speaker Notes (4:15 – 5:30, Ongsa)**:
  > *"Let's see the live service running on Cloud Run. In our web interface, notice our decoupled health and readiness probes. When we submit a radiograph, the model diagnoses a bone fracture with 99.7% confidence in under 100 milliseconds. But clinicians don't trust black boxes: with our Grad-CAM explainability feature, we can toggle the attention heatmap to reveal the glowing highlight directly over the fracture fissure, proving the model is looking at the injury, not image background artifacts."*

---

## Slide 7: Engineered Failure Defense (Faulty PACS)
- **Visual**: 4-Layer Defense Chain flowchart. Terminal output of `scripts/inject_failure.py` showing rejected vs accepted attacks. Formula of Shannon Entropy.
- **Header**: Engineered Failure Mode: Corrupted PACS Injection
- **Bullets**:
  - **The Attack Scenario**: Simulated a faulty hospital PACS sending truncated network packets, empty files, and dead detector screens.
  - **4-Layer Protection Chain**:
    1. *File Format Gate*: Intercepts non-images and truncated headers $\rightarrow$ `HTTP 422`.
    2. *Shannon Entropy Gate*: Detects blank, solid-color, or dead frames ($H < 1.0$) $\rightarrow$ `HTTP 422`.
    3. *Neural Processing*: Robust PyTorch forward pass.
    4. *Confidence Triage Gate*: Conf $< 0.60 \rightarrow$ flags `"uncertain": true` for human review.
  - **Live Injection Demo**: 7 invalid inputs safely rejected; 2 out-of-distribution inputs flagged as uncertain; **0 crashes**.
- **Speaker Notes (5:30 – 6:45, Xinyi)**:
  > *"The differentiator in our capstone is our deliberate engineered failure. We simulated a malfunctioning PACS feed injecting 13 different attack vectors: dead detector frames, truncated byte streams, and corrupted files. Instead of crashing with a 500 error, our 4-layer defense chain intercepts them. We calculate Shannon entropy on pixel distributions: if a dead detector sends an all-black frame with entropy under 1.0, it is immediately rejected with HTTP 422. Non-medical images are flagged as uncertain, ensuring clinicians are never misled."*

---

## Slide 8: Observability, Metrics & Drift Alerting
- **Visual**: Telemetry dashboard showing rolling request rate, p95 latency, error rate, and PSI distribution chart. AlertManager cooldown notification flow.
- **Header**: Real-Time Telemetry & Cooldown Alerting
- **Bullets**:
  - **5 Dashboard Metrics (`/metrics` & `/slo`)**:
    1. Request rate (total & RPS).
    2. Latency percentiles (p50, p90, p95, p99).
    3. HTTP error rate.
    4. Rolling confidence and uncertainty fraction.
    5. Statistical drift scores.
  - **Drift Algorithms**: Population Stability Index (PSI) and Kolmogorov-Smirnov test comparing real-time inference vs validation baseline.
  - **Cooldown Deduplication**: 15-minute alert cooldown prevents alarm fatigue during continuous incidents.
- **Speaker Notes (6:45 – 7:15, Xinyi)**:
  > *"Our service tracks real-time operations through Prometheus-compatible endpoints. We monitor latency SLOs and statistical drift using Population Stability Index and KS tests. If image distribution drifts, an alert triggers. Furthermore, our AlertManager implements a 15-minute cooldown deduplication window so on-call engineers aren't flooded with duplicate alerts."*

---

## Slide 9: FinOps Cost Report & Budget Breakdown
- **Visual**: GCP cost breakdown donut chart and monthly projection table comparing actual spend (~340 THB) against the 800 THB budget.
- **Header**: FinOps Cost Report & Budget Discipline
- **Bullets**:
  - **Budget Cap**: 800 THB limit for the term.
  - **Actual Estimated Spend**: **~340 THB total** (42.5% utilization).
  - **Unit Economics**: **~0.19 THB per 1,000 predictions**.
  - **Why So Efficient?**:
    - EfficientNet-B0 (5.3M params) runs on CPU (1 vCPU, 512 MiB RAM) in ~100 ms.
    - Zero expensive GPU VMs rented.
    - Cloud Run scales to zero instances when idle.
  - **Safe Teardown**: Documented checklist and script to destroy all GCP resources upon course completion.
- **Speaker Notes (7:15 – 7:45, Sahatsawat)**:
  > *"In MLOps, cost discipline is critical. We were allocated an 800 THB budget. By choosing an EfficientNet-B0 architecture that runs efficiently on CPU, we avoid expensive GPU instances. A thousand predictions cost only 0.19 THB. Our entire 4-month operational footprint is estimated at 340 THB, well below half our budget limit. When the semester ends, our teardown procedure cleanly removes all cloud storage and container images."*

---

## Slide 10: Roadmap & Defense Readiness
- **Visual**: Capstone architecture recap with green checkmarks over all 5 rubric criteria. "What Another Week Buys" roadmap list.
- **Header**: What Another Week Would Buy & Ready for Defense
- **Bullets**:
  - **Next Sprint Priorities**:
    1. *Canary / Shadow Deployments*: Cloud Run traffic splitting (10% Challenger, 90% Champion) for risk-free model rollouts.
    2. *Native DICOM Protocol Server*: Ingest DICOM objects directly over TCP port 104 from hospital PACS.
    3. *Continuous Training (CT)*: Automated Cloud Storage event triggers retraining pipelines on Vertex AI.
  - **Quality Summary**: 82/82 tests pass, 0 lint warnings, 0 portability leaks, verified metrics.
  - **Ready for Instructor Q&A!**
- **Speaker Notes (7:45 – 8:00, Sahatsawat)**:
  > *"If we had another week, we would implement canary traffic splitting in Cloud Run and direct DICOM TCP ingestion. In summary, our system passes all 82 tests, adheres to strict portability, and defends against corrupt inputs. Thank you, and we welcome your questions!"*
