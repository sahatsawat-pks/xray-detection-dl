# ITCS355 Capstone — Presentation & Defense Guide (15 Marks)

> **Format**: 8-minute presentation + 5-minute defense / live Q&A.  
> **Team**: Sahatsawat Nitjaphant (6688249), Ongsa Raksalam (6688093), Thanadon Yindeesuk (6688152).  
> **Grading Focus**: The operational system, reliability, failure handling, and costs. **Do not spend time on model selection — nobody is grading it.**

---

## ⏱️ 8-Minute Presentation Breakdown

| Section | Time | Speaker | Content / Visual |
|---|---|---|---|
| **1. The Problem in 60s** | 0:00 – 1:00 | Thanadon | Emergency room X-ray triage bottleneck; 4.7:1 class imbalance; high-recall requirement (false negative is fatal). |
| **2. Architecture Contract** | 1:00 – 2:30 | Sahatsawat | 3-layer architecture (`src/` neutral, `cloudlayer/` adapter, `service/` FastAPI). Show `make portability-audit` output. |
| **3. Live Service Demo** | 2:30 – 5:00 | Ongsa | Start `make serve`, show Swagger UI at `/docs`, execute `/health` and `/ready`, run a valid X-ray prediction. Show `/metrics` & `/slo`. |
| **4. Engineered Failure Mode** | 5:00 – 7:00 | Thanadon | The Corrupted PACS Injection scenario; run `python scripts/inject_failure.py`; demonstrate 4-layer protection chain & alert firing. |
| **5. Cost & What Another Week Buys** | 7:00 – 8:00 | Sahatsawat | ~0.19 THB per 1k predictions, 340 THB/term vs 800 THB budget; roadmap for next sprint. |

---

## 🖥️ Live Demo Cheatsheet

### Terminal Setup (Before Stepping Up)

Open two split terminal panes:
- **Pane 1 (Server)**:
  ```bash
  make serve
  ```
- **Pane 2 (Commands & Injection)**:
  Ready to run commands.
- **Browser Window**:
  Open:
  - `http://localhost:8000/docs` (FastAPI Swagger UI)
  - `http://localhost:8000/metrics`
  - `http://localhost:8000/alerts`

---

### Step-by-Step Demo Flow

#### 1. Prove Process Health & Model Readiness (30s)
- In browser or terminal:
  ```bash
  curl -s http://localhost:8000/health | jq
  curl -s http://localhost:8000/ready | jq
  ```
- **Say**: *"Notice that `/health` (liveness) and `/ready` (readiness) are decoupled. The service is alive immediately, but Kubernetes only routes prediction traffic once the model is warmed up in memory."*

#### 2. Run a Normal Inference (30s)
- In Swagger UI (`/docs`), open `POST /predict`. Upload a normal X-ray (`dataset/images/Fractured/IMG0000019.jpg`).
- Click **Execute**. Show response:
  ```json
  {
    "label": "Fractured",
    "confidence": 0.985,
    "fractured_probability": 0.985,
    "model_version": "local-dev",
    "model_name": "ModelFinal",
    "inference_time_ms": 42.5,
    "uncertain": false
  }
  ```
- **Highlight**: `model_version` tag ensures full lineage tracing back to the exact code and dataset commit.

#### 3. Live Engineered Failure Demo (90s)
- In Pane 2, run:
  ```bash
  python scripts/inject_failure.py
  ```
- Show the terminal output stepping through 9 attack vectors:
  - **Truncated JPEG** $\rightarrow$ 🛡️ `422 Unprocessable Entity` (Image decoding error)
  - **All-Black Frame** (Dead detector) $\rightarrow$ 🛡️ `422 Unprocessable Entity` (Entropy $< 1.0$)
  - **All-White Frame** (Overexposed) $\rightarrow$ 🛡️ `422 Unprocessable Entity` (Entropy $< 1.0$)
  - **Random Noise / Non-Medical Image** $\rightarrow$ ⚠️ `uncertain: true` ($< 0.60$ confidence flagged for review)
- In browser, refresh `http://localhost:8000/alerts`:
  - Show the drift / anomaly alert triggered by the injection!

---

## 🎯 Defending Against Unexpected Input (Instructor Trap)

> **Requirement from Course Brief**: *"Expect the instructor to send unexpected input during your demo; handling it gracefully scores, and crashing scores partial credit if your logging makes the cause obvious within a minute."*

If the instructor hands you a USB drive, sends a random file, or asks you to upload a non-X-ray (e.g. photo of a cat, PDF, word doc, or truncated file):

1. **Upload it directly to `POST /predict`** in Swagger UI with confidence.
2. **What will happen**:
   - If it's **non-image / corrupted**: The service immediately returns `HTTP 422` with a structured JSON error:
     ```json
     {"detail": "Image is corrupted or truncated: ..."}
     ```
   - If it's a **blank / solid image**: The service returns `HTTP 422`:
     ```json
     {"detail": "Image appears blank or degenerate (entropy=0.00)"}
     ```
   - If it's a **random photo**: It passes image decoding, but the entropy and softmax probability will either fail validation or return `"uncertain": true`.
3. **Point to Terminal 1**: Show the structured JSON log emitted with timestamp, error code, and reason.
4. **Say to the Instructor**:
   > *"Our service enforces a 4-layer defense chain: format parsing, dimension checks, Shannon entropy validation (catching dead detector frames), and confidence thresholding. Unprocessable inputs return HTTP 422 before reaching the tensor graph, avoiding silent failures."*

---

## 💰 Cost Defense (FinOps)

If asked about costs:
- **Cost per 1,000 predictions**: $\approx 0.19$ THB.
- **Why so cheap?**: EfficientNet-B0 requires only 5.3M parameters and runs on Cloud Run (1 vCPU, 512 MiB RAM) in under 100ms. No expensive GPU instances needed.
- **Idle cost**: **0 THB** (Cloud Run scales to zero instances when no traffic arrives).
- **Term total**: $\approx 340$ THB out of 800 THB budget (43% utilized).

---

## 🚀 "What Another Week Would Buy" (Closing Pitch)

1. **Shadow Deployments / Canary Routing**: Implement traffic-splitting in `service/` to test candidate models on live shadow traffic before promotion.
2. **DICOM Protocol Adapter**: Add native DICOM format ingestion directly interfacing with hospital PACS servers over TCP port 104.
3. **Automated Continuous Training (CT)**: Wire Cloud Storage bucket notifications to trigger Vertex AI training pipeline when new validated labels land.
