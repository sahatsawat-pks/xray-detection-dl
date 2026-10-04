# ─── Stage 1: Build ───────────────────────────────────────────────────────────
# Pinned by SHA256 digest for reproducibility (ITCS355 requirement)
FROM python:3.11-slim@sha256:6f31d6e9ba2b0a787a3f81c37b004155b87b9efa1b771182bd550c1615745be5 AS base

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.in /tmp/requirements.in
RUN pip install --no-cache-dir -r /tmp/requirements.in

# ─── Stage 2: Copy project ────────────────────────────────────────────────────
# Layer 1 — provider-neutral source
COPY src/        /app/src/
COPY dl_utils.py /app/dl_utils.py

# Layer 2 — cloud adapter (only needed for cloud operations)
COPY cloudlayer/ /app/cloudlayer/

# Layer 3 — FastAPI production inference service
COPY service/    /app/service/

# Monitoring & observability
COPY monitoring/ /app/monitoring/

# Pre-trained models
COPY models/     /app/models/

# App layer — demo UIs (optional)
COPY app/        /app/app/
COPY model.py    /app/model.py

# Sample images for demo predictions
COPY dataset/images/Fractured/IMG0000019.jpg    /app/dataset/images/Fractured/
COPY dataset/images/Fractured/IMG0000025.jpg    /app/dataset/images/Fractured/
COPY dataset/images/Fractured/IMG0000044.jpg    /app/dataset/images/Fractured/
COPY dataset/images/Non_fractured/IMG0000000.jpg /app/dataset/images/Non_fractured/
COPY dataset/images/Non_fractured/IMG0000001.jpg /app/dataset/images/Non_fractured/
COPY dataset/images/Non_fractured/IMG0000002.jpg /app/dataset/images/Non_fractured/

# ─── Runtime ──────────────────────────────────────────────────────────────────
EXPOSE 8000
EXPOSE 7860

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

WORKDIR /app

# Production Default: FastAPI Inference Service on port 8000 (serves /health, /ready, /predict)
CMD ["python", "-m", "uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
