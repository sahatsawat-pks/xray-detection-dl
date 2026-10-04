# ─── Stage 1: Build ───────────────────────────────────────────────────────────
# Pinned by SHA256 digest for reproducibility (ITCS355 requirement)
FROM python:3.11-slim@sha256:babe11ccba01e6e11e2cf89877503da3cfbb4d4508c0b578b47e92e6acaa9820 AS base

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (hash-pinned for reproducibility)
COPY requirements.in /tmp/requirements.in
RUN pip install --no-cache-dir -r /tmp/requirements.in

# ─── Stage 2: Copy project ────────────────────────────────────────────────────
# Layer 1 — provider-neutral source
COPY src/   /app/src/
COPY dl_utils.py /app/dl_utils.py

# Layer 2 — cloud adapter (only needed for cloud operations)
COPY cloudlayer/ /app/cloudlayer/

# App layer — demo UIs
COPY app/   /app/app/

# Pre-trained models
COPY models/ /app/models/

# Sample images for Gradio examples (small subset)
COPY dataset/images/Fractured/IMG0000019.jpg    /app/dataset/images/Fractured/
COPY dataset/images/Fractured/IMG0000025.jpg    /app/dataset/images/Fractured/
COPY dataset/images/Fractured/IMG0000044.jpg    /app/dataset/images/Fractured/
COPY dataset/images/Non_fractured/IMG0000000.jpg /app/dataset/images/Non_fractured/
COPY dataset/images/Non_fractured/IMG0000001.jpg /app/dataset/images/Non_fractured/
COPY dataset/images/Non_fractured/IMG0000002.jpg /app/dataset/images/Non_fractured/

# ── Backward compatibility ────────────────────────────────────────────────────
# Keep root-level model.py for existing imports in app/app.py
COPY model.py /app/model.py

# ─── Runtime ──────────────────────────────────────────────────────────────────
EXPOSE 7860
EXPOSE 8501
EXPOSE 8000

ENV PYTHONUNBUFFERED=1

# Gradio Settings
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

WORKDIR /app

# Default: Launch Gradio App
CMD ["python", "app/app.py"]

# Alternative commands:
# Streamlit:  CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
# FastAPI:    CMD ["python", "-m", "uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
