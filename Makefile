# ═══════════════════════════════════════════════════════════════════════════════
# Makefile — ITCS355 Capstone: Bone Fracture X-Ray Detection
# ═══════════════════════════════════════════════════════════════════════════════
# Central command interface. Every operation goes through here.
# Usage: make <target>

SHELL      := /bin/bash
PYTHON     := python3
PIP        := pip
PYTEST     := $(PYTHON) -m pytest
RUFF       := $(PYTHON) -m ruff
DOCKER     := docker

IMAGE_NAME := fracture-detector
IMAGE_TAG  := $(shell git rev-parse --short HEAD 2>/dev/null || echo "dev")
PLATFORM   := linux/amd64

.DEFAULT_GOAL := help

# ── Help ──────────────────────────────────────────────────────────────────────
.PHONY: help
help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ═══════════════════════════════════════════════════════════════════════════════
# Setup & Data
# ═══════════════════════════════════════════════════════════════════════════════
.PHONY: setup
setup:  ## Create venv and install pinned dependencies
	$(PYTHON) -m venv .venv || true
	. .venv/bin/activate && $(PIP) install --upgrade pip
	. .venv/bin/activate && $(PIP) install -r requirements.in
	@echo "✅ Setup complete"

.PHONY: data
data:  ## Pull dataset via DVC (or verify it exists)
	@if command -v dvc &>/dev/null && [ -f dataset/images.dvc ]; then \
		dvc pull; \
	elif [ -d dataset/images ]; then \
		echo "✅ Dataset already present at dataset/images/"; \
	else \
		echo "❌ Dataset not found. Place images in dataset/images/ or run dvc pull."; \
		exit 1; \
	fi

# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════
.PHONY: train
train:  ## Run training with MLflow tracking
	$(PYTHON) -m src.train

.PHONY: train-model
train-model:  ## Train a specific model: make train-model MODEL=ModelFinal
	$(PYTHON) -m src.train --model $(MODEL)

# ═══════════════════════════════════════════════════════════════════════════════
# Reproduce & Verify
# ═══════════════════════════════════════════════════════════════════════════════
.PHONY: reproduce
reproduce: setup data train verify  ## One-command full reproduction
	@echo "✅ Reproduction complete"

.PHONY: verify
verify:  ## Compare latest metrics against README claim
	$(PYTHON) scripts/verify_metric.py

.PHONY: gate
gate:  ## Run evaluation gate on candidate model
	$(PYTHON) scripts/evaluation_gate.py --metrics results/ModelFinal_results.json --metric AUC --floor 0.85

.PHONY: register
register:  ## Register trained model in MLflow model registry
	$(PYTHON) scripts/register.py

# ═══════════════════════════════════════════════════════════════════════════════
# Quality Gates
# ═══════════════════════════════════════════════════════════════════════════════
.PHONY: lint
lint:  ## Lint source code with ruff
	$(RUFF) check src/ tests/ service/ || true
	@echo "✅ Lint complete"

.PHONY: test
test:  ## Run all pytest suites
	$(PYTEST) tests/ -v --tb=short
	@echo "✅ All tests passed"

.PHONY: test-unit
test-unit:  ## Run unit tests only (models)
	$(PYTEST) tests/test_model.py -v --tb=short

.PHONY: test-data
test-data:  ## Run data contract tests only
	$(PYTEST) tests/test_data.py -v --tb=short

.PHONY: portability-audit
portability-audit:  ## Enforce 3-layer rule: no cloud SDK in src/ or tests/
	@bash scripts/portability_audit.sh

# ═══════════════════════════════════════════════════════════════════════════════
# Docker
# ═══════════════════════════════════════════════════════════════════════════════
.PHONY: image-build
image-build:  ## Build Docker image (linux/amd64, SHA-tagged)
	$(DOCKER) build --platform $(PLATFORM) \
		-t $(IMAGE_NAME):$(IMAGE_TAG) \
		-t $(IMAGE_NAME):latest \
		.
	@echo "✅ Built $(IMAGE_NAME):$(IMAGE_TAG)"

.PHONY: image-push
image-push: image-build  ## Push Docker image to registry
	@if [ -z "$(CONTAINER_REGISTRY)" ]; then \
		echo "❌ CONTAINER_REGISTRY not set in cloud.env"; exit 1; \
	fi
	$(DOCKER) tag $(IMAGE_NAME):$(IMAGE_TAG) $(CONTAINER_REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)
	$(DOCKER) push $(CONTAINER_REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)
	@echo "✅ Pushed to $(CONTAINER_REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)"

.PHONY: image-run
image-run:  ## Run the Docker image locally
	$(DOCKER) run -p 7860:7860 -p 8000:8000 $(IMAGE_NAME):$(IMAGE_TAG)

# ═══════════════════════════════════════════════════════════════════════════════
# Cloud
# ═══════════════════════════════════════════════════════════════════════════════
.PHONY: cloud-check
cloud-check:  ## Verify cloud.env has all required slots
	@$(PYTHON) scripts/cloud_check.py

.PHONY: mlflow-ui
mlflow-ui:  ## Launch MLflow tracking UI
	mlflow ui --port 5000

.PHONY: teardown
teardown:  ## Delete tagged cloud resources
	@echo "⚠️  Teardown: check Cloud Console for resources tagged course=itcs355"
	@echo "   Run: gcloud resource-manager tags list --project=$$PROJECT_ID"

.PHONY: cost-report
cost-report:  ## Print cost estimate
	@echo "=== Cost Report ==="
	@echo "See docs/cost_report.md for detailed breakdown"

# ═══════════════════════════════════════════════════════════════════════════════
# Serving
# ═══════════════════════════════════════════════════════════════════════════════
.PHONY: serve
serve:  ## Run FastAPI serving locally
	$(PYTHON) -m uvicorn service.app:app --host 0.0.0.0 --port 8000 --reload

.PHONY: serve-gradio
serve-gradio:  ## Run Gradio demo app
	cd app && $(PYTHON) app.py

# ═══════════════════════════════════════════════════════════════════════════════
# Credential Safety
# ═══════════════════════════════════════════════════════════════════════════════
.PHONY: credential-check
credential-check:  ## Scan git history for leaked credentials
	@echo "=== Credential Scan ==="
	@found=$$(git log -p 2>/dev/null | grep -E "AKIA[0-9A-Z]{16}|BEGIN [A-Z ]*PRIVATE KEY|client_secret\s*=\s*['\"][^'\"]+['\"]" | head -5); \
	if [ -n "$$found" ]; then \
		echo "$$found"; \
		echo "❌ FAIL — Credentials found in git history!"; exit 1; \
	else \
		echo "✅ PASS — No credentials found"; \
	fi
