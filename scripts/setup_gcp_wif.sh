#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# setup_gcp_wif.sh — Configure Workload Identity Federation (WIF) for GitHub Actions
# ═══════════════════════════════════════════════════════════════════════════════
# This script configures GCP Workload Identity Federation so GitHub Actions can
# deploy to Cloud Run securely WITHOUT long-lived service account keys
# (avoiding automatic course deductions).
#
# Usage:
#   bash scripts/setup_gcp_wif.sh

set -euo pipefail

PROJECT_ID="itcs355-6688249"
POOL_NAME="github-pool"
PROVIDER_NAME="github-provider"
SA_NAME="github-actions-sa"
GITHUB_REPO="sahatsawat-pks/xray-detection-dl"

echo "=== Setting up Workload Identity Federation for GCP ==="
echo "Project ID:   ${PROJECT_ID}"
echo "GitHub Repo:  ${GITHUB_REPO}"
echo

# 1. Enable required GCP services
echo "[1/6] Enabling required GCP APIs..."
gcloud services enable \
  iamcredentials.googleapis.com \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  --project="${PROJECT_ID}"

# 2. Get Project Number
PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)")
echo "      Project Number: ${PROJECT_NUMBER}"

# 3. Create Service Account (if not exists)
echo "[2/6] Ensuring Service Account exists..."
if ! gcloud iam service-accounts describe "${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" --project="${PROJECT_ID}" &>/dev/null; then
  gcloud iam service-accounts create "${SA_NAME}" \
    --display-name="GitHub Actions CD Service Account" \
    --project="${PROJECT_ID}"
  echo "      Created service account ${SA_NAME}"
else
  echo "      Service account ${SA_NAME} already exists"
fi

# 4. Create Workload Identity Pool (if not exists)
echo "[3/6] Ensuring Workload Identity Pool exists..."
if ! gcloud iam workload-identity-pools describe "${POOL_NAME}" --project="${PROJECT_ID}" --location="global" &>/dev/null; then
  gcloud iam workload-identity-pools create "${POOL_NAME}" \
    --project="${PROJECT_ID}" \
    --location="global" \
    --display-name="GitHub Actions Pool"
  echo "      Created pool ${POOL_NAME}"
else
  echo "      Pool ${POOL_NAME} already exists"
fi

# 5. Create Workload Identity Provider (if not exists)
echo "[4/6] Ensuring Workload Identity Provider exists..."
if ! gcloud iam workload-identity-pools providers describe "${PROVIDER_NAME}" \
  --project="${PROJECT_ID}" \
  --location="global" \
  --workload-identity-pool="${POOL_NAME}" &>/dev/null; then
  gcloud iam workload-identity-pools providers create-oidc "${PROVIDER_NAME}" \
    --project="${PROJECT_ID}" \
    --location="global" \
    --workload-identity-pool="${POOL_NAME}" \
    --display-name="GitHub Actions Provider" \
    --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository" \
    --attribute-condition="assertion.repository == '${GITHUB_REPO}'" \
    --issuer-uri="https://token.actions.githubusercontent.com"
  echo "      Created provider ${PROVIDER_NAME}"
else
  echo "      Provider ${PROVIDER_NAME} already exists"
fi

# 6. Allow GitHub Actions repository to impersonate the Service Account
echo "[5/6] Binding GitHub Actions repo to Service Account..."
gcloud iam service-accounts add-iam-policy-binding \
  "${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --project="${PROJECT_ID}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_NAME}/attribute.repository/${GITHUB_REPO}"

# 7. Grant necessary permissions to the Service Account
echo "[6/6] Granting IAM roles to Service Account..."
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/run.admin" --condition=None --quiet >/dev/null

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer" --condition=None --quiet >/dev/null

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser" --condition=None --quiet >/dev/null

# Create Artifact Registry repo for xray if not exists
gcloud artifacts repositories create xray \
  --repository-format=docker \
  --location=asia-southeast1 \
  --description="X-ray Docker repository" \
  --project="${PROJECT_ID}" 2>/dev/null || true

WIF_PROVIDER="projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_NAME}/providers/${PROVIDER_NAME}"
WIF_SERVICE_ACCOUNT="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo
echo "=========================================================================="
echo "✅ SUCCESS! Add these 3 secrets to your GitHub repository:"
echo "   Go to: https://github.com/${GITHUB_REPO}/settings/secrets/actions"
echo "=========================================================================="
echo
echo "1. GCP_PROJECT_ID"
echo "   ${PROJECT_ID}"
echo
echo "2. WIF_PROVIDER"
echo "   ${WIF_PROVIDER}"
echo
echo "3. WIF_SERVICE_ACCOUNT"
echo "   ${WIF_SERVICE_ACCOUNT}"
echo
echo "=========================================================================="
