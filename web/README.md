# Web UI — Bone Fracture X-Ray Detection (Vercel Deployment)

This directory contains the production web interface for the ITCS355 Capstone, designed to be deployed on **Vercel**.

## Architecture

- **Frontend (UI)**: Hosted on **Vercel** (global edge CDN, instant loading, drag-and-drop radiograph uploader).
- **Backend (AI Inference)**: Hosted on **Google Cloud Run** (`https://xray-fracture-predict-24jrf436va-as.a.run.app`).

## Deploying to Vercel (2 Options)

### Option 1: Via Vercel Web Dashboard (Easiest)
1. Go to [https://vercel.com/new](https://vercel.com/new).
2. Select your GitHub repository: `sahatsawat-pks/xray-detection-dl`.
3. In **Build & Output Settings**:
   - **Framework Preset**: Other
   - **Root Directory**: `web` (or leave default since root `vercel.json` points to `web`).
4. Click **Deploy**. Done!

### Option 2: Via Vercel CLI
```bash
# In the repository root
npx vercel
```
