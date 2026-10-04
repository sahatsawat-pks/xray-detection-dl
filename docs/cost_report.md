# Cost Report — ITCS355 Capstone

> Budget limit: 800 THB/term. Honest reporting — inflated estimates are penalised.

## Current Spend

| Resource | Service | Monthly Est. | Term Est. (4 months) | Status |
|---|---|---|---|---|
| **Object Storage** | Cloud Storage (Standard) | ~10 THB | ~40 THB | Active |
| **Container Registry** | Artifact Registry | ~5 THB | ~20 THB | Active |
| **Inference Endpoint** | Cloud Run (CPU, scale-to-zero) | ~50 THB | ~200 THB | Active |
| **MLflow Server** | Cloud Run (256 MiB, min 0) | ~20 THB | ~80 THB | Optional |
| **Monitoring** | Cloud Monitoring | Free tier | Free | Active |
| **CI/CD** | GitHub Actions | Free tier (2,000 min/mo) | Free | Active |
| **Total estimated** | | **~85 THB/mo** | **~340 THB/term** | |

## Cost Breakdown

### Cloud Storage (~10 THB/month)
- Dataset images: ~2 GB (DVC-tracked)
- Model checkpoints: ~100 MB (3 models × ~25 MB each)
- MLflow artifacts: ~50 MB
- Standard storage: \$0.020/GB/month in `asia-southeast1`
- **Calculation**: 2.15 GB × \$0.020 = \$0.043/mo ≈ 1.5 THB. Rounded up for operations.

### Artifact Registry (~5 THB/month)
- Docker image: ~1.5 GB (Python 3.11-slim + PyTorch CPU)
- Storage: \$0.10/GB/month
- **Calculation**: 1.5 GB × \$0.10 = \$0.15/mo ≈ 5 THB

### Cloud Run — Inference (~50 THB/month)
- Configuration: 1 vCPU, 512 MiB RAM, min-instances=0
- Scale-to-zero when idle — **no cost when not serving**
- vCPU: \$0.00002400/vCPU-second
- Memory: \$0.00000250/GiB-second
- Estimated: ~2,000 requests/month during development/testing
- Per request: ~200ms × (vCPU + memory) ≈ \$0.000006
- **Calculation**: 2,000 × \$0.000006 = \$0.012/mo + \$1 startup overhead ≈ 50 THB

### MLflow Server (~20 THB/month)
- Cloud Run service with 256 MiB RAM, min-instances=0
- Only runs during active development
- **Can be run locally to save costs** — `make mlflow-ui`

### Free Tier Resources
- **Cloud Monitoring**: First 150 MB of metrics ingestion free
- **GitHub Actions**: 2,000 minutes/month free for public repos
- **Cloud Build**: 120 build-minutes/day free

## Cost per 1,000 Predictions

| Component | Cost per 1K requests |
|---|---|
| Cloud Run compute | ~0.18 THB |
| Network egress | ~0.01 THB |
| **Total** | **~0.19 THB per 1,000 predictions** |

## Cost Control Measures

1. **Scale-to-zero**: Cloud Run min-instances=0 — zero cost when idle
2. **CPU-only inference**: No GPU needed for EfficientNet-B0 at 224×224 (p95 < 200ms on CPU)
3. **Resource tagging**: All resources tagged `course=itcs355 student=6688249` for tracking
4. **Budget alert**: Set at 600 THB (75% of 800 THB limit) in Cloud Console
5. **Teardown script**: `make teardown` lists all tagged resources for cleanup

## Comparison to Budget

```
Budget:     800 THB ████████████████████████████████████████ 100%
Estimated:  340 THB ████████████████░░░░░░░░░░░░░░░░░░░░░░░░  43%
Remaining:  460 THB                 ░░░░░░░░░░░░░░░░░░░░░░░░  57%
```

**Healthy margin** — 57% of budget remains for unexpected costs or extended testing.

## Teardown Checklist

Run at end of term to avoid ongoing charges:

- [ ] Delete Cloud Run services: `gcloud run services delete xray-fracture-predict --region=asia-southeast1`
- [ ] Delete Cloud Storage bucket: `gsutil rm -r gs://itcs355-6688249-data`
- [ ] Delete Artifact Registry repo: `gcloud artifacts repositories delete xray --location=asia-southeast1`
- [ ] Verify no remaining resources: `gcloud resource-manager tags list --project=itcs355-6688249`
- [ ] Disable billing on the project (safety net)
