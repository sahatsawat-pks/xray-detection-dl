# Load Test — Bone Fracture X-Ray Detection API

## Pre-Declared Latency Target

> **Committed before running the test** (ITCS355 requirement)

| Metric | Target |
|---|---|
| **p95 latency** | < 200 ms at 10 concurrent users |
| **Error rate** | < 1% |

## Running the Test

```bash
# Start the API server
make serve

# In another terminal, run the load test
k6 run loadtest/smoke.js

# Or against a deployed service
k6 run loadtest/smoke.js --env BASE_URL=https://your-deployed-url
```

## Test Stages

| Stage | Duration | VUs | Purpose |
|---|---|---|---|
| Warm-up | 10s | 1 | Let the server warm up |
| Baseline | 30s | 1 | Measure single-user latency |
| Ramp | 15s | 1→10 | Gradual increase |
| Sustained | 30s | 10 | **Primary SLO measurement** |
| Stress ramp | 15s | 10→50 | Find breaking point |
| Stress | 30s | 50 | Document degradation |
| Cool-down | 10s | 50→0 | Graceful wind-down |

## Results

> **Fill in after running the test honestly.**

| VUs | p50 (ms) | p95 (ms) | p99 (ms) | Error Rate | Throughput (req/s) |
|---|---|---|---|---|---|
| 1 | | | | | |
| 10 | | | | | |
| 50 | | | | | |

### Where it breaks

> Document the concurrency level where p95 exceeds the target or error rate climbs above 1%.
> This is expected — the point is to know your limits honestly.

### Cost per 1,000 predictions

> Calculated from: (inference instance hours / predictions served) × hourly rate
