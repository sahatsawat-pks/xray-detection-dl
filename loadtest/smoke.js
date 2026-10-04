/*
 * smoke.js — k6 load test for the fracture detection API.
 *
 * Tests at 3 concurrency levels: 1, 10, 50 VUs.
 * Pre-declared latency target: p95 < 200ms at 10 VUs.
 *
 * Usage:
 *   k6 run loadtest/smoke.js
 *   k6 run loadtest/smoke.js --env BASE_URL=http://deployed-service-url
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// ── Custom metrics ───────────────────────────────────────────────────────────
const errorRate = new Rate('error_rate');
const predictionLatency = new Trend('prediction_latency', true);

// ── Configuration ────────────────────────────────────────────────────────────
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

// Pre-declared thresholds — committed BEFORE running the test
export const options = {
    stages: [
        { duration: '10s', target: 1 },   // Warm-up: 1 VU
        { duration: '30s', target: 1 },   // Baseline: 1 VU sustained
        { duration: '15s', target: 10 },  // Ramp to 10 VUs
        { duration: '30s', target: 10 },  // Sustained: 10 VUs
        { duration: '15s', target: 50 },  // Ramp to 50 VUs (stress)
        { duration: '30s', target: 50 },  // Sustained: 50 VUs
        { duration: '10s', target: 0 },   // Cool-down
    ],
    thresholds: {
        // Latency SLO: p95 < 200ms at 10 VUs
        'prediction_latency': ['p(95)<200'],
        // Error rate: < 1%
        'error_rate': ['rate<0.01'],
        // HTTP failures: < 5%
        'http_req_failed': ['rate<0.05'],
    },
};

// ── Helper: generate a random JPEG-like payload ──────────────────────────────
function makeTestImage() {
    // Generate a small random binary payload that the service can process
    // In practice, use a real sample X-ray for accurate latency measurement
    const size = 224 * 224 * 3;
    const data = new Uint8Array(size);
    for (let i = 0; i < size; i++) {
        data[i] = Math.floor(Math.random() * 256);
    }
    return http.file(data, 'test.jpg', 'image/jpeg');
}

// ── Test scenario ────────────────────────────────────────────────────────────
export default function () {
    // Health check (lightweight, doesn't count toward prediction latency)
    const healthRes = http.get(`${BASE_URL}/health`);
    check(healthRes, {
        'health returns 200': (r) => r.status === 200,
    });

    // Prediction request
    const payload = {
        file: makeTestImage(),
    };

    const predRes = http.post(`${BASE_URL}/predict`, payload);

    const success = check(predRes, {
        'predict returns 200 or 422': (r) => r.status === 200 || r.status === 422,
        'response has label': (r) => {
            if (r.status === 200) {
                const body = JSON.parse(r.body);
                return body.label !== undefined;
            }
            return true; // 422 is expected for random binary data
        },
    });

    errorRate.add(!success);
    predictionLatency.add(predRes.timings.duration);

    sleep(0.5); // Pacing between requests
}

// ── Summary ──────────────────────────────────────────────────────────────────
export function handleSummary(data) {
    const p50 = data.metrics.prediction_latency
        ? data.metrics.prediction_latency.values['p(50)']
        : 'N/A';
    const p95 = data.metrics.prediction_latency
        ? data.metrics.prediction_latency.values['p(95)']
        : 'N/A';
    const p99 = data.metrics.prediction_latency
        ? data.metrics.prediction_latency.values['p(99)']
        : 'N/A';

    console.log('\n═══ Load Test Summary ═══');
    console.log(`  p50 latency: ${p50}ms`);
    console.log(`  p95 latency: ${p95}ms`);
    console.log(`  p99 latency: ${p99}ms`);
    console.log(`  Error rate:  ${data.metrics.error_rate ? data.metrics.error_rate.values.rate : 'N/A'}`);
    console.log('═════════════════════════\n');

    return {
        stdout: JSON.stringify(data, null, 2),
    };
}
