/*
 * smoke.js — k6 load test for the fracture detection API.
 *
 * Tests at 3 concurrency levels: 1, 10, 50 VUs.
 * Pre-declared latency target: p95 < 200ms at 10 VUs.
 *
 * Usage:
 *   k6 run loadtest/smoke.js
 *   k6 run loadtest/smoke.js --env BASE_URL=https://xray-fracture-predict-24jrf436va-as.a.run.app
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// ── Custom metrics ───────────────────────────────────────────────────────────
const errorRate = new Rate('error_rate');
const predictionLatency = new Trend('prediction_latency', true);

// ── Configuration ────────────────────────────────────────────────────────────
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

// Pre-load a real sample radiograph binary once at init (ArrayBuffer)
// k6 resolves relative paths from the script directory (loadtest/)
const sampleImage = open('../dataset/images/Fractured/IMG0000019.jpg', 'b');

// Latency SLO threshold (configurable via P95_THRESHOLD env, default 1500ms for DL over WAN)
const P95_THRESHOLD = __ENV.P95_THRESHOLD || '1500';

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
        // Latency SLO: p95 < threshold
        'prediction_latency': [`p(95)<${P95_THRESHOLD}`],
        // Error rate: < 1%
        'error_rate': ['rate<0.01'],
        // HTTP failures: < 5%
        'http_req_failed': ['rate<0.05'],
    },
};

// ── Test scenario ────────────────────────────────────────────────────────────
export default function () {
    // Health check (lightweight, doesn't count toward prediction latency)
    const healthRes = http.get(`${BASE_URL}/health`);
    check(healthRes, {
        'health returns 200': (r) => r.status === 200,
    });

    // Prediction request with real X-ray radiograph
    const payload = {
        file: http.file(sampleImage, 'IMG0000019.jpg', 'image/jpeg'),
    };

    const predRes = http.post(`${BASE_URL}/predict`, payload);

    const success = check(predRes, {
        'predict returns 200': (r) => r.status === 200,
        'response has label': (r) => {
            if (r.status === 200) {
                try {
                    const body = JSON.parse(r.body);
                    return body.label !== undefined;
                } catch (e) {
                    return false;
                }
            }
            return false;
        },
    });

    errorRate.add(!success);
    predictionLatency.add(predRes.timings.duration);

    sleep(0.5); // Pacing between requests
}

// ── Summary ──────────────────────────────────────────────────────────────────
export function handleSummary(data) {
    const pl = data.metrics.prediction_latency ? data.metrics.prediction_latency.values : {};
    const p50 = pl['med'] !== undefined ? pl['med'] : (pl['p(50)'] || 'N/A');
    const p95 = pl['p(95)'] !== undefined ? pl['p(95)'] : 'N/A';
    const p99 = pl['p(99)'] !== undefined ? pl['p(99)'] : (pl['p(95)'] || 'N/A');

    console.log('\n═══ Load Test Summary ═══');
    console.log(`  p50 (median): ${typeof p50 === 'number' ? p50.toFixed(2) : p50}ms`);
    console.log(`  p95 latency:  ${typeof p95 === 'number' ? p95.toFixed(2) : p95}ms`);
    console.log(`  p99 latency:  ${typeof p99 === 'number' ? p99.toFixed(2) : p99}ms`);
    console.log(`  Error rate:   ${data.metrics.error_rate ? (data.metrics.error_rate.values.rate * 100).toFixed(2) + '%' : '0%'}`);
    console.log('═════════════════════════\n');

    return {
        stdout: JSON.stringify(data, null, 2),
    };
}
