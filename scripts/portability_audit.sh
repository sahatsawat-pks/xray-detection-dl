#!/usr/bin/env bash
# portability_audit.sh — Ensure no cloud SDK imports in src/ or tests/
#
# The three-layer contract:
#   src/   = Layer 1, provider-neutral
#   tests/ = must not depend on cloud
#   cloudlayer/ = Layer 2, the ONLY place cloud SDKs may appear

set -euo pipefail

FAILED=0
PATTERNS=(
    's3://'
    'abfss://'
    'gs://'
    'amazonaws'
    'azure'
    'googleapis'
    'google.cloud'
    'boto3'
    'botocore'
    'azure.storage'
    'azure.identity'
)

echo "=== Portability Audit ==="
echo "Scanning src/ and tests/ for cloud-specific strings..."
echo

for pattern in "${PATTERNS[@]}"; do
    if grep -rn --include='*.py' "$pattern" src/ tests/ 2>/dev/null; then
        echo "❌ FAIL: Found '$pattern' in src/ or tests/"
        FAILED=1
    fi
done

if [ $FAILED -eq 0 ]; then
    echo "✅ PASS — No cloud-specific strings found in src/ or tests/"
    exit 0
else
    echo
    echo "❌ AUDIT FAILED — Cloud SDK references found outside cloudlayer/"
    exit 1
fi
