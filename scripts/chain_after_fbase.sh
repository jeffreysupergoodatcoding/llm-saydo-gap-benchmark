#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."

LOG=/tmp/chain_after_fbase.log
echo "[chain] waiting for F-base summary..." | tee $LOG

# Wait for F-base summary
until test -f results/phase10_F-base_summary.json; do
  sleep 30
done
echo "[chain] F-base done. Starting F-nobase..." | tee -a $LOG

PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python scripts/phase10_arms.py --arm F-nobase --per-bucket 200 --model gemini-2.5-flash 2>&1 | tee -a $LOG || true

echo "[chain] Running Phase 11 gap analysis..." | tee -a $LOG
PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python scripts/phase11_gap.py 2>&1 | tee -a $LOG || true

echo "[chain] Running Phase 11b verbatim coherence..." | tee -a $LOG
PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python scripts/phase11_verbatim.py 2>&1 | tee -a $LOG || true

echo "[chain] Running Phase 11c counterfactual..." | tee -a $LOG
PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python scripts/phase11_counterfactual.py 2>&1 | tee -a $LOG || true

echo "[chain] Writing report_v2.md..." | tee -a $LOG
PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python scripts/phase12_report.py 2>&1 | tee -a $LOG || true

echo "[chain] all done." | tee -a $LOG
