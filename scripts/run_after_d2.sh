#!/usr/bin/env bash
# Wait for Phase 4b D2 to finish, then run D3 ablation, Phase 4c, Phase 5 (with LLM), Phase 7 report.
set -euo pipefail
cd "$(dirname "$0")/.."

LOG=/tmp/run_after_d2.log
echo "[run-chain] waiting for phase4b D2..." > $LOG

# Wait for D2 result file
until test -f results/phase4b_D2.json; do
  sleep 30
done
echo "[run-chain] D2 done. Running D3 ablation (n=1000)..." | tee -a $LOG

PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python scripts/phase4b_llm_twin.py --variant D3 --n 1000 --model gemini-2.5-flash 2>&1 | tee -a $LOG || true

echo "[run-chain] Running Phase 4c regime analysis..." | tee -a $LOG
PYTHONUNBUFFERED=1 PYTHONPATH=. DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib uv run python scripts/phase4c_regime.py 2>&1 | tee -a $LOG || true

echo "[run-chain] Re-running Phase 5 with LLM rep..." | tee -a $LOG
PYTHONUNBUFFERED=1 PYTHONPATH=. DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib uv run python scripts/phase5_distributional.py 2>&1 | tee -a $LOG || true

echo "[run-chain] Re-running Phase 6 to refresh Pareto with LLM..." | tee -a $LOG
PYTHONUNBUFFERED=1 PYTHONPATH=. DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib uv run python scripts/phase6_levers.py 2>&1 | tee -a $LOG || true

echo "[run-chain] Writing Phase 7 report (auto-gen)..." | tee -a $LOG
PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python scripts/phase7_report.py 2>&1 | tee -a $LOG || true

echo "[run-chain] Injecting LLM numbers into manual report.md..." | tee -a $LOG
PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python scripts/phase7_inject_llm.py 2>&1 | tee -a $LOG || true

echo "[run-chain] All done." | tee -a $LOG
