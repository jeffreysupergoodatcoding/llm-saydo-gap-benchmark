# When Do LLM Digital Twins Beat Classical Sequence Models?

A reproducible study of LLM-based digital twins vs classical sequence recommenders on the H&M Personalized Fashion Recommendations dataset, for per-customer 30-day repeat-purchase prediction.

## What this repo does

- Loads the H&M dataset (31M txns, 1.4M customers, 2018-09 → 2020-09).
- Builds customer-disjoint temporal splits with strict cutoff-guard leakage checks.
- Compares four representations: RFM aggregates, bag-of-categories, sequence models (SASRec / BERT4Rec / GRU4Rec), and an LLM digital twin (Park-2024 interview-style narrative + gpt-4o-mini).
- Produces data-volume scaling curves, regime win-map (activity bucket × history length), Park-style normalized accuracy, Wasserstein-1 distributional metric, and a cost-accuracy Pareto frontier.

## Reproduce

Requirements:
- Python 3.13 + `uv`
- Homebrew `libomp` on macOS (for LightGBM)
- An OpenAI API key in `.env` (`OPENAI_API_KEY=...`) for Phase 4b (~$2-5 in API costs at $20 cap)

```bash
# 1. Install deps
uv sync

# 2. Download H&M data (3.5 GB transactions + small auxiliary)
mkdir -p data/raw
curl -L "https://huggingface.co/datasets/einrafh/hnm-fashion-recommendations-data/resolve/main/data/raw/articles.csv" -o data/raw/articles.csv
curl -L "https://huggingface.co/datasets/einrafh/hnm-fashion-recommendations-data/resolve/main/data/raw/customers.csv" -o data/raw/customers.csv
curl -L "https://huggingface.co/datasets/einrafh/hnm-fashion-recommendations-data/resolve/main/data/raw/transactions_train.csv" -o data/raw/transactions_train.csv

# 3. Convert to Parquet
PYTHONPATH=. uv run python -c "from src.data import csv_to_parquet; csv_to_parquet()"

# 4. Run phases in order
DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib PYTHONPATH=. uv run python scripts/phase1_splits.py
PYTHONPATH=. uv run python -m src.leakage_audit
DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib PYTHONPATH=. uv run python scripts/phase2_baselines.py
DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib PYTHONPATH=. uv run python scripts/phase3_scaling.py
DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib PYTHONPATH=. uv run python scripts/phase4a_reps_abc.py
PYTHONPATH=. uv run python scripts/phase4b_llm_twin.py --variant D2 --n 5000
PYTHONPATH=. uv run python scripts/phase4b_llm_twin.py --variant D3 --n 1000
PYTHONPATH=. uv run python scripts/phase4c_regime.py
PYTHONPATH=. uv run python scripts/phase5_distributional.py
DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib PYTHONPATH=. uv run python scripts/phase6_levers.py
```

LLM responses are cached to `cache/llm/` keyed by content hash, so reruns are free.

## Layout

```
study/
  src/                — library code
    data.py             — loaders, Parquet conversion
    splits.py           — temporal customer-disjoint splits
    features.py         — RFM, bag-of-categories, sequences, LLM narrative (+ cutoff_guard)
    eval.py             — PR-AUC, ROC, Brier, ECE, Wasserstein, Park normalized acc, bootstrap CIs
    leakage_audit.py    — independent verifier
    llm_client.py       — multi-provider, disk cache, $20 cap
    models/             — tabular (logistic, LightGBM) + sequence (SASRec, BERT4Rec, GRU4Rec)
  scripts/            — per-phase orchestrators
  data/               — raw + processed + splits
  results/            — JSON + PNG output, one file per phase
  cache/              — LLM response cache
  preregistration.md  — hypotheses + protocol committed before Phase 4
  decisions_log.md    — running log of all decisions and amendments
  references.bib      — full citations
  report.md           — final research report
```

## License & data attribution

- Code: MIT.
- Data: H&M Personalized Fashion Recommendations (Kaggle, H&M Group). Use the HuggingFace mirror `einrafh/hnm-fashion-recommendations-data` if Kaggle creds are unavailable.

## Citation

If you use this code or results, please cite the forthcoming paper (in preparation).
