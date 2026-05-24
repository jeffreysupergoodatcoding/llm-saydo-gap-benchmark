# Pre-Registration — Regime Analysis of LLM Digital Twins vs Classical Sequence Models on H&M

**Committed**: 2026-05-23 (before Phase 4 begins)

## Research question

Under what data conditions (customer history length, activity bucket) do Park-2024-style LLM digital twins outperform tuned classical sequence recommenders on next-period customer purchase prediction, and at what cost?

## Dataset

- **Primary**: H&M Personalized Fashion Recommendations (Kaggle 2022).
- Window: 2018-09-20 → 2020-09-22.
- Stratified subsample: **N=50,000 customers**, strata = activity bucket {1, 2–5, 6–20, 21–100, 100+} pre-cutoff transactions.
- LLM rep run on a further stratified N=5,000-customer subsample (cost cap).

## Splits

- `T_train_cutoff = 2020-07-22`, train label window `[2020-07-22, 2020-08-22)`.
- `T_test_cutoff = 2020-08-22`, test label window `[2020-08-22, 2020-09-22)`.
- 80/10/10 customer-disjoint train/val/test partition. Random seed = 42.
- All features computed with `t_dat < T_cutoff` only; enforced by `@cutoff_guard` decorator.

## Prediction target

Per-customer 30-day repeat purchase (binary): ≥1 transaction in label window, conditional on ≥1 transaction before cutoff.

## Representations

- **A** RFM aggregates → logistic + LightGBM
- **B** Bag-of-category counts (+ RFM) → logistic + LightGBM
- **C** Sequence models → SASRec, BERT4Rec (sampled CE), GRU4Rec
- **D** LLM digital twin → gpt-4o-mini with Park-style narrative prompt

Reviewer-bar baselines: majority class, popularity+recency heuristic, LightGBM-LTR on full hand-engineered features.

## Pre-registered hypotheses

- **H1**: For customers with ≤4 pre-cutoff transactions, Rep D PR-AUC > best classical PR-AUC, Δ ≥ 0.02.
- **H2**: For customers with ≥16 pre-cutoff transactions, best classical PR-AUC > Rep D PR-AUC, Δ ≥ 0.02.
- **H3**: Var(predicted) / Var(observed) < 1 for all reps; Levene's test p < 0.05.
- **H4**: Wasserstein-1 rank order across reps ≠ PR-AUC rank order (Spearman ρ between rankings < 0).
- **H5**: At matched PR-AUC ±0.005, best classical model is ≥10× cheaper per prediction than Rep D.

## Primary metric

**PR-AUC** with paired bootstrap 95% CI (B=1000).

## Secondary metrics

ROC-AUC, Brier, ECE, Wasserstein-1, Park-style normalized accuracy, per-segment Spearman ρ, $/prediction, latency.

## Multiple-comparison correction

Holm-Bonferroni across 6 pairwise rep comparisons; Benjamini-Hochberg FDR across regime cells.

## Random seeds

- Sequence models (C): 5 seeds each, report mean ± std.
- Tabular (A, B): 1 seed (deterministic).
- LLM (D): temperature = 0 (deterministic).

## Decision rules

- If H1 confirmed AND H2 confirmed → headline "regime crossover."
- If H1 refuted AND H2 confirmed → headline "classical dominates, twins as fallback."
- If H3 confirmed → reported as cross-domain replication of Columbia 2025.
- If H4 confirmed → reported as distributional/individual decoupling on retail.
- Surprising results (any rep beats baselines by >0.15 PR-AUC) trigger leakage audit before being believed.

## Analysis code paths

Every metric computation routed through `src/eval.py`; every feature through `src/features.py` with `@cutoff_guard`. Reproducibility via `python -m study.run_all`.

## Acceptance

All 5 hypotheses reported as confirmed/refuted with effect sizes and CIs. Negative results published as findings.
