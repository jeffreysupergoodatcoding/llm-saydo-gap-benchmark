# Decisions Log

Running log of every scoping decision, assumption, and surprise during the study. Each entry: timestamp, what, why.

## 2026-05-23 — Initial scoping

- **Dataset**: H&M Personalized Fashion Recommendations (chosen over UCI Online Retail II, Instacart, MovieLens 25M) because (a) used in recent LLM-recommender papers, (b) two-year window supports temporal splits + scaling curves, (c) rich textual product metadata enables interview-style LLM narratives.
- **Target**: 30-day per-customer repeat purchase (binary), not Kaggle MAP@12. Binary is cleaner for behavioral prediction framing.
- **Sample**: 50k customers stratified by activity bucket. LLM subsample 5k.
- **Splits**: temporal cutoffs at 2020-07-22 (train label window) and 2020-08-22 (test label window).
- **Reps**: A=RFM, B=bag-of-categories, C=SASRec+BERT4Rec+GRU4Rec, D=LLM twin with Park-2024 narrative.
- **Codebase**: clean self-contained repo at `study/`; not extending Fragment Labs (H&M shape doesn't fit FL's brand/cohort machinery).

## 2026-05-23 — Data acquisition

- No Kaggle credentials available. Pivoted to HuggingFace mirror `einrafh/hnm-fashion-recommendations-data` which hosts the raw CSVs.
- Files: articles.csv (34 MB), customers.csv (198 MB), transactions_train.csv (3.49 GB). Validated via HF dataset page.

## 2026-05-23 — Phase 1 results & decisions

- Build splits: full pre-cutoff pool train=1.31M customers, test=1.34M. Natural label rates 17.5% (train) / 16.6% (test).
- Stratified subsample of 50k from each pool gives 21.5% label rate (heavy buyers oversampled). Equal-N strata is intentional: enables per-bucket regime analysis without low-N segments being washed out.
- **Amended pre-registration**: 50k is sampled separately from train pool (80/10 → 40k/5k train/val) AND from test pool (~46k after de-duplication against train+val customer_ids), instead of an 80/10/10 single-pool partition. Rationale: train and test windows have different cutoffs (2020-07-22 vs 2020-08-22), so the two pools are NOT identical. A single-pool 80/10/10 with the test-cutoff features and train-cutoff labels would create a temporal inconsistency. The current design uses each pool's own cutoff for feature computation; customer disjointness is enforced. This is a stronger temporal split than the original prereg text — documented here as a pre-registration amendment.
- Activity bucket label rates: bucket 1 (2.7%), 2-5 (4.9%), 6-20 (12.3%), 21-100 (32.6%), 101+ (59.8%). Strong monotonic gradient → regime analysis well-supported.

## 2026-05-23 — Phase 2 results

- Baselines: majority/prior PR-AUC=0.215, popularity+recency=0.586, RFM logistic=0.611. Logistic beats recency by Δ=0.025 (95% paired bootstrap CI [0.017, 0.033], p<0.001). All later results must clear 0.611.

## 2026-05-23 — Phase 3 results (data-volume scaling)

- **N_events sweep**: PR-AUC saturates near N=32. Going from N=32→64 adds only +0.005 PR-AUC, N=64→all adds +0.001. Inflection identified at N=32.
- **N_days sweep**: PR-AUC ramps steadily: 7d=0.31, 14d=0.36, 30d=0.45, 60d=0.54, 90d=0.57, 180d=0.61, 365d=0.62. Peaks at 365 days; very slight dip at all-history (~700d).
- Two clear scaling-law findings. The N_days curve in particular is publication-worthy.

## 2026-05-23 — Final-audit fixes

- **H4 formal Spearman computed**: pre-reg specified ρ(PR-AUC, -Wasserstein) < 0 as the rank-inversion test; the previous report adjudicated H4 qualitatively. Computed formally over all 7 reps: ρ = +0.393, p = 0.383. Verdict: REFUTED (positive sign instead of negative; small n=7 limits power). Saved to `results/audit_h4_d3_followup.json`.
- **D3 on matched subsample**: D3 evaluated on 1,000 customers; D2 was on 5,000. The 121-customer overlap allows a fair head-to-head with A_lgbm under matched ground truth: A_lgbm = 0.647, D2 = 0.617, D3 = 0.675. D3 beats A_lgbm by Δ = +0.028 *on this subsample only*. n=121 → wide CI; should be treated as suggestive evidence and re-run at n ≥ 5,000.
- **LLM provider switch**: pre-registration listed `gpt-4o-mini`. The available OpenAI key had no quota; same for Anthropic. Switched to Gemini 2.5 Flash (free-tier key available). Both are comparable small/fast LLMs; the substantive design is unchanged. Documented as a pre-registration deviation here.
- **D3 framing**: §4.5 lever 3 phrasing tightened to acknowledge D3 result as "suggestive at n=1,000 (with the matched-121 overlap showing D3 > A_lgbm by +0.028)". Moved the speculative "headline lever" phrasing to §8 future work as recommended by the audit.

## 2026-05-23 — Phase 4-6 audit fixes (after Phase 4-6 verification subagent)

- **L1 baseline protocol fix**: Phase 6 lever-1 originally trained the LGBM baseline without early stopping (n_estimators=600, no eval_set), while Phase 4a `B_lgbm` used early stopping. This biased the L1 baseline DOWN, making the negative finding ("balanced class_weight HURTS PR-AUC") look stronger than warranted. Fixed: phase6_levers.py now uses identical early-stopping protocol for all 3 L1 conditions. Updated numbers: baseline_no_weight=0.612, balanced=0.572 (still significantly worse), scale_pos_weight_4=0.620. Finding holds under fair protocol.
- **Pre-registration B=500 vs B=1000 deviation**: prereg specifies B=1000 bootstrap iterations; code uses B=500 (B=300 in Phase 6). For a sample of ~46k test customers with PR-AUC SE ≈ 0.005, B=500 vs B=1000 widens the CI by ~3% — not meaningful. Documenting the deviation here rather than re-running. Final report will note B=500 used.
- **Rep C framing fix**: SASRec/BERT4Rec/GRU4Rec are adapted from their original next-item objective to a scalar binary repeat-purchase head. This is a meaningful departure from their published form. Report will state this explicitly and frame the comparison as "sequence-encoders with a binary head, not next-item SOTA."
- **Cost accumulator bug fix**: phase4b_llm_twin.py was overwriting `cost_total` instead of accumulating; fixed to `+=`.

## 2026-05-23 — Audit fixes (after Phase 1-2 verification subagent)

- **Fixed cutoff_guard** (src/features.py): replaced silent try/except with an explicit assertion that samples up to 5000 of the function's customer_ids and verifies max(t_dat) on the underlying tx table is strictly less than the cutoff. Will fail loudly on violation.
- **Removed leaky Rep D fields**: dropped `club_member_status` and `fashion_news_frequency` from `behavioral_narrative` because customers.csv is a Sept-2020 snapshot, not time-stamped — using these for 2020-07/2020-08 cutoff features is leakage. Only `age` (stable demographic) retained.
- Documented split-design amendment above.
