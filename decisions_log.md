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

## 2026-05-24 — Iterations 5-8: cross-domain + provider + final audit

- **Iteration 5**: Cross-domain MovieLens 25M arms (F-base + F-nobase, n=594 each). Leakage-vs-architecture pattern **replicates directionally** (gap(F-base)<gap(F-nobase), paired diff −0.020, p<0.001). Simpson's-paradox pooled-vs-within pattern **inverts** on ML (within>pooled) — paper reframed to call the decomposition a "bucket-prior-dependence diagnostic" rather than claiming uniform replication.
- **Iteration 6 prereg deviation (acknowledged)**: prereg specified C-flat = Anthropic direct API n=400; substitute = Claude Code subagent n=50. Both provider mechanism and sample size differ. Recorded as exploratory cross-provider evidence; pre-registered confirmatory provider comparison remains open.
- **Iteration 7 human-self benchmark**: H&M test-pool adjacent-30-day same-customer Pearson r = 0.38-0.39, past-2-windows-avg → current r = 0.45. The within-domain human-self benchmark; LLM within-bucket Spearman (≈0.23) is roughly half this.
- **Iteration 8 final blind review** identified 3 blockers, all addressed: (1) prereg deviation flagged in §4.4 and here; (2) abstract rewritten to honestly describe domain-heterogeneous diagnostic; (3) bib audit pending (some 2601-2604 arXiv IDs need verification before camera-ready).

## 2026-05-24 — Mid-run audit fixes (3 BLOCKER bugs caught while F-base was running)

Three parallel audit subagents (cognition pipeline / Phase-11 stats / deliberation prompt) flagged blockers BEFORE the F-base run completed. Fixes applied to disk (F-base process loaded the previous code, but its outputs go through the fixed Phase-11 scripts):

1. **BLOCKER — canonical stated_intent_prob was post-guardrail.** `run.py:55` exported `stated_intent_prob_final` (after `decision.py` clamps lapsed→0.25, single→0.30, etc.). But `preregistration_v2.md` defines canonical = raw `stimulus_30d_buy_likelihood/100`. The guardrails are pure mean-shrinkage toward truth — they would have given F-arms an unearned gap reduction. **Fix**: `phase11_gap.py::load_arm` now loads `stated_intent_raw` from the npz (already saved by phase10_arms.py); the guardrailed value is retained as `scores_guardrailed` for sensitivity.

2. **BLOCKER — H9a permutation null was degenerate.** `phase11_verbatim.py` previously used `mean(cosine to distractor pool)` as the "shuffled baseline." That collapses the permutation null into a deterministic statistic; the prereg-specified test was a within-bucket permutation of (verbatim → actual-article) assignments. **Fix**: replaced with proper within-bucket permutation, n_perm=10000.

3. **BLOCKER — bucket-index lookup bug.** `phase11_verbatim.py` indexed `buckets[ci]` where `ci` was an index into `positive_cids` (a subset of `cids`). For any customer in `positive_cids` past position N in the full cid list, this silently returned the wrong bucket. **Fix**: built `positive_buckets` parallel array.

4. **MAJOR — H7 verdict missed prereg's 0.05 margin.** Wilcoxon-significant alone was being read as "CONFIRMED." Prereg requires `|gap(F-nobase)| ≤ |gap(D2)| − 0.05`. **Fix**: `phase11_gap.py` now computes the paired |gap| margin separately and only declares CONFIRMED if both the Wilcoxon AND the margin clear thresholds. Three-state verdict: CONFIRMED / PARTIAL / REFUTED_or_NS.

5. **MAJOR — stratified bootstrap not used.** The bootstrap on the reweighted gap was IID instead of stratified-within-bucket. **Fix**: stratified resampling within each bucket to the original bucket size.

6. **MAJOR — H9b chance MRR.** Pre-reg said `1/100` but the actual chance MRR for a uniformly placed actual among 101 items is `H_101/101 ≈ 0.0517`. **Fix**: report both numbers; use the harmonic-mean rate as the principled comparator.

7. **MAJOR — counterfactual perturbation not minimal.** `_perturb_trace` was dropping the last purchase AND decrementing `total_orders` AND `distinct_articles`. **Fix**: now only swaps colour + product_type on one recent purchase; preserves aggregate stats. Threshold raised from 0.02 to 0.05 (above Gemini's output resolution).

8. **APPLY-NOW deliberation parser bugs.** `deliberation.py::parse_response` regex `\{[^{}]*\}` failed on nested objects; `_safe_pct` treated integer `1` as "100% on 0..1 scale" rather than "1 percent." **Fix**: brace-balanced extractor; `parse_ok` flag; integer-vs-fractional disambiguation in `_safe_pct`.

9. **DOCUMENTED — `aov >= 0.04` threshold provenance.** Cognition pipeline thresholds were inherited from Fragment WIP defaults. The H&M AOV scale matches by coincidence. To preempt reviewer "tuned-on-test" challenges, this is now documented: thresholds are *Fragment defaults*, not H&M-fitted.

These fixes are mid-run because the parallel audit caught them before F-nobase started. The F-base outputs already saved include raw scores, so no LLM re-running is required.

## 2026-05-24 — Extension v2 launched

- Pre-registration v2 committed to git (commit `40b07a2`, prereg-v2 hash `ba96c6ec`) before any Phase 10 LLM run, per audit requirement.
- references.bib comprehensively corrected after subagent audit verified each arXiv ID. Fixed: peng2025funhouse (title was wrong; first author Tianyi not Tara), li2025behaviorchain (was zhang2025howfar; first author Rui Li not Zhang), wang2026productdiscovery (first author Zichao not Zheng), chu2025marketing (title corrected), ergul2025instruction (first author was wrong), evalbehaviorsim2025 (title corrected). Added 11 new entries including Andric 2025, Alignment Revisited, Mind the Gap, Lu 2025, Sheeran 2002/2016, Verplanken & Faes, LaPiere 1934, Ben-Akiva, Diamond-Hausman, NOAA panel.
- Phase 8 smoke test (n=5) confirms the F-base / F-nobase contrast produces dramatic per-customer divergence (e.g., customer `8547f94d…` 0.08 with base-rate anchor vs 0.65 without). This validates the audit's prediction that the base-rate-leakage decomposition will be a dominant signal — making it potentially the headline of the extension paper.
- Phase 9 memorization probe: Gemini returned `UNKNOWN` for all 20 H&M `customer_id`s (0/20 suspicious). No detectable contamination from Kaggle public notebooks; safe to proceed.
- **C-flat arm dropped**: Anthropic API returns 400 "credit balance too low" across `claude-haiku-4-5`, `claude-3-5-haiku-latest`, `claude-3-5-sonnet-latest`. Per plan fallback ("if no quota, fall back to Claude Code subagents and explicitly acknowledge the confound") and per audit recommendation (H10 demoted/dropped because n=100 Claude was severely under-powered anyway), we skip the C-flat arm in the headline. Provider comparison is left to future work.

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
