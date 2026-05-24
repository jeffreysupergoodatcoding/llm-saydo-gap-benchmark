# Pre-Registration v2 — *Say-Do Gap of LLM Digital Twins on H&M*

**Committed**: 2026-05-24, before any Phase 10 LLM run.

This pre-registration extends the original `preregistration.md` (which adjudicated H1–H5 in `report.md`). H1–H5 are now closed; this document covers H6–H10 — but per the methodology audit, several of those have been demoted to replications. The final pre-registered set is **H7 + H9 (confirmatory)** and **R1 + R2 (replication / effect-size only)**.

---

## Dataset and splits

Same as `preregistration.md`. H&M Personalized Fashion Recommendations, two temporal cutoffs, customer-disjoint splits. Test pool: 46,865 customers. No re-draw.

## Core-1,000 sampling

A stratified sample of **1,000 test customers**, drawn deterministically with seed = 42, stratified equally across the five activity buckets (200 per bucket × {1, 2-5, 6-20, 21-100, 101+}). This same set is used across F-base, F-nobase, and the matched slice of D2 to enable paired tests.

The C-flat (Claude direct API) arm uses a **subset of 400** of the core-1,000, again stratified (80 per bucket).

## Arms

| Arm | Provider | Architecture | Prompt | n | Notes |
|---|---|---|---|---|---|
| **D2** | Gemini 2.5 Flash | flat narrative | as in v1 | 5,000 | Already run, scores reused from `results/phase4b_D2_scores.npz` |
| **F-base** | Gemini 2.5 Flash | Park-2023-lineage 5-stage cognition pipeline (Fragment Labs port) | full prompt **incl.** H&M per-bucket base-rate table | 1,000 (core-1,000) | New |
| **F-nobase** | Gemini 2.5 Flash | same 5-stage cognition pipeline | identical prompt **excl.** base-rate table | 1,000 (same core-1,000) | New; isolates pipeline contribution from base-rate leakage |
| **C-flat** | Anthropic direct API (`claude-haiku-4-5` default; upgrade if quota permits) | flat narrative | identical to D2 prompt | 400 (subset of core-1,000) | New; direct API not Claude Code subagent |

## Confirmatory hypotheses

Both at family-wise α = 0.05 with **Bonferroni correction → α = 0.025 per test**.

### H7 — Cognition architecture closes the gap, after leakage control

On the core-1,000 (matched customers across arms), the **signed mean gap**
\[
\text{gap}(X) = \overline{\text{stated\_intent\_prob}_X} - \overline{\text{actual\_label}_X}
\]
satisfies
\[
|\text{gap}(F\text{-nobase})| \le |\text{gap}(D2_{\,|\,\text{core-1k}})| - 0.05
\]
i.e. the cognition pipeline reduces the magnitude of the gap by ≥ 0.05 *after* removing the in-prompt base-rate table. Tested via paired Wilcoxon signed-rank on per-customer `|stated − actual|`.

The leaky variant `gap(F-base) vs gap(D2)` is also reported, but **only as a sensitivity check**. If `|gap(F-base) − gap(F-nobase)| > |gap(F-nobase) − gap(D2)|`, then the headline reframes to *"apparent architecture gain is mostly the leaked base-rate table"* — a still-publishable negative result.

### H9 — Verbatim coherence beats permutation baseline

For the F-base and F-nobase arms (which emit a `verbatim_reaction` text field), define per-customer:
- `cos_actual` = cosine(embed(verbatim), embed(actual next-purchased article's `prod_name + detail_desc`))
- `mrr` = mean reciprocal rank of the true next-article when verbatim is scored against 100 in-bucket-within-week distractor articles (sampled deterministically with seed=42).

Tested:
- **H9a**: `cos_actual` exceeds within-bucket-within-week shuffled-pair baseline (paired permutation, B = 10,000), aggregated across F-base + F-nobase.
- **H9b**: MRR > 1/100 = 0.01 by a margin of at least 0.05 (so the LLM's quote is meaningfully predictive of which specific article).

H9 passes only if both H9a and H9b clear α = 0.025 with the Bonferroni correction folded in.

**Embedding model**: `text-embedding-3-small` (OpenAI) or `bge-large-en-v1.5` (HF), chosen at run time to be **disjoint** from the LLM provider (avoid co-training contamination). If neither has quota, fall back to Gemini `text-embedding-004` and acknowledge the confound in the limitations section.

## Replication / effect-size only

### R1 — Intent inflation

Report `gap(arm)` for every arm (sign, magnitude, 95% bootstrap CI, B = 1000). No NHST. The D2 pilot showed `gap(D2) ≈ +0.41`; we expect new arms to remain inflated.

### R2 — Heterogeneous gap

Report `gap(arm | activity_bucket = b)` for b ∈ {1, 2-5, 6-20, 21-100, 101+} in every arm. No NHST. D2 pattern: gap shrinks as bucket increases.

## Canonical stated_intent_prob extraction

The LLM's `stated_intent_prob` is **`stimulus_30d_buy_likelihood / 100`** from the Fragment deliberation JSON, or for flat-prompt arms the scalar `p` (defined to be the same construct in the prompt). One extraction rule, all arms.

## Primary metric

`gap = mean(stated_intent_prob) − mean(actual_label)`, with:

- **Test-distribution reweighting**: equal-strata sampling oversamples high-activity buckets. Weight each customer by (test-population fraction of their bucket) / (sample fraction of their bucket) before averaging. Mitigates the audit's label-rate-drift concern.
- **Bootstrap 95% CI** at B = 1000 (raised from B = 500 in the v1 prereg to honor the original protocol).

## Secondary metrics

- Calibration plot (5 quantile bins for n=400; 10 bins for n=1,000)
- ECE per arm
- Brier on `stated_intent_prob`
- Spearman ρ(stated, actual) per arm
- Park-style normalized accuracy (reused from v1)

## Multiple-comparison correction

- Confirmatory family = {H7, H9}. Bonferroni → α = 0.025 each.
- H9 itself has two sub-tests (H9a, H9b); both must clear α = 0.025 → effective per-sub-test α = 0.0125. Treated as a single conjunctive test.

## Controls (mandatory, all reported)

### Control 1 — Base-rate-leakage decomposition
`Δ_F = gap(F-base) − gap(F-nobase)`, with paired CI. If `|Δ_F| > |gap(F-nobase) − gap(D2)|`, headline reframes as above.

### Control 2 — Kaggle memorization inversion probe
Pre-Phase-10: 20 hashed-only inputs to Claude and Gemini; checked for any leak of unhashed ID or transaction content. **Halt on any non-trivial leak**, redesign prompt to hash everything that could be looked up.

### Control 3 — Counterfactual trace perturbation
n = 50 customers from the core-1,000; each scored twice: once with the real trace and once with a minimally perturbed trace (drop last purchase; swap one product type; swap one colour). Report:
- mean `|Δ stated_intent_prob|` (per customer)
- mean cosine shift in verbatim embedding
If both are below noise (mean `|Δp|` < 0.02 and verbatim cosine shift < 0.02), the LLM is anchoring on global priors, not the specific trace. Reported regardless of outcome.

### Control 4 — Quote-specificity audit
For each verbatim, compute type-token ratio (TTR) and length. Bin customers into high-specificity (top-quartile TTR) vs low. Re-run H9 conditional on specificity. If H9 passes only on the high-specificity subset, footnote with that caveat.

## Hyperparameter freeze

The Fragment cognition pipeline's free parameters (60/40 LLM-vs-affect blend in `decision.py`; friction component weights in `affect.py`; memory retrieval k=5) are **frozen at the WIP defaults** lifted from Fragment Labs source. **No tuning** on H&M training data. Recorded in `decisions_log.md`.

## Pre-registration deviations from v1

- **B raised to 1000** (v1 used B = 500 due to time; v2 honors the original target).
- **gpt-4o-mini → Gemini 2.5 Flash** (Gemini for the F-* arms; Claude for the C-flat arm). Documented in `decisions_log.md` from v1; carried forward.
- **Hypotheses tightened from 5 to 2 confirmatory + 2 replication**, per methodology audit (n=100 was severely under-powered for original H10).

## Acceptance

Each of H7 and H9 reported with effect size, paired CI, Bonferroni-corrected p, in `report_v2.md` §5. R1 and R2 reported as effect sizes only. All four controls reported.

Hash of this file committed to git before Phase 10 launches; hash recorded in `report_v2.md`.
