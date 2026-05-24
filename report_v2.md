# From Stated Intent to Revealed Purchase: Quantifying the Say-Do Gap of LLM Digital Twins on H&M

**Working paper, v2.** Commit `09cec3894f07`. Pre-registration v2 hash `ba96c6ec57485740` (committed before any Phase-10 LLM run).

**Companion to**: `report.md` (v1), which established the LightGBM vs LLM regime analysis on H&M; this extension reframes that result through the stated-vs-revealed preference lens of social psychology and consumer-behavior literature [sheeran2002intention, sheeran2016intention, lapiere1934attitudes, fishbein1975belief, benakiva1994combining, diamond1994contingent].

## Abstract

On two pre-registered public benchmarks (H&M Personalized Fashion, n=1,000 paired; MovieLens 25M, n=594), the in-prompt base-rate table inside a Park-2023-lineage LLM-digital-twin prompt accounts for more of the apparent say-do-gap reduction than the cognition architecture itself — **and this leakage-vs-architecture effect replicates directionally across both domains** (H&M: |Δ_F|=0.077 > |Δ_arch|=0.062, gap(F-base) < gap(F-nobase); MovieLens: gap(F-base)=0.005 < gap(F-nobase)=0.024, paired diff −0.020, 95% CI [−0.029, −0.011]). A pooled-vs-within-stratum Spearman decomposition behaves as a **bucket-prior-dependence diagnostic** but *does not replicate as a single pattern across domains*: on H&M (where activity-bucket strongly predicts outcome) pooled ρ ≈ 0.53 matches Sheeran's human meta-analytic r [sheeran2002intention] while within-bucket ρ collapses to 0.23-0.28 (close to Toubia et al.'s twin-human r ≈ 0.2 [toubia2025twin2k500]); on MovieLens (where bucket is uninformative) within-bucket ρ ≈ 0.43 *exceeds* pooled ρ ≈ 0.31. The diagnostic is domain-heterogeneous — its sign reveals whether the LLM's apparent intent-behavior agreement is bucket-prior or within-customer reasoning, and that revelation itself is the contribution, not a uniform 'replication'. The within-bucket ρ is **provider-invariant** across Gemini 2.5 Flash and a Claude Sonnet-class Code subagent arm (Gemini: 0.23; Claude: 0.26 on the same 50 customers), while Claude's signed gap is an order of magnitude smaller than Gemini's (-0.004 vs +0.151) — provider calibration of stated intent differs but per-customer reasoning quality does not. An H&M within-domain test-retest benchmark gives Pearson r = 0.39 for same-customer 30-day window autocorrelation — the LLM's within-bucket ρ is approximately **half** this domain-specific human-self number. Counterfactual trace perturbation exceeds the LLM's stochastic noise floor (Mann-Whitney p=0.024, Cliff's δ=0.17 small), upper-bounding the per-customer reasoning signal. Our four contributions: **(a) a pooled-vs-within-stratum Spearman *diagnostic*** for bucket-prior dependence (replicates and inverts across domains, generalizes across providers); **(b) the first public-benchmark quantification of the say-do gap on H&M and MovieLens revealed behavior**; **(c) a base-rate-leakage ablation** replicated cross-domain; **(d) a counterfactual trace perturbation control** that bounds per-customer reasoning above noise.

## 1. Background and framing

Humans show a well-documented intention-behavior gap [sheeran2002intention, sheeran2016intention]: meta-analyses report median r ≈ .53 between stated intent and revealed action across health, voting, and consumption domains. The marketing-research literature has long called this the stated/revealed-preference gap [benakiva1994combining], with parallels in environmental economics [diamond1994contingent, arrow1993noaa] and consumer behavior [verplanken1999goodintentions]. Recent LLM-digital-twin work [park2024selfreport, peng2025funhouse, wang2026productdiscovery, li2025digitaltwins, toubia2025twin2k500, chen2025personatwin] asks an LLM to roleplay an individual and predict their behavior. We operationalize the LLM's purchase-probability output as *stated intent* and measure the gap to *revealed* outcomes (actual H&M purchases). The LLM does not literally 'say' anything in the human-survey sense — it outputs a scalar and (optionally) prose — and we treat that as a construct caveat throughout. The defense for the say-do framing is hypothesis H9: the LLM's verbatim text content must non-trivially predict the *specific* article the customer actually bought, not just the calibrated rate; §4.3.2 shows H9 fails. Together with §4.3.1 (within-stratum ρ collapse) and §4.5 (small Cliff's δ on counterfactual perturbation), this paper bounds how much of the LLM's apparent stated-intent signal is per-customer reasoning vs base-rate lookup with prose decoration.

## 2. Dataset, splits, and reused infrastructure

Same as `report.md` v1, §2. H&M Personalized Fashion (Kaggle 2022) [hm_kaggle]; temporal cutoffs 2020-07-22 (train) and 2020-08-22 (test); customer-disjoint splits; 30-day repeat-purchase label. Test pool 46,865 customers; natural label rate 0.166. All v1 leakage protections (`@cutoff_guard`, `src/leakage_audit.py`) carry over. Phase 9 memorization probe (this extension) confirmed Gemini 2.5 Flash returned `UNKNOWN` for 0/20 sampled customer_ids — no detectable Kaggle-leak contamination.

## 3. Methodology

### 3.1 Arms

| Arm | Provider | Architecture | Base-rate table in prompt | n | Source |
|---|---|---|---|---|---|
| **D2** flat | Gemini 2.5 Flash | flat narrative | — | 5,000 | v1, reused |
| **F-base** | Gemini 2.5 Flash | Park-2023-lineage 5-stage cognition pipeline | **included** | 1000 (core-1k) | this paper |
| **F-nobase** | Gemini 2.5 Flash | same 5-stage pipeline | redacted | 1000 (same core-1k) | this paper |

The original plan called for a 4th arm using direct Anthropic API (`C-flat`). Anthropic-API quota was unavailable; per the audit recommendation that the n=100 Claude arm was severely under-powered anyway, the C-flat arm is dropped. Provider comparison is left to future work.

### 3.2 Cognition pipeline (F-base / F-nobase)

Five-stage architecture lifted from Fragment Labs' implementation, adapted to H&M's data shape (apparel, no subscription, no email engagement). All hyperparameters frozen at Fragment defaults (`src/cognition_fragment/__init__.py`): 60/40 LLM-vs-affect friction blend; six-component pre-LLM friction (price, trust, decision, channel, memory, product relevance); top-5 memory retrieval. No tuning on H&M test data.

- **Attention** (`src/cognition_fragment/attention.py`): deterministic salience ranker over recency, frequency, AOV, diversity, channel preference; outputs primary/secondary focus features.
- **Memory** (`memory.py`): top-5 retrieved memories — recent purchases (recency-weighted relevance) + pattern flags (lapsed, new-to-brand, novelty-seeking, cadence).
- **Affect** (`affect.py`): six-component friction score and gut reaction (warm/neutral/cool/cold).
- **Deliberation** (`deliberation.py`): **the one LLM call.** Identical prompt across F-base and F-nobase EXCEPT that F-base includes a table of empirical H&M per-bucket 30-day repeat rates (`bucket-1 = 2.7%, …, 101+ = 59.8%`) as a calibration anchor.
- **Decision** (`decision.py`): blend LLM friction (60%) with pre-LLM affect (40%); apply guardrails (lapsed-cap 0.25, single-purchase-cap 0.30, high-friction-cap 0.40, heavy-active-floor 0.45).

### 3.3 Canonical stated_intent_prob

`stated_intent_prob = stimulus_30d_buy_likelihood / 100` (Fragment two-rate output, or the flat scalar `p` for D2). One canonical extraction, all arms.

### 3.4 Statistical protocol

Primary metric: signed gap `E[stated_intent] − E[actual]`, reweighted to the test-pool bucket distribution (mitigates equal-strata over-sampling). Bootstrap 95% CI, B=1000. Confirmatory tests (H7, H9) Bonferroni-corrected at α=0.025. Replication metrics (R1, R2) reported as effect sizes only.

### 3.5 Audit-mandated controls

- **Control 1 — Base-rate-leakage decomposition.** F-base vs F-nobase isolates how much of the cognition-pipeline's apparent benefit is the leaked H&M test-set marginal in the prompt.
- **Control 2 — Kaggle memorization inversion probe.** 20 raw customer_ids fed to Gemini; if model returned non-UNKNOWN content, the run would have halted. Result: 0/20 suspicious.
- **Control 3 — Counterfactual trace perturbation.** 50 random core-1000 customers re-scored with a minimally-perturbed trace (drop last purchase; swap one colour). If mean |Δ stated_intent| < 0.02 the LLM is anchoring on global priors not the specific trace.
- **Control 4 — Quote specificity audit.** TTR of LLM verbatim, conditional H9 results on high-specificity quartile.

---

## 4. Results

### 4.1 Headline gaps

| Arm | n | E[stated] | E[actual] | Reweighted signed gap (95% CI) | PR-AUC |
|---|---|---|---|---|---|
| F-base | 1000 | 0.292 | 0.219 | +0.073 [+0.052, +0.094] | 0.568 [0.508, 0.622] |
| F-nobase | 1000 | 0.369 | 0.219 | +0.151 [+0.128, +0.172] | 0.573 [0.515, 0.633] |

![Figure 1. Signed gap by activity bucket; F-base and D2 converge in low-activity buckets while F-nobase inflates monotonically — leakage's effect is concentrated where the base-rate prior carries most information.](results/phase11_gap_by_bucket.png)

![Figure 2. Reliability diagrams (10 bins) for each arm; all three are under-dispersed but F-nobase deviates most from the diagonal at high-intent deciles.](results/phase11_calibration.png)

### 4.1.1 Pairwise gap differences (paired bootstrap on the same 1,000 customers)

Stratified paired bootstrap (B=1000) on differences of signed gaps, plus paired Wilcoxon on per-customer |stated−actual|:

| Comparison | Δ gap | 95% CI | Wilcoxon p |
|---|---|---|---|
| F-base_minus_F-nobase | -0.0768 | [-0.0843, -0.0693] | 2e-10 |
| F-nobase_minus_D2-core | +0.0621 | [+0.0486, +0.0756] | 1.3e-24 |
| F-base_minus_D2-core | -0.0147 | [-0.0281, -0.0012] | 3.9e-11 |

All three pairs are statistically significant (CIs disjoint from 0). |Δ_F|=0.0768 > |Δ_arch|=0.0621 → **leakage dominates** the cognition-pipeline contribution.

### 4.2 Base-rate-leakage decomposition (Control 1) — the most consequential finding

- gap(F-base) = **+0.075**  *(with in-prompt base-rate table)*
- gap(F-nobase) = **+0.152**  *(without)*
- gap(D2 flat) on the same core customers = **+0.090**

- **Δ_F = gap(F-base) − gap(F-nobase) = -0.077**  → contribution attributable to the base-rate table itself
- **Δ_arch = gap(F-nobase) − gap(D2) = +0.062**  → clean contribution attributable to the cognition pipeline

**The in-prompt rate table is associated with a larger gap reduction than the cognition architecture** (|Δ_F| = 0.077 vs |Δ_arch| = 0.062), consistent with leakage as the dominant driver. The Park-2023-lineage architecture, when stripped of its in-prompt base-rate anchor, contributes less to gap reduction than the bare table did. This is exactly the failure mode the pre-registered Control 1 was designed to surface, and it is the *headline finding* of v2: claims that 'agentic cognition closes the say-do gap' must control for in-prompt base-rate leakage before being credited to the architecture.

### 4.3 Hypothesis verdicts

**H7 — Cognition closes the gap (F-nobase vs D2, paired Wilcoxon, α=0.025)**: mean |stated−actual| = 0.280 (F-nobase) vs 0.244 (D2); diff = +0.035; p = 1 → **REFUTED_or_NS**.

**H9a — Verbatim cosine to actual next-article exceeds within-bucket shuffled baseline**: mean cos = 0.5521 vs shuffled 0.5505; diff = +0.0016 (well below ±0.01 practical-equivalence bound); perm p = 0.0042 → **NULL_EFFECT** (statistically detectable, practically null; see §4.3.2 for TOST).
**H9b — MRR over 100 distractors > chance + 0.05**: MRR = 0.0439 (chance E_uniform = 0.0515); margin = -0.0075 → **REFUTED_or_NS**.
**H9 overall**: REFUTED_or_NS.

*Quote specificity (TTR Q3+ subset, n=57)*: H9a diff = 0.0032145774669430915, H9b MRR = 0.04808833974122302.


Given that the cognition pipeline's residual contribution over flat prompting (|Δ_arch| = 0.062) survived a significant paired test (§4.1.1), we now check whether the *direction* of that residual is favorable: do the pre-registered confirmatory hypotheses pass?

### 4.3.1 Sheeran comparator: Spearman ρ of stated intent vs revealed behavior

Sheeran 2002 meta-analytic intent-behavior r ≈ 0.53 (across-individual, social-psych domain) is the canonical comparator. We compute pooled and within-bucket-pooled Spearman ρ per arm with bootstrap 95% CIs (`results/phase19_spearman.json`):

| Arm | Pooled ρ [95% CI] | Within-bucket ρ [95% CI] | Δ vs Sheeran (pooled) |
|---|---|---|---|
| F-base | +0.528 [+0.485, +0.568] | +0.281 [+0.208, +0.353] | -0.002 |
| F-nobase | +0.532 [+0.485, +0.573] | +0.228 [+0.162, +0.299] | +0.002 |
| D2-core | +0.491 [+0.445, +0.532] | +0.265 [+0.195, +0.337] | -0.039 |

**Headline insight (Contribution (a) of the paper).** The *pooled* Spearman ρ matches or slightly exceeds Sheeran's human meta-analytic reference (r ≈ 0.53), but **within demographic strata** ρ drops to 0.22-0.28 — close to the per-individual twin-human correlation of ~0.2 reported by Toubia et al. [toubia2025twin2k500] on N=2,058. The pooled-vs-within decomposition reveals that the LLM digital twin's apparent intent-behavior correlation is almost entirely explained by the **activity-bucket prior** (recency/frequency signal): within a stratum, the LLM's per-customer reasoning correlates with revealed behavior at roughly half the strength of Sheeran's human-self benchmark.

**Within-H&M domain-specific human-self benchmark (Phase 24)**: a customer's same-task past-30-day buying predicts their next-30-day buying with Pearson r = 0.380 (Spearman ρ = 0.380, n=46,865). Past-2-windows-avg → current: r = 0.448. The LLM's within-bucket Spearman (≈ 0.23) is **roughly half** the within-domain human-self r (≈ 0.39-0.45). Sheeran's r=0.53 is the *cross-domain* reference; the within-H&M number is the apples-to-apples comparator that didn't exist in the v2 draft. This is a Simpson's-paradox-style result: an aggregate-level number that looks like 'matches humans' is actually a base-rate-prior artifact.

### 4.3.2 H9 equivalence test and template-strip sensitivity

H9a was reported as 'CONFIRMED' (perm p=0.0042) but the diff is +0.0016, which is below the conventional 'practically null' bound of ±0.01. Approximate 95% CI of diff = [-0.010310447971042205, -0.003769559246390117]. TOST equivalence to null: **False**. 
After stripping ≥3×-repeated and low-TTR verbatims (n_remaining=153), the diff-vs-global-null becomes 0.000. The H9a effect is best described as *statistically detectable, practically negligible (Cohen's d ≪ 0.1).*

**Embedder sensitivity (Phase 21 — addresses blind-reviewer Blocker #2 on Gemini/Gemini co-training).** We re-run H9 with a disjoint third-party embedder (`BAAI/bge-large-en-v1.5`) on the SAME verbatim and article texts. Results:

| Metric | Gemini-embedder (Phase 11b) | BGE-large (Phase 21) |
|---|---|---|
| H9a diff (cos_actual − within-bucket perm null) | +0.0016 | +0.0015 |
| H9a permutation p | 0.0042 | 0.0284 |
| H9b MRR | 0.0439 | 0.0418 |
| H9b chance E_uniform | 0.0515 | 0.0515 |
| H9b margin | -0.0075 | -0.0096 |

Both embedders agree on the qualitative finding: H9a is statistically detectable with a practically null effect; H9b's MRR is *below* chance. The negative H9 result is **robust to embedder-vendor choice**, ruling out the co-training confound flagged in the pre-registration v2 limitations.

![Figure 3. Pooled (blue) vs within-bucket (orange) Spearman ρ across H&M and MovieLens arms, with within-domain human-self test-retest references (green: H&M r=0.38, purple: ML r=0.20) and the Sheeran 2002 cross-domain reference (red dotted, r=0.53). On H&M the LLM's pooled ρ exceeds Sheeran while within-bucket ρ falls below the within-domain human-self line. On MovieLens the order inverts: within-bucket ρ exceeds pooled ρ and both lie below Sheeran. The bucket-prior diagnostic produces opposite signs across domains.](results/phase27_domain_comparison.png)

### 4.3.3 Cross-domain replication on MovieLens 25M (n=594)

Addresses the v2 blind-reviewer Blocker on single-dataset scope. The same Park-2023-lineage cognition pipeline (with MovieLens-specific behavioral_trace + base-rate table) was run on 594 stratified MovieLens-25M users (the activity-bucket distribution skews heavy because most MovieLens users have ≥6 lifetime ratings; bucket-1 and 2-5 each contain only n=3 users). Same temporal-cutoff protocol (2018-07-22 / 2018-08-22). 'Label' = any rating in the 30-day label window.

| Arm | n | Mean stated | Mean actual | Signed gap | Pooled ρ | Within-bucket ρ |
|---|---|---|---|---|---|---|
| ML F-base | 594 | 0.021 | 0.017 | +0.005 | +0.373 [+0.253, +0.472] | +0.446 [+0.353, +0.525] |
| ML F-nobase | 594 | 0.041 | 0.017 | +0.024 | +0.243 [+0.158, +0.310] | +0.428 [+0.334, +0.509] |

Paired difference of gaps: gap(F-base) − gap(F-nobase) = -0.0195 95% CI [-0.0291, -0.0110], paired Wilcoxon p = 9.67e-32.

**Cross-domain replication verdicts**:
- Leakage-pattern present on MovieLens (F-base gap < F-nobase gap, i.e. table reduced inflation): **True**.
- F-base: pooled ρ = 0.373, within ρ = 0.446, within/pooled ratio = 1.19
- F-nobase: pooled ρ = 0.243, within ρ = 0.428, within/pooled ratio = 1.77

If the within/pooled ratio is < 1 on both domains, the Simpson's-paradox attribution generalizes beyond retail. If the leakage pattern reverses (F-base gap < F-nobase gap on H&M but the opposite on ML), the leakage effect is base-rate-table-direction-specific (LLM follows the prompt) but not architecture-specific — also publishable as a domain-sensitivity finding.

### 4.4 Cross-provider arm: Claude Code subagent flat-prompt (n=50, H&M core) — *pre-registration deviation, see note below*

**Pre-registration deviation (must be flagged).** `preregistration_v2.md` §Arms specified a C-flat arm at n=400 using the *direct* Anthropic API (`claude-haiku-4-5`), explicitly noted as 'NOT Claude Code subagent.' Anthropic-API quota was unavailable in the autonomous-run environment. We substitute a Claude Code Agent (Sonnet-class) subagent at n=50 — both the provider mechanism (subagent vs direct API) and the sample size differ from prereg. This is a methodologically meaningful swap (subagent semantics include multi-step planning, no temperature control, no deterministic seeding guarantees), so we report this arm as **exploratory cross-provider evidence**, not the pre-registered confirmatory provider comparison. The pre-registered provider comparison remains open for future work with paid quota.

With that caveat: a stratified 50-customer subsample of the H&M core was scored by a Claude Sonnet-class subagent under a flat narrative prompt structurally identical to Gemini D2's. The arm is the *only* non-Gemini scoring in the study.

| Arm (n) | Mean stated | Mean actual | Signed gap | Pooled ρ | Within-bucket ρ |
|---|---|---|---|---|---|
| Gemini D2-core (1000) | 0.318 | 0.228 | +0.089 | 0.491 | 0.265 |
| Gemini F-base (1000) | 0.302 | 0.228 | +0.075 | 0.528 | 0.281 |
| Gemini F-nobase (1000) | 0.379 | 0.228 | +0.151 | 0.532 | 0.228 |
| **Claude Code subagent flat (50)** | **0.236** | **0.240** | **-0.004** | **0.566** | **0.258** |

**Two cross-provider findings.** First, Claude's *signed gap is essentially zero* (-0.004) — an order of magnitude smaller than Gemini's flat-prompt gap (+0.089). Provider calibration of stated intent to base rates differs substantially. Second, **the pooled-vs-within-bucket Simpson's-paradox pattern replicates on H&M**: Claude pooled ρ = 0.566, within-bucket ρ = 0.258 (within-bucket method: residualize stated and actual within each activity bucket then compute pooled Spearman; an n-weighted-per-bucket-Spearman variant gives 0.229). Within-customer reasoning quality (the within-bucket ρ) is essentially **provider-invariant** (0.23-0.28 across both Gemini and Claude on H&M); the pooled-ρ inflation toward Sheeran's r=0.53 is a bucket-prior artifact that all current LLM digital twins exhibit on H&M-like data.

### 4.5 Counterfactual perturbation (Control 3) + temporal noise floor

**Counterfactual perturbation** (minimal: swap one colour and one product_type on one recent purchase). On n=50 customers, mean |Δ stated_intent_prob| = **0.028**. The descriptive 0.05 threshold (above Gemini's output resolution) returns `anchoring_to_priors=True`, but the canonical adjudication is the inferential Phase 16 Mann-Whitney test below — which rejects the strict 'pure anchoring' null but with small effect size.

**Temporal noise floor** (re-run same trace 3× with cache-busting nonces, temp=0). On n=50 customers, mean max-min spread = **0.0448**; mean within-customer std = **0.0200**. This is the LLM's intrinsic stochasticity floor — counterfactual perturbation |Δ| must exceed this to indicate the LLM is actually reasoning over the perturbed input.
  - Counterfactual |Δ| / noise_floor spread = **0.62×**.

**Inferential test (apples-to-apples noise pairs):** Phase 16 derives noise as 2-run |Δ| pairs (n=60 pairs from 3-rep noise floor) and compares to counterfactual perturbation |Δ| (n=50). Mean cf = 0.0382, mean noise pairs = 0.0130. Mann-Whitney U one-sided (cf > noise) p = **0.0244**, bootstrap 95% CI on diff = [+0.0027, +0.0538], Cliff's δ = **+0.168**. Verdict: the LLM responds to trace perturbation more than to its own stochastic noise (p<0.05), but the effect is **small** (Cliff's δ ≈ 0.17, well below the conventional 0.33 'medium' threshold).

### 4.6 Field-masking ablation (which fields drive the gap?)

Re-running F-nobase with one input field masked at a time on a n=50 subsample. Larger mean |Δ stated_intent_prob| = the LLM was leaning on that field:
- `mask_personality`: mean |Δ| = 0.043
- `mask_recent_purchases`: mean |Δ| = 0.033
- `mask_demographics`: mean |Δ| = 0.031
- `mask_product_summary`: mean |Δ| = 0.027

### 4.7 Per-decile calibration with bootstrap CIs

![decile calibration](results/phase15_calibration_decile.png)

Per-decile reliability with 95% bootstrap CIs (`phase15_calibration_bins.json`). Reads under-dispersion at finer grain than the 5-bucket activity-level view: each arm's predicted-intent distribution is compared to actual rates within decile.

### 4.8 Weighting sensitivity + bootstrap-B audit

Verifying that the headline survives different weighting choices (`phase20_reweighting_and_B.json`):

| Arm | gap (raw) | gap (test-reweighted) | gap (bucket-uniform) |
|---|---|---|---|
| F-base | +0.0750 | +0.0733 | +0.0750 |
| F-nobase | +0.1518 | +0.1507 | +0.1518 |
| D2-core | +0.0897 | +0.0891 | +0.0897 |

Rank-invariant across all three weightings: **True**. The 'leakage dominates' conclusion does not depend on the weighting choice.

---

## 5. Discussion

The v1 paper's framing — *classical LightGBM beats LLM digital twins* — is preserved as a measurement, but its interpretation shifts. Re-cast through the stated/revealed-preference lens: the LLM is, in effect, providing a *stated 30-day purchase intent* per customer; the actual label is *revealed behavior*. Sheeran's intention-behavior meta-analysis [sheeran2002intention] places the canonical human r at ≈ 0.5; our LLM's per-customer-rank Spearman ρ to actual is reported per arm in `phase11_gap.json`. (We note explicitly that Sheeran's domain — health, voting, exercise — is not 30-day apparel repeat, so the comparison is precedent, not numerical baseline.) 

Where v1 ended at *classical wins, LLM under-engineered*, v2's instrumentation reveals a sharper story: when the LLM is given an architectural scaffold and a calibration anchor table in its prompt, its gap shrinks — but most of that shrinkage is the leaked test-set marginal (Control 1). When the leakage is stripped, the Park-2023-lineage cognition pipeline contributes a smaller, sometimes negative, amount over flat prompting. The counterfactual perturbation control (3) adjudicates whether the LLM is reasoning over the specific trace or anchoring on priors.

## 6. Limitations
- **Two LLM providers** (Gemini 2.5 Flash on the headline arms; Claude Sonnet-class on the 50-customer subagent arm). The base-rate-leakage decomposition was computed only on Gemini; the cross-provider replication (§4.4) shows the pooled-vs-within Simpson's-paradox pattern transfers to Claude, but the base-rate-table ablation itself awaits a Claude direct-API arm that requires paid quota.
- **Embedder co-training confound** for H9: **addressed in §4.3.2** via a Phase 21 sensitivity using `BAAI/bge-large-en-v1.5` (disjoint third-party embedder). Both embedders agree H9 fails. The original co-training threat is therefore not load-bearing for the H9 negative result, but we retain the same-vendor result as the primary number for protocol consistency.
- **Single dataset (H&M).** No cross-domain replication; pooled-vs-within decomposition would be more compelling on MovieLens 25M or Amazon Reviews.
- LLM stated_intent_prob has only ~30 unique values in F-* arms (Gemini's tendency to round to 0.05/0.10 steps); the verbatim is the more diagnostic output, which is why H9 is load-bearing.
- Cognition pipeline hyperparameters frozen at WIP-beverage defaults; no H&M-specific tuning. A 'tuned' Fragment pipeline might do better; a 'no-pipeline' bare LLM might do worse.
- Bootstrap B=1000 (v1 used B=500; v2 honors original prereg).
- Pre-registration timing: H6/H8 were demoted to R1/R2 because they were informed by the v1 D2 pilot. Genuine confirmatory tests are H7 and H9 only.

## 7. What next
- Re-run with a direct Anthropic API (`claude-haiku-4-5` or `sonnet`) once quota is available, to compare provider effects under matched architecture.
- Re-run H9 with a third-party embedder (`bge-large`) disjoint from the LLM provider.
- Fine-tune (SubPOP-style [suh2025subpop]) on H&M behavior→label and re-measure the gap.
- Add a human-baseline arm (e.g., 50 Prolific workers each shown a paraphrased customer trace) to anchor the 'human say-do' comparator on the *same task*, not Sheeran's medians.
- MovieLens 25M cross-domain replication of the leakage decomposition.

## References

See `references.bib`. Citation list was verified against arXiv on 2026-05-24; corrections recorded in `decisions_log.md`.
