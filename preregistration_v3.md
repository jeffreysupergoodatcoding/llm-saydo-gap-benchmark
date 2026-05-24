# Preregistration v3 — A Consequential-Decision Sandbox and a Method Catalog for Closing the LLM Say-Do Gap

**This is the preregistration for the MAIN PAPER.** The v1/v2 benchmarking (now archived as `paper.md`) serves as background motivation; v3 is not an extension but the headline contribution. Committed before any v3 LLM calls. Pre-registration hash recorded at end. Cutoff: 2026-05-24. Replaces preregistration_v3 draft 1, which was rewritten in full after a tri-agent audit identified that the original sandbox collapsed to a 3-gate AND of binary choices and that 5 of the 9 proposed methods were generic prompt-engineering tricks that would behave identically as v2 single-shot calls.

## 1. Central question

**Can published or theoretically motivated interventions close the LLM digital-twin say-do gap inside a consequential decision sandbox, or is the gap structurally bound to the way current LLMs simulate individuals?**

If yes: which interventions, by how much, and at what cost in complexity?
If no: report negative results truthfully, document what each intervention does and doesn't do, and articulate the open problem.

## 2. Headline thesis (single sentence)

> Across eight literature-grounded and sandbox-native interventions on a 30-day H&M behavioral sandbox with exogenous stochastic stimuli and a depletable attention budget, no method recovers individual-level customer signal beyond a base-rate floor; gap reductions are explainable by mean-shrinkage toward the prior rather than by genuine within-customer reasoning, implying the say-do gap is **not currently solvable** by prompt-, retrieval-, or agent-policy-level interventions on third-party LLMs without explicit individual-conditioning data.

This is the *hypothesis-as-stated* before we run experiments. If results contradict it, we update the headline accordingly and report the contradiction honestly.

## 3. Why the prior benchmarking matters as background

`paper.md` (v1/v2) established:
- 7-point signed say-do gap on flat-prompt Gemini twin.
- ≤2 points attributable to Park-2023-lineage cognition architecture; remainder explainable by in-prompt base-rate-table leakage.
- Cross-domain partial replication on MovieLens (with honest acknowledgment of Simpson's paradox inversion).
- Within-domain human-self anchor: customers themselves predict their own next-window purchase at Pearson r ≈ 0.20.

These establish the *what* (a real, replicable gap with a known leakage component) that motivates the *how-to-close* question of this paper.

## 4. What's new in this paper

- **A consequential-decision sandbox** with three properties the audit identified as load-bearing for non-collapse: (i) exogenous stochastic stimuli, (ii) a depletable per-window attention/budget resource, (iii) persistent state across cycles (prior commitments visible to later weeks).
- **A method catalog** of 8 methods: 4 literature baselines that *exist* in say-do-gap and prompt-engineering work, and 4 *sandbox-native* methods that single-shot scalar elicitation cannot represent at all.
- **A new primary metric — `commitment_shrinkage`** — with a permutation null so we don't measure a self-consistency tautology.
- **A new primary metric — `funnel_realized_purchase_rate`** that is a true behavioral measurement of the agent itself (the agent's actions inside the sandbox, not its self-reported scalar).
- **A structured counter-claims section** addressing four specific reviewer attacks identified by the audit (see §11).

## 3. Pre-registered hypotheses (Bonferroni α = 0.025 over confirmatory)

### Confirmatory (two)

- **H10 (sandbox closure)**: At least one method M ∈ {M1, M3, M8, M9, S1, S2, S3, S4} achieves `|signed_gap_sandbox(M)|` ≤ 0.05 on the core-1000 sandbox set, where `signed_gap_sandbox(M) = E[funnel_realized_purchase_rate(M)] − E[f_actual]`. Paired bootstrap CI (B=1000, stratified within bucket) excludes ±0.05 envelope.
  - Envelope widened from ±0.04 → ±0.05 in response to MDE audit (paired-SE at n=400 stratified ≈ 0.018; ±0.05 envelope is ≈ 2.7σ — achievable).
- **H11 (sandbox-native methods bring sandbox-specific signal)**: At least one sandbox-native method S ∈ {S1, S2, S3, S4} achieves strictly larger within-bucket Spearman ρ (between agent score and `f_actual`, computed within each activity bucket and weighted by bucket size) than zero-shot M1 on core-1000, by ≥ 0.03. Paired bootstrap CI excludes 0.
  - Within-bucket ρ is preregistered because (per audit) the v2 result is that pooled ρ is artificially inflated by base-rate variance; only within-bucket ρ measures genuine individual-conditioning. This is the quantity LightGBM ensembling cannot trivially dominate.

### Exploratory (effect-size only, no NHST)

- **R3**: Sandbox-method ranking correlates with the same method's published efficacy on intent-behavior tasks in the social-psychology literature (Spearman ρ, n=8).
- **R4**: `commitment_shrinkage(M)` = `scalar_gap(M) − sandbox_gap(M)` is non-zero net of a permutation null in which DP outcomes are shuffled within each method's marginal action distribution.
- **R5**: Adding budget pressure tightens commitment_shrinkage; reported by re-running M1 with budget=∞ as ablation.

### Dropped from draft 1

- Original H10 (|gap|<0.04 at n=200) — too tight against measured paired SE; widened to 0.05 at n=400.
- Original H11 (M7 hybrid beats M1) — audit flagged as trivially true because LightGBM (PR-AUC 0.622) dominates LLM (PR-AUC 0.57); replaced with within-bucket-ρ test that LightGBM cannot dominate.

## 4. The sandbox (v3 redesigned)

### 4.1 State

```
state := {
  customer_trace_pre_cutoff,    # frozen; from v2 behavioral_trace
  funnel_history_window,        # list[(week, dp, action, item)] — visible to agent in later DPs
  attention_budget,              # int starting at B=3, depleted on ENGAGE/CONSIDER
  week_t,                        # 0..3 within 30-day window
  stimulus_t                     # see §4.2 — stochastic, exogenous, 3-candidate menu
}
```

### 4.2 Stimuli (the audit-mandated exogenous component)

For each (customer, week), `stimulus_t` is a **3-candidate menu** drawn deterministically from data with seed = HASH(customer_id, week):

- **Candidate A — in-section**: most-frequently-purchased index_group (section) for this customer historically.
- **Candidate B — cross-section**: a different section, sampled inversely proportional to customer's prior frequency in it (a stretch item).
- **Candidate C — OOD**: a globally-popular item drawn from a section the customer has never purchased in.

The agent sees descriptive blurbs (product_type, colour, garment_group) but not customer_id and not the section labels. The agent's DP2 CONSIDER action *names* which candidate it engages with.

### 4.3 Action space (3 DPs per week, 4 weeks)

| DP | Question | Actions | Budget cost |
|---|---|---|---|
| DP1 | Stimulus arrives | SKIP / ENGAGE | ENGAGE costs 1 |
| DP2 (if ENGAGE) | Browse menu | EXIT / CONSIDER(A∣B∣C) | CONSIDER costs 1 |
| DP3 (if CONSIDER) | Decide on item | ABANDON / PURCHASE | PURCHASE costs 1 |

`attention_budget` starts at 3, depleted across all weeks. When budget = 0, all DP1s force-SKIP. **This is the consequential resource** — the agent must decide which week's stimulus is worth its remaining attention.

### 4.4 Persistence

The full `funnel_history_window` (prior weeks' actions including candidate IDs and prior purchases) is shown to the agent at every DP. This is what allows methods like **S1 Reflexion** and **S3 Tree-of-thoughts** to use multi-step structure.

### 4.5 Sandbox outcome and labels

Per customer per method M:

- `f̂_LLM(M) = 1` iff PURCHASE occurred in ≥ 1 week within the 30-day window.
- `f_actual` = 1 iff the customer actually purchased ≥ 1 item in the actual 30-day window (same label as v2).
- `chosen_item(M, w)` = candidate {A,B,C} picked at DP2 if reached; for DP2-validation, we compute MRR of `chosen_item` against the customer's actual next-purchased article by article-description cosine in BGE-small space (H9-style; see paper.md §6.3 H9b for the protocol).

### 4.6 Sandbox metrics

| Metric | Definition |
|---|---|
| `funnel_realized_purchase_rate(M)` | mean over n of `f̂_LLM(M)` |
| `sandbox_signed_gap(M)` | `funnel_realized_purchase_rate(M) − E[f_actual]` |
| `scalar_signed_gap(M)` | `E[scalar_prob(M)] − E[f_actual]` (M1-style scalar from existing pipeline) |
| `commitment_shrinkage(M)` | `scalar_signed_gap(M) − sandbox_signed_gap(M)`; tested vs DP-shuffle null |
| `within_bucket_ρ(M)` | bucket-weighted Spearman of agent score vs `f_actual` |
| `DP2_chosen_item_MRR(M)` | MRR of chosen candidate against actual next-purchase content embedding |

### 4.7 Sample size

**core-1000 = 200 customers × 5 activity buckets**, customer-disjoint from v2 core-1000 (drawn from the remaining 45,865 test-set customers), drawn deterministically with seed=2026. Stratified within bucket. Audit MDE: paired SE ≈ 0.011 at n=1000 stratified → ±0.05 envelope is detectable at >99% power; this gives the headline thesis a fair chance of being falsified.

Sample size scaled up from draft 1 (n=400) in response to user request for "large-scale, statistically robust" testing.

### 4.8 Cost

- 1000 customers × up to 12 DPs × 8 methods. With method-specific overheads (S1 doubles DP calls for self-critique; S2/S4 add 1 setup call; S3 single ToT call per DP):
- Net Gemini Flash 2.5 calls ≈ 1000 × ~65 ≈ 65,000. At ~$0.000158/call ≈ $10.30. Hard cost cap raised to **$25** with three safety circuit-breakers: (i) per-method cost ceiling $4, (ii) per-method n-cap if cost projected to exceed (iii) hard kill at $25 total session cost.
- Also: $1 scalar-arm reuse (8 methods × 1000 customers × 1 scalar) ≈ $1.27. Total budget request: **$12** mid-case, **$25** hard cap.

## 5. The 8 methods (revised after audit)

### Literature baselines (4)

| ID | Name | What it does | Sandbox use of structure |
|---|---|---|---|
| **M1** | Zero-shot | Vanilla per-DP prompt with customer trace and stimulus | Reference (no extra structure) |
| **M3** | Few-shot k-NN ICL | 5 RFM-nearest customers + their full funnel histories with funnel-action labels prepended | Mild (uses funnel as label) |
| **M8** | RAG with outcome labels | Retrieve 5 prior trajectories *with their realized funnel outcomes*; keys include funnel state at the DP | Strong (per-DP retrieval) |
| **M9** | Implementation-intentions prompt | Gollwitzer "if X then Y" forced format; tested at DP1 with a forward plan | Mild (single declarative) |

### Sandbox-native (4)

| ID | Name | What it does | Why this requires the sandbox |
|---|---|---|---|
| **S1** | Reflexion-in-funnel | After each DP, agent emits a 1-sentence self-critique that is appended to `funnel_history_window` and is read at the next DP | Requires multi-step structure — no analog in single-shot v2 |
| **S2** | Outcome-conditioned planning | Before DP1, agent is asked to first *imagine* the PURCHASE leaf, write the backward trajectory from PURCHASE → DP3 → DP2 → DP1 the customer would have taken, and *then* commit DP1 | Forces path-consistency between intention and the action sequence; cannot exist without a path |
| **S3** | Tree-of-thoughts over funnel branches | Agent enumerates all 8 funnel rollouts (SKIP / ENGAGE×EXIT / ENGAGE×CONSIDER×{A,B,C}×{ABANDON,PURCHASE}), self-scores each with a 0–10 plausibility rubric, picks the highest-scored | Uses the deterministic-transition assumption *as a feature* (audit framing accepted) |
| **S4** | Commitment device | At DP1 the agent declares a maximum number of purchases this window (0/1/2/3) and a maximum attention spend; later DPs are hard-constrained to that declaration | Directly tests the Gollwitzer mechanism *through the sandbox structure*, not through prompt format |

### Cut from draft 1 (with audit reasoning)

- **M2** Few-shot random ICL — dominated by M3 (same construct, weaker version).
- **M4** Self-consistency — pre-registered as variance ↓ not bias ↓; doesn't speak to the gap.
- **M5** Chain-of-thought — generic; temperature-0 JSON already constrains reasoning.
- **M6** Isotonic calibration — non-architectural; v2 already showed base-rate leakage explains most scalar-gap variance.
- **M7** Hybrid LLM+LGBM — H11 audit verdict: trivially wins via the stronger predictor's mean.

(M6 + M7 are reported as a *footnote sanity check* on the scalar arm only — not in the sandbox comparison.)

## 6. Analysis plan

1. Compute scalar and sandbox metrics for M1, M3, M8, M9, S1, S2, S3, S4 on core-1000.
2. Tabulate per-method: scalar_gap, sandbox_gap, commitment_shrinkage, within_bucket_ρ, DP2_chosen_item_MRR; bootstrap CIs.
3. **H10 adjudication**: which methods cross |sandbox_gap| ≤ 0.05 with CI excluding ±0.05.
4. **H11 adjudication**: paired bootstrap of within_bucket_ρ(S*) − within_bucket_ρ(M1).
5. **R3**: literature-strength prior (S2 > M3 > M9 > others) Spearman-correlated with measured ρ.
6. **R4**: permutation null for `commitment_shrinkage` (DP outcomes shuffled within method's marginal action distribution); does any method beat the null?
7. **R5**: ablation with budget=∞ on M1 only — does removing budget pressure absorb commitment_shrinkage?
8. **Per-bucket breakdown** of sandbox_gap (5 buckets × 8 methods).
9. **Memorization probe** on the new core-1000 IDs (HMAC reuse).
10. **Counterfactual perturbation** in the sandbox: 50 customers with one swapped historical purchase color/type; measure mean |Δ f̂_LLM| and Δ chosen_item.

## 7. Stop conditions

- If memorization-probe inversion succeeds on any core-1000 ID: halt and reframe.
- If sandbox-vs-scalar correlation across methods is < 0.3 *and* sandbox doesn't beat scalar on any preregistered metric: treat sandbox as a separate construct, not a refinement, and reframe paper around the divergence itself.

## 11. Counter-claims and limitations addressed pre-commit

- **"Your sandbox is just a structured prompt; it has no environment dynamics."** Accepted partially. We add exogenous stochastic stimuli (§4.2) and a depletable budget (§4.3) so that DP1 is consequential (skipping now preserves budget for later weeks). Transitions remain deterministic because adding transition stochasticity confounds credit assignment across methods; we will frame the sandbox as *decision-elicitation under commitment pressure*, not as a world model.
- **"You're going to find no method works and call that the result."** Risk is real. To pre-commit against motivated reasoning: (i) headline thesis above is the *expected* outcome — if any method clears H10, we report it as the headline; (ii) per-method ranking is published regardless; (iii) `within_bucket_ρ` is reported with bootstrap CIs for every method; (iv) `commitment_shrinkage` permutation null is set in stone.
- **"You can't claim third-party LLMs can't do it when you only ran one (Gemini Flash)."** Accepted. We add a Claude Code subagent arm for M1, M3 only (cost-bounded) as a provider-replication sensitivity check.
- **"The 8-method comparison is multiple-testing inflation."** We apply Bonferroni α=0.025 over confirmatory H10+H11, exactly as stated. Per-method scalar gaps are reported descriptively without significance asterisks.

## 12. Out of scope (held to future work, with reason)

- Stochastic *transitions* (vs stochastic stimuli, which we do include): deferred because adding transition stochasticity confounds method-comparison credit assignment.
- Multi-customer interaction sandboxes (e.g., cohort dynamics, price-elasticity feedback).
- Fine-tuning interventions. v3 stays at prompt / retrieval / agent-policy level.
- Real-money A/B against an actual H&M holdout — no access.

## 9. Citations to add for the v3 methods (verified pre-commit)

- Madaan et al. 2023 "Self-Refine" / Shinn et al. 2023 "Reflexion" — for S1.
- Yao et al. 2023 "Tree of Thoughts" — for S3.
- Gollwitzer 1999 "Implementation intentions: Strong effects of simple plans" — for M9 (already cited).
- Wei et al. 2022 CoT (cut from methods; still cited in Related Work).
- Andric 2025 / Mind the Gap 2026 / Alignment Revisited 2025 — already cited as prior say-do-gap-in-LLM work.

## 10. Hash and commit discipline

- Pre-registration committed before any v3 LLM call.
- Hash (sha256 of this file at commit time, pre-update): `47a938b1383c2ef9ac1b092133de99e4cfda92bca4dce14b166e94b26bee0103`. Committed before any Phase 34 LLM call.
- Any change to pre-registration after commit is logged as a **deviation** in `decisions_log.md` with timestamp + reason.
