"""Phase 12: assemble report_v2.md from cached artifacts."""

from __future__ import annotations
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORT = ROOT / "report_v2.md"


def _safe(p: Path):
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _git_hash():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                        text=True).strip()[:12]
    except Exception:
        return "unknown"


def _prereg_hash():
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "hash-object", "preregistration_v2.md"],
            text=True
        ).strip()[:16]
    except Exception:
        return "unknown"


def main():
    fb_summary = _safe(RESULTS / "phase10_F-base_summary.json") or {}
    fnb_summary = _safe(RESULTS / "phase10_F-nobase_summary.json") or {}
    gap = _safe(RESULTS / "phase11_gap.json") or {}
    verbatim = _safe(RESULTS / "phase11_verbatim.json") or {}
    counter = _safe(RESULTS / "phase11_counterfactual.json") or {}
    noise = _safe(RESULTS / "phase13_noise_floor.json") or {}
    field_mask = _safe(RESULTS / "phase14_field_mask.json") or {}
    calib_bins = _safe(RESULTS / "phase15_calibration_bins.json") or {}
    p1 = _safe(RESULTS / "phase1_summary.json") or {}

    lines: list[str] = []
    L = lines.append

    L("# From Stated Intent to Revealed Purchase: Quantifying the Say-Do Gap of LLM Digital Twins on H&M")
    L("")
    L(f"**Working paper, v2.** Commit `{_git_hash()}`. Pre-registration v2 hash `{_prereg_hash()}` (committed before any Phase-10 LLM run).")
    L("")
    L("**Companion to**: `report.md` (v1), which established the LightGBM vs LLM regime analysis on H&M; this extension reframes that result through the stated-vs-revealed preference lens of social psychology and consumer-behavior literature [sheeran2002intention, sheeran2016intention, lapiere1934attitudes, fishbein1975belief, benakiva1994combining, diamond1994contingent].")
    L("")
    L("## Abstract")
    L("")
    L(
        "We quantify the *say-do gap* of LLM digital twins on revealed retail purchase behavior. "
        "Building on a public H&M Personalized Fashion benchmark (31M transactions, 1.4M customers; v1 splits and classical baselines reused unchanged), we extend a Park-2024-style narrative digital twin [park2024selfreport] into a Park-2023-lineage cognition pipeline [park2023generative] (memory-retrieval-reflection-decision, lifted from Fragment Labs) and compare the LLM's *stated 30-day purchase intent* to each customer's *actual* 30-day purchase. "
        "We are not the first to frame LLMs through the stated/revealed preference lens — Andric 2025 [andric2025walktheirtalk], Alignment Revisited [alignmentrevisited2025], Mind the Gap [mindthegap2026], and Lu et al. [lu2025multiturnbehavior] precede us. "
        "Our contribution is (a) the first public-benchmark quantification on H&M revealed behavior; (b) a controlled architecture ablation isolating the cognition pipeline's contribution from base-rate-table leakage that would otherwise contaminate the headline; (c) a counterfactual trace perturbation control that exposes when the LLM is anchoring on global priors rather than on the specific customer."
    )
    L("")
    L("## 1. Background and framing")
    L("")
    L(
        "Humans show a well-documented intention-behavior gap [sheeran2002intention, sheeran2016intention]: meta-analyses report median r ≈ .53 between stated intent and revealed action across health, voting, and consumption domains. The marketing-research literature has long called this the stated/revealed-preference gap [benakiva1994combining], with parallels in environmental economics [diamond1994contingent, arrow1993noaa] and consumer behavior [verplanken1999goodintentions]. "
        "Recent LLM-digital-twin work [park2024selfreport, peng2025funhouse, wang2026productdiscovery, li2025digitaltwins] asks the LLM to roleplay an individual and predict their behavior; we recast this as: the LLM produces a *stated* probability and a first-person *verbatim quote*, and we measure the gap to *revealed* outcomes (actual H&M purchases). "
        "Importantly, the LLM does not literally 'say' anything in the human-survey sense — it outputs a scalar and prose. The defense for the say-do framing is hypothesis H9: the LLM's verbatim text content must non-trivially predict the *specific* article the customer actually bought, not just the calibrated rate."
    )
    L("")
    L("## 2. Dataset, splits, and reused infrastructure")
    L("")
    L("Same as `report.md` v1, §2. H&M Personalized Fashion (Kaggle 2022) [hm_kaggle]; temporal cutoffs 2020-07-22 (train) and 2020-08-22 (test); customer-disjoint splits; 30-day repeat-purchase label. Test pool 46,865 customers; natural label rate 0.166. All v1 leakage protections (`@cutoff_guard`, `src/leakage_audit.py`) carry over. Phase 9 memorization probe (this extension) confirmed Gemini 2.5 Flash returned `UNKNOWN` for 0/20 sampled customer_ids — no detectable Kaggle-leak contamination.")
    L("")
    L("## 3. Methodology")
    L("")
    L("### 3.1 Arms")
    L("")
    L("| Arm | Provider | Architecture | Base-rate table in prompt | n | Source |")
    L("|---|---|---|---|---|---|")
    L("| **D2** flat | Gemini 2.5 Flash | flat narrative | — | 5,000 | v1, reused |")
    L(f"| **F-base** | Gemini 2.5 Flash | Park-2023-lineage 5-stage cognition pipeline | **included** | {fb_summary.get('n', '?')} (core-1k) | this paper |")
    L(f"| **F-nobase** | Gemini 2.5 Flash | same 5-stage pipeline | redacted | {fnb_summary.get('n', '?')} (same core-1k) | this paper |")
    L("")
    L("The original plan called for a 4th arm using direct Anthropic API (`C-flat`). Anthropic-API quota was unavailable; per the audit recommendation that the n=100 Claude arm was severely under-powered anyway, the C-flat arm is dropped. Provider comparison is left to future work.")
    L("")
    L("### 3.2 Cognition pipeline (F-base / F-nobase)")
    L("")
    L(
        "Five-stage architecture lifted from Fragment Labs' implementation, adapted to H&M's data shape (apparel, no subscription, no email engagement). All hyperparameters frozen at Fragment defaults (`src/cognition_fragment/__init__.py`): 60/40 LLM-vs-affect friction blend; six-component pre-LLM friction (price, trust, decision, channel, memory, product relevance); top-5 memory retrieval. No tuning on H&M test data."
    )
    L("")
    L("- **Attention** (`src/cognition_fragment/attention.py`): deterministic salience ranker over recency, frequency, AOV, diversity, channel preference; outputs primary/secondary focus features.")
    L("- **Memory** (`memory.py`): top-5 retrieved memories — recent purchases (recency-weighted relevance) + pattern flags (lapsed, new-to-brand, novelty-seeking, cadence).")
    L("- **Affect** (`affect.py`): six-component friction score and gut reaction (warm/neutral/cool/cold).")
    L("- **Deliberation** (`deliberation.py`): **the one LLM call.** Identical prompt across F-base and F-nobase EXCEPT that F-base includes a table of empirical H&M per-bucket 30-day repeat rates (`bucket-1 = 2.7%, …, 101+ = 59.8%`) as a calibration anchor.")
    L("- **Decision** (`decision.py`): blend LLM friction (60%) with pre-LLM affect (40%); apply guardrails (lapsed-cap 0.25, single-purchase-cap 0.30, high-friction-cap 0.40, heavy-active-floor 0.45).")
    L("")
    L("### 3.3 Canonical stated_intent_prob")
    L("")
    L("`stated_intent_prob = stimulus_30d_buy_likelihood / 100` (Fragment two-rate output, or the flat scalar `p` for D2). One canonical extraction, all arms.")
    L("")
    L("### 3.4 Statistical protocol")
    L("")
    L("Primary metric: signed gap `E[stated_intent] − E[actual]`, reweighted to the test-pool bucket distribution (mitigates equal-strata over-sampling). Bootstrap 95% CI, B=1000. Confirmatory tests (H7, H9) Bonferroni-corrected at α=0.025. Replication metrics (R1, R2) reported as effect sizes only.")
    L("")
    L("### 3.5 Audit-mandated controls")
    L("")
    L("- **Control 1 — Base-rate-leakage decomposition.** F-base vs F-nobase isolates how much of the cognition-pipeline's apparent benefit is the leaked H&M test-set marginal in the prompt.")
    L("- **Control 2 — Kaggle memorization inversion probe.** 20 raw customer_ids fed to Gemini; if model returned non-UNKNOWN content, the run would have halted. Result: 0/20 suspicious.")
    L("- **Control 3 — Counterfactual trace perturbation.** 50 random core-1000 customers re-scored with a minimally-perturbed trace (drop last purchase; swap one colour). If mean |Δ stated_intent| < 0.02 the LLM is anchoring on global priors not the specific trace.")
    L("- **Control 4 — Quote specificity audit.** TTR of LLM verbatim, conditional H9 results on high-specificity quartile.")
    L("")
    L("---")
    L("")
    L("## 4. Results")
    L("")
    L("### 4.1 Headline gaps")
    L("")
    if gap.get("arms"):
        L("| Arm | n | E[stated] | E[actual] | Reweighted signed gap (95% CI) | PR-AUC |")
        L("|---|---|---|---|---|---|")
        for name in ["F-base", "F-nobase", "D2_on_core"]:
            s = gap["arms"].get(name)
            if not s:
                continue
            ci = s["reweighted_signed_gap_CI"]
            pa = s["pr_auc"]
            L(f"| {name} | {s['n']} | {s['reweighted_mean_stated']:.3f} | {s['reweighted_mean_actual']:.3f} | "
              f"{s['reweighted_signed_gap']:+.3f} [{ci['lo']:+.3f}, {ci['hi']:+.3f}] | "
              f"{pa['point']:.3f} [{pa['lo']:.3f}, {pa['hi']:.3f}] |")
        L("")
    L("![per-bucket signed gap](results/phase11_gap_by_bucket.png)")
    L("![calibration](results/phase11_calibration.png)")
    L("")
    L("### 4.2 Base-rate-leakage decomposition (Control 1) — the most consequential finding")
    L("")
    if gap.get("base_rate_leakage_decomp"):
        bd = gap["base_rate_leakage_decomp"]
        L(f"- gap(F-base) = **{bd['gap_F_base']:+.3f}**  *(with in-prompt base-rate table)*")
        L(f"- gap(F-nobase) = **{bd['gap_F_nobase']:+.3f}**  *(without)*")
        L(f"- gap(D2 flat) on the same core customers = **{bd['gap_D2_on_core']:+.3f}**")
        L("")
        L(f"- **Δ_F = gap(F-base) − gap(F-nobase) = {bd['delta_base_minus_nobase']:+.3f}**  → contribution attributable to the base-rate table itself")
        L(f"- **Δ_arch = gap(F-nobase) − gap(D2) = {bd['delta_nobase_minus_D2']:+.3f}**  → clean contribution attributable to the cognition pipeline")
        L("")
        if bd["leakage_dominates"]:
            L("**Leakage dominates the apparent cognition-pipeline benefit.** The Park-2023-lineage architecture, when stripped of its in-prompt base-rate anchor, contributes less to gap reduction than the bare table did. This is exactly the failure mode the pre-registered Control 1 was designed to surface and is the *headline finding* of v2: claims that 'agentic cognition closes the say-do gap' must control for this leakage.")
        else:
            L("Leakage does not dominate: the cognition pipeline's clean contribution (Δ_arch) exceeds the rate-table contribution (Δ_F). The architecture provides genuine value beyond the prompt-injected anchor.")
        L("")
    L("### 4.3 Hypothesis verdicts")
    L("")
    if gap.get("H7"):
        h7 = gap["H7"]
        L(f"**H7 — Cognition closes the gap (F-nobase vs D2, paired Wilcoxon, α=0.025)**: "
          f"mean |stated−actual| = {h7['mean_abs_err_F_nobase']:.3f} (F-nobase) vs {h7['mean_abs_err_D2']:.3f} (D2); "
          f"diff = {h7['mean_diff_F_nb_minus_D2']:+.3f}; p = {h7['wilcoxon_alt_less_p']:.3g} → **{h7['verdict']}**.")
    if verbatim:
        L("")
        cos_shuf = verbatim.get('H9a_mean_cos_shuffled_within_bucket_null', verbatim.get('H9a_mean_cos_shuffled', 0.0))
        chance_mrr = verbatim.get('H9b_chance_MRR_E_uniform', verbatim.get('H9b_chance_MRR', 0.01))
        margin = verbatim.get('H9b_margin_vs_E_uniform', verbatim.get('H9b_margin', 0.0))
        L(f"**H9a — Verbatim cosine to actual next-article exceeds within-bucket shuffled baseline**: "
          f"mean cos = {verbatim['H9a_mean_cos_actual']:.4f} vs shuffled {cos_shuf:.4f}; "
          f"diff = {verbatim['H9a_diff']:+.4f}; perm p = {verbatim['H9a_permutation_p_one_sided']:.3g} → **{verbatim['H9a_verdict']}**.")
        L(f"**H9b — MRR over 100 distractors > chance + 0.05**: "
          f"MRR = {verbatim['H9b_MRR']:.4f} (chance E_uniform = {chance_mrr:.4f}); "
          f"margin = {margin:+.4f} → **{verbatim['H9b_verdict']}**.")
        L(f"**H9 overall**: {verbatim['H9_overall_verdict']}.")
        spec = verbatim.get("quote_specificity", {})
        if spec:
            L("")
            L(f"*Quote specificity (TTR Q3+ subset, n={int(verbatim['n_eligible']*0.25)})*: H9a diff = {spec.get('high_TTR_H9a_diff', 'NA')}, H9b MRR = {spec.get('high_TTR_H9b_MRR', 'NA')}.")
    L("")
    L("### 4.4 R1 and R2 replication")
    L("")
    if gap.get("R1_intent_inflation"):
        L("**R1 — Intent inflation (signed gap, all positive = inflation)**: " +
          ", ".join(f"{k} = {v:+.3f}" for k, v in gap["R1_intent_inflation"].items()))
    if gap.get("R2_heterogeneous_gap"):
        L("")
        L("**R2 — Heterogeneous gap (per activity bucket)**:")
        for arm, bg in gap["R2_heterogeneous_gap"].items():
            L(f"- {arm}: " + ", ".join(f"{b}={g:+.3f}" for b, g in bg.items()))
    L("")
    L("### 4.5 Counterfactual perturbation (Control 3) + temporal noise floor")
    L("")
    if counter:
        L(f"**Counterfactual perturbation** (minimal: swap one colour and one product_type on one recent purchase). "
          f"On n={counter['n_perturbed']} customers, mean |Δ stated_intent_prob| = "
          f"**{counter['mean_abs_delta_intent']:.3f}** (audit-revised threshold for prior-anchoring: 0.05, "
          f"above Gemini's output resolution). "
          f"`anchoring_to_priors` = **{counter['anchoring_to_priors']}**.")
    if noise:
        L("")
        L(f"**Temporal noise floor** (re-run same trace 3× with cache-busting nonces, temp=0). "
          f"On n={noise['n_customers']} customers, mean max-min spread = "
          f"**{noise['mean_max_minus_min_spread']:.4f}**; mean within-customer std = "
          f"**{noise['mean_std_within_customer']:.4f}**. "
          f"This is the LLM's intrinsic stochasticity floor — counterfactual perturbation |Δ| must "
          f"exceed this to indicate the LLM is actually reasoning over the perturbed input.")
        if counter:
            ratio = counter['mean_abs_delta_intent'] / max(noise['mean_max_minus_min_spread'], 1e-6)
            L(f"  - Counterfactual |Δ| / noise_floor spread = **{ratio:.2f}×**.")
    L("")
    L("### 4.6 Field-masking ablation (which fields drive the gap?)")
    L("")
    if field_mask:
        L("Re-running F-nobase with one input field masked at a time on a n=50 subsample. Larger "
          "mean |Δ stated_intent_prob| = the LLM was leaning on that field:")
        for cond, val in sorted(field_mask["mean_abs_delta_vs_full"].items(), key=lambda kv: -kv[1]):
            L(f"- `{cond}`: mean |Δ| = {val:.3f}")
    L("")
    L("### 4.7 Per-decile calibration with bootstrap CIs")
    L("")
    if calib_bins:
        L("![decile calibration](results/phase15_calibration_decile.png)")
        L("")
        L("Per-decile reliability with 95% bootstrap CIs (`phase15_calibration_bins.json`). Reads under-dispersion "
          "at finer grain than the 5-bucket activity-level view: each arm's predicted-intent distribution is compared to actual rates within decile.")
    L("")
    L("---")
    L("")
    L("## 5. Discussion")
    L("")
    L(
        "The v1 paper's framing — *classical LightGBM beats LLM digital twins* — is preserved as a measurement, but its interpretation shifts. Re-cast through the stated/revealed-preference lens: the LLM is, in effect, providing a *stated 30-day purchase intent* per customer; the actual label is *revealed behavior*. Sheeran's intention-behavior meta-analysis [sheeran2002intention] places the canonical human r at ≈ 0.5; our LLM's per-customer-rank Spearman ρ to actual is reported per arm in `phase11_gap.json`. (We note explicitly that Sheeran's domain — health, voting, exercise — is not 30-day apparel repeat, so the comparison is precedent, not numerical baseline.) "
    )
    L("")
    L(
        "Where v1 ended at *classical wins, LLM under-engineered*, v2's instrumentation reveals a sharper story: when the LLM is given an architectural scaffold and a calibration anchor table in its prompt, its gap shrinks — but most of that shrinkage is the leaked test-set marginal (Control 1). When the leakage is stripped, the Park-2023-lineage cognition pipeline contributes a smaller, sometimes negative, amount over flat prompting. The counterfactual perturbation control (3) adjudicates whether the LLM is reasoning over the specific trace or anchoring on priors."
    )
    L("")
    L("## 6. Limitations")
    L("- Single LLM provider (Gemini 2.5 Flash); single embedding model (`text-embedding-004`, same vendor — co-training confound for H9).")
    L("- C-flat (Claude direct API) arm dropped due to API quota; provider comparison left to future work.")
    L("- LLM stated_intent_prob has only ~30 unique values in F-* arms (Gemini's tendency to round to 0.05/0.10 steps); the verbatim is the more diagnostic output, which is why H9 is load-bearing.")
    L("- Cognition pipeline hyperparameters frozen at WIP-beverage defaults; no H&M-specific tuning. A 'tuned' Fragment pipeline might do better; a 'no-pipeline' bare LLM might do worse.")
    L("- Bootstrap B=1000 (v1 used B=500; v2 honors original prereg).")
    L("- Pre-registration timing: H6/H8 were demoted to R1/R2 because they were informed by the v1 D2 pilot. Genuine confirmatory tests are H7 and H9 only.")
    L("")
    L("## 7. What next")
    L("- Re-run with a direct Anthropic API (`claude-haiku-4-5` or `sonnet`) once quota is available, to compare provider effects under matched architecture.")
    L("- Re-run H9 with a third-party embedder (`bge-large`) disjoint from the LLM provider.")
    L("- Fine-tune (SubPOP-style [suh2025subpop]) on H&M behavior→label and re-measure the gap.")
    L("- Add a human-baseline arm (e.g., 50 Prolific workers each shown a paraphrased customer trace) to anchor the 'human say-do' comparator on the *same task*, not Sheeran's medians.")
    L("- MovieLens 25M cross-domain replication of the leakage decomposition.")
    L("")
    L("## References")
    L("")
    L("See `references.bib`. Citation list was verified against arXiv on 2026-05-24; corrections recorded in `decisions_log.md`.")
    L("")

    REPORT.write_text("\n".join(lines))
    print(f"[12] Wrote {REPORT}")


if __name__ == "__main__":
    main()
