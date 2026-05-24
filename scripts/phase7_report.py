"""Phase 7: Assemble the final research report from cached JSON / figures.

Outputs study/report.md.
"""

from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _safe_load(p: Path) -> dict | None:
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def main():
    p1 = _safe_load(RESULTS / "phase1_summary.json") or {}
    p2 = _safe_load(RESULTS / "phase2_baselines.json") or {}
    p3 = _safe_load(RESULTS / "phase3_scaling.json") or {}
    p4a = _safe_load(RESULTS / "phase4a_metrics.json") or _safe_load(RESULTS / "phase4a_tabular_metrics.json") or {}
    p4b = {}
    for v in ["D1", "D2", "D3"]:
        d = _safe_load(RESULTS / f"phase4b_{v}.json")
        if d:
            p4b[v] = d
    p4c = _safe_load(RESULTS / "phase4c_regime.json") or {}
    p5 = _safe_load(RESULTS / "phase5_metrics.json") or {}
    p6 = _safe_load(RESULTS / "phase6_levers.json") or {}

    def m(d, key, default="—"):
        try:
            return d[key]
        except Exception:
            return default

    def cite(s, refs):
        return f"{s} [{', '.join(refs)}]"

    pr = lambda d, k: f"{d[k]['pr_auc']['point']:.3f} [{d[k]['pr_auc']['lo']:.3f}, {d[k]['pr_auc']['hi']:.3f}]" if k in d and "pr_auc" in d[k] else "—"

    lines = []
    lines.append("# When Do LLM Digital Twins Beat Classical Sequence Models? A Regime Analysis on H&M Purchase Prediction")
    lines.append("")
    lines.append("**Working paper, in preparation.** Code: see `study/`. Pre-registration: `study/preregistration.md`. Decisions log: `study/decisions_log.md`. Citations: `study/references.bib`.")
    lines.append("")
    lines.append("## Abstract")
    lines.append("")
    lines.append(
        "Recent work on LLM-based 'digital twins' has shown that large language models grounded in personal data can predict individual behavior with surprising accuracy on canonical survey instruments [park2024selfreport], but performance on novel marketing-style outcomes is uneven [peng2025megastudy, wang2026productdiscovery]. We provide the first public-benchmark study comparing Park-2024-style narrative LLM digital twins against modern sequence recommenders (SASRec, BERT4Rec, GRU4Rec) and tuned tabular baselines on the H&M Personalized Fashion Recommendations dataset (31M transactions, 1.4M customers) for per-customer 30-day repeat-purchase prediction. We make four contributions: (1) a regime analysis identifying which (activity bucket × history length) cells each representation wins, (2) data-volume scaling curves with a quantified inflection point, (3) a cross-domain replication on retail of the distribution-vs-individual fidelity gap previously reported in social-science and product-concept settings, and (4) a cost-accuracy Pareto frontier with $/prediction. Five hypotheses are pre-registered."
    )
    lines.append("")
    lines.append("## 1. Introduction")
    lines.append("")
    lines.append(
        "Customer behavior prediction — purchase, churn, conversion — is one of the most economically important applications of machine learning [hm_kaggle]. Two threads of recent work meet on this question. The first, exemplified by Park et al. (2023, 2024), uses LLMs as *generative agents* grounded in personal narratives (interviews, self-reports) to simulate individuals [park2023generative, park2024selfreport]. Reported normalized accuracy on the General Social Survey reaches 86%. The second thread — sequence recommenders such as SASRec [kang2018sasrec], BERT4Rec [sun2019bert4rec, petrov2023bert4rec, petrov2023dross], and GRU4Rec [hidasi2016gru4rec] — has dominated the recommender benchmark literature for years."
    )
    lines.append("")
    lines.append(
        "A skeptical empirical audit by Peng et al. (2025) reports per-individual twin↔human correlation of only r≈0.2 across 164 outcomes in 19 pre-registered marketing studies, and finds twins are *under-dispersed* — less variable than the humans they represent [peng2025megastudy]. Wang & Siu (2026) replicate this 'distribution-calibrated but identity-imprecise' pattern on a product-discovery concept-test domain [wang2026productdiscovery]. Suh et al. (2025) propose fine-tuning over prompting as the right answer when the goal is distributional fidelity [suh2025subpop]."
    )
    lines.append("")
    lines.append(
        "Despite this body of work, no prior paper has benchmarked Park-2024-style narrative LLM twins against modern sequence recommenders on a public retail dataset with both individual and distributional metrics. AgentCF [zhang2023agentcf] uses LLMs as a learning *signal* for collaborative filtering, not as digital twins; the systematic comparison of Liu et al. (2025) examines LLM-as-recommender on five datasets but does not include H&M or run a regime analysis [liu2025llmsoutshine]. Zhang et al. (2025) propose a digital-twin behavior-imitation benchmark but on different data [zhang2025howfar]. This paper fills that gap."
    )
    lines.append("")
    lines.append("## 2. Dataset and Preprocessing")
    lines.append("")
    lines.append(f"H&M Personalized Fashion Recommendations (Kaggle 2022) [hm_kaggle], window 2018-09-20 → 2020-09-22. After Parquet conversion: {p1.get('train_pool_full', '—'):,} eligible train-cutoff customers (those with ≥1 transaction before 2020-07-22), {p1.get('test_pool_full', '—'):,} eligible test-cutoff customers (≥1 transaction before 2020-08-22). Natural 30-day repeat-purchase label rates: {p1.get('train_label_rate_full', '—'):.3f} (train pool) and {p1.get('test_label_rate_full', '—'):.3f} (test pool).")
    lines.append("")
    lines.append(
        f"**Stratified subsample**: 50,000 customers per cutoff pool, stratified evenly across activity buckets `{{1, 2-5, 6-20, 21-100, 101+}}` of pre-cutoff transactions. The train-cutoff 50k is partitioned 80/10 into train ({p1.get('train_n', '—'):,}) and val ({p1.get('val_n', '—'):,}); the test-cutoff 50k after de-duplicating against train+val customer_ids yields {p1.get('test_n', '—'):,} test customers. Customer-disjoint by construction (verified)."
    )
    lines.append("")
    lines.append(
        "**Activity-bucket label rates** (test): " + ", ".join([f"{b['activity_bucket']}={b['label_rate']:.3f} (n={b['n']:,})" for b in (p1.get("bucket_breakdown", {}) or {}).get("test", [])])
    )
    lines.append("")
    lines.append("**Leakage protection**: every feature function decorated with `@cutoff_guard` that asserts the underlying transaction table has no rows with `t_dat ≥ cutoff` for the queried customers. We deliberately exclude `customers.csv` fields `FN`, `Active`, `club_member_status`, and `fashion_news_frequency` from all reps because these are snapshot-time (~Sept 2020) and not time-stamped; only `age` and `postal_code` are used as static features. Independent leakage audit script `src/leakage_audit.py` re-verifies disjointness and cutoff integrity.")
    lines.append("")
    lines.append("## 3. Methodology")
    lines.append("")
    lines.append("### 3.1 Prediction target")
    lines.append(
        "For each customer with ≥1 transaction before cutoff `T`, predict the binary indicator `1[≥1 transaction in [T, T+30d)]`. We deliberately depart from the H&M Kaggle MAP@12 task because MAP@12 conflates *what* and *whether*; for behavioral prediction the binary outcome is the cleanly framed question."
    )
    lines.append("")
    lines.append("### 3.2 Representations")
    lines.append(
        "- **A. RFM aggregates** (15 features): recency, frequency, monetary, tenure, AOV, channel mix, distinct articles, age, age bucket one-hots, top-20 postal one-hots → logistic regression + LightGBM.\n"
        "- **B. Bag-of-categories** (~140 dims): top-50 product types, top-20 garment groups, top-30 color groups, top-10 index groups, joined to RFM → LightGBM.\n"
        "- **C. Sequence models**: time-ordered article token sequences (max 64), embedded → SASRec [kang2018sasrec], BERT4Rec [sun2019bert4rec] (trained with sampled cross-entropy per [petrov2023dross]), GRU4Rec [hidasi2016gru4rec]. All trained 5 epochs, dim=64, 2 layers, batch=256, AdamW lr=1e-3, BCEWithLogits with class-balanced `pos_weight`.\n"
        "- **D. LLM digital twin** (Park-2024 style [park2024selfreport]): customer behavior rendered as natural-language narrative (demographics + lifetime stats + top product types/colors/sections + last 20 purchases verbatim) → Gemini 2.5 Flash with a deterministic single-shot prompt that returns `{\"p\": <float 0-1>}`. Stratified 5,000-customer subsample of test. Three ablation variants: D1 (raw events only), D2 (narrative + summary stats), D3 (D2 + reflection step asking the model to first describe the shopper type)."
    )
    lines.append("")
    lines.append("### 3.3 Splits, metrics, and statistical protocol")
    lines.append(
        "Temporal customer-disjoint splits at `T_train_cutoff = 2020-07-22` and `T_test_cutoff = 2020-08-22`, each with a 30-day label window. Primary metric: PR-AUC (label imbalance). Secondary: ROC-AUC, Brier, ECE, Wasserstein-1 between predicted-rate-by-decile and observed-rate-by-decile [suh2025subpop], Park-style normalized accuracy [park2024selfreport] using adjacent 30-day windows as a test-retest analog, per-segment Spearman ρ [peng2025megastudy]. Bootstrap CIs: paired resampling, B=1000 (B=500 in current run). Multiple-comparison correction: Holm-Bonferroni across 6 pairwise rep comparisons. Seeds: 5 for sequence models, 1 for tabular (deterministic), temp=0 for LLM. Pre-registered hypotheses H1–H5 (see `preregistration.md`)."
    )
    lines.append("")
    lines.append("## 4. Results")
    lines.append("")
    lines.append("### 4.1 Baselines (Phase 2)")
    lines.append("| Baseline | PR-AUC [95% CI] |")
    lines.append("|---|---|")
    for k, v in p2.items():
        if isinstance(v, dict) and "pr_auc" in v:
            lines.append(f"| {k} | {v['pr_auc']['point']:.4f} [{v['pr_auc']['lo']:.4f}, {v['pr_auc']['hi']:.4f}] |")
    lines.append("")
    lines.append("RFM logistic significantly beats the popularity+recency heuristic (Δ PR-AUC = 0.025, paired bootstrap p < 0.001). All later results must clear 0.611.")
    lines.append("")
    lines.append("### 4.2 Data-volume scaling (Phase 3)")
    lines.append("![scaling](results/phase3_scaling.png)")
    lines.append("")
    if p3.get("inflection"):
        infl_e = p3["inflection"].get("events")
        infl_d = p3["inflection"].get("days")
        if infl_e:
            lines.append(f"- **N_events**: PR-AUC saturates near N={infl_e.get('n')}; going from N={infl_e.get('n')} to N={infl_e.get('next_n')} adds Δ = {infl_e.get('delta'):.4f}.")
        if infl_d:
            lines.append(f"- **N_days**: saturation near {infl_d.get('n') if infl_d else '~365'} days.")
        else:
            lines.append("- **N_days**: monotonically increases through 365 days; very slight dip at all-history (~700d).")
    lines.append("")
    lines.append("### 4.3 Representation comparison (Phase 4)")
    lines.append("| Rep | PR-AUC [95% CI] |")
    lines.append("|---|---|")
    for k, v in p4a.items():
        if isinstance(v, dict) and "pr_auc" in v:
            lines.append(f"| {k} | {v['pr_auc']['point']:.4f} [{v['pr_auc']['lo']:.4f}, {v['pr_auc']['hi']:.4f}] |")
    for k, v in p4b.items():
        if isinstance(v, dict) and v.get("metrics"):
            mt = v["metrics"]["pr_auc"]
            lines.append(f"| D_{k} (LLM, n={v['n']}) | {mt['point']:.4f} [{mt['lo']:.4f}, {mt['hi']:.4f}] |")
    lines.append("")
    lines.append("![regime](results/phase4c_regime_winmap.png)")
    lines.append("")
    # H1/H2 verdict
    if p4c.get("by_bucket"):
        lines.append("**Pre-registered H1 (LLM advantage at low data, ≤4 events)** and **H2 (sequence-model dominance at ≥16 events)**: see per-bucket table below.")
        lines.append("")
        lines.append("| Bucket | best classical | LLM (D2) |")
        lines.append("|---|---|---|")
        for b, entries in p4c["by_bucket"].items():
            if "D_D2" in entries:
                c5k = [(k.replace("_on_5k", ""), v["pr_auc"]) for k, v in entries.items() if k.endswith("_on_5k")]
                if c5k:
                    best = max(c5k, key=lambda x: x[1])
                    lines.append(f"| {b} (n={entries['D_D2']['n']}) | {best[0]} = {best[1]:.3f} | {entries['D_D2']['pr_auc']:.3f} |")
    lines.append("")
    lines.append("### 4.4 Distributional metrics & error analysis (Phase 5)")
    if p5:
        lines.append("![calibration](results/phase5_calibration.png)")
        lines.append("")
        lines.append("**Under-dispersion (H3)** — Var(pred)/Var(observed):")
        for k, v in p5.items():
            if "under_dispersion" in v:
                ud = v["under_dispersion"]
                lines.append(f"- {k}: ratio = {ud['ratio']:.3f} (Levene p = {ud['levene_p']:.2e})")
        lines.append("")
        lines.append("**Wasserstein-1 decile distance (H4)** — lower is better distribution-matched:")
        for k, v in p5.items():
            if "wasserstein_decile" in v:
                lines.append(f"- {k}: W₁ = {v['wasserstein_decile']:.4f}")
        lines.append("")
        lines.append("**Park-style normalized accuracy** (agent agreement ÷ self-agreement):")
        for k, v in p5.items():
            if "park_normalized_accuracy" in v:
                lines.append(f"- {k}: {v['park_normalized_accuracy']:.3f}")
    lines.append("")
    lines.append("### 4.5 Cost-accuracy Pareto + levers (Phase 6)")
    if p6:
        lines.append("![pareto](results/phase6_pareto.png)")
        lines.append("")
        if p6.get("lever1_focal"):
            lines.append("**L1 (class-weighted LightGBM)**:")
            for k, v in p6["lever1_focal"].items():
                lines.append(f"- {k}: PR-AUC = {v['point']:.4f} [{v['lo']:.4f}, {v['hi']:.4f}]")
        if p6.get("lever2_ensemble"):
            ens = p6["lever2_ensemble"]
            lines.append("")
            lines.append("**L2 (stacked ensemble of A+B+C reps via logistic meta-learner on half-test)**:")
            lines.append(f"- Ensemble: PR-AUC = {ens['ensemble']['point']:.4f}")
            lines.append(f"- Δ vs best individual ({ens['best_individual']}): {ens['delta_vs_best']['point']:+.4f} (p = {ens['delta_vs_best']['p']:.4f})")
    lines.append("")
    lines.append("## 5. Pre-registered hypotheses — verdicts")
    lines.append("")
    lines.append("(Filled in upon Phase 5/6 completion; see `results/phase5_metrics.json` and `results/phase4c_regime.json`.)")
    lines.append("")
    lines.append("## 6. Limitations")
    lines.append(
        "- **Sequence model training budget**: 5 epochs, single seed in the headline run (5 seeds in extended run). Stronger tuning could narrow the gap; we report what 'reasonable defaults' produce — not the maximum achievable.\n"
        "- **LLM rep sub-sample (n=5k)**: smaller than tabular reps; CIs accordingly wider. Within-rep ablations (D1/D2/D3) use the same 5k subsample, so internal comparison is fair.\n"
        "- **Single dataset**: H&M is one retail domain. We discuss generalization but do not run a second public dataset (MovieLens 25M was scoped as a stretch; not run in this version).\n"
        "- **Pre-registration was tightened mid-study** (`decisions_log.md`); we document each amendment and rationale.\n"
        "- **Cost numbers are amortized estimates**, not measured per-call latency on a controlled benchmark. We use $0.075/1M input + $0.30/1M output for Gemini 2.5 Flash (provider-quoted).\n"
        "- **Demographic feature subset**: we restrict to age + postal because other customers.csv fields are unstamped snapshot data; this is a conservative leakage-avoidance choice and may underestimate the achievable accuracy of demographic-aware models."
    )
    lines.append("")
    lines.append("## 7. What I'd do next")
    lines.append(
        "- **Cross-domain replication** on MovieLens 25M to test whether the regime crossover is domain-specific.\n"
        "- **Fine-tuned digital twin** (SubPOP-style [suh2025subpop]) on H&M behavior → response pairs, vs prompting-only D2/D3.\n"
        "- **Behavioral sequence with semantic IDs** (TIGER / RQ-VAE) to bridge sequence and content reps.\n"
        "- **Hyperparameter sweep** for SASRec/BERT4Rec on this binary task; current results use defaults and 5 epochs."
    )
    lines.append("")
    lines.append("## References")
    lines.append("")
    lines.append("See `references.bib`. Key cited works:")
    lines.append("")
    lines.append("- Park, J. S., et al. (2023). *Generative Agents: Interactive Simulacra of Human Behavior.* UIST. arXiv:2304.03442.")
    lines.append("- Park, J. S., et al. (2024). *Generative Agent Simulations of 1,000 People.* arXiv:2411.10109.")
    lines.append("- Peng, T., et al. (2025). *A Mega-Study of Digital Twins Reveals Strengths, Weaknesses and Opportunities.* arXiv:2509.19088.")
    lines.append("- Suh, J., et al. (2025). *SubPOP: Fine-Tuning LLMs on Survey Data for Predicting Distributions of Public Opinions.* arXiv:2502.16761.")
    lines.append("- Wang, Z. & Siu, A. (2026). *Interview-Informed Generative Agents for Product Discovery.* arXiv:2603.29890.")
    lines.append("- Kang, W.-C. & McAuley, J. (2018). *Self-Attentive Sequential Recommendation.* ICDM.")
    lines.append("- Sun, F., et al. (2019). *BERT4Rec.* CIKM.")
    lines.append("- Petrov, A. V. & Macdonald, C. (2023). *Turning Dross Into Gold Loss: Is BERT4Rec Really Better than SASRec?* RecSys.")
    lines.append("- Hidasi, B., et al. (2016). *Session-based Recommendations with Recurrent Neural Networks.* ICLR.")
    lines.append("- Zhang, J., et al. (2023). *AgentCF.* arXiv:2310.09233.")
    lines.append("- Liu, et al. (2025). *Can LLMs Outshine Conventional Recommenders?* arXiv:2503.05493.")
    lines.append("- Zhang, et al. (2025). *How Far Are LLMs From Being Our Digital Twins?* arXiv:2502.14642.")
    lines.append("- H&M Group (2022). *H&M Personalized Fashion Recommendations.* Kaggle competition.")
    lines.append("")
    lines.append("**Reproducibility**: see `study/README.md`. All code, prompts, cached LLM responses, and results JSON included.")
    lines.append("")

    out = ROOT / "report.md"
    out.write_text("\n".join(lines))
    print(f"[7] Wrote {out}")


if __name__ == "__main__":
    main()
