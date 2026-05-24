"""Phase 11: say-do gap analysis.

Computes:
- Per-arm signed gap with test-distribution reweighting + bootstrap CI
- Per-bucket gap (R2)
- H7 paired test (F-nobase vs D2 on the core-1000 subset)
- Base-rate-leakage decomposition: gap(F-base) - gap(F-nobase) vs gap(F-nobase) - gap(D2)
- Calibration curves (5 bins for n=400, 10 bins for n>=1000)

H9 (verbatim coherence) and counterfactual perturbation are separate scripts.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, spearmanr

from src import SEED
from src.eval import all_metrics, pr_auc, bootstrap_ci, paired_bootstrap_diff

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

BUCKETS_ORDER = ["1", "2-5", "6-20", "21-100", "101+"]
# Natural test-distribution bucket proportions (from phase1_summary.json).
TEST_DIST = {"1": 9302, "2-5": 9754, "6-20": 9817, "21-100": 9717, "101+": 8275}
TOTAL = sum(TEST_DIST.values())
TEST_PROBS = {k: v / TOTAL for k, v in TEST_DIST.items()}


def reweight_to_test_distribution(scores: np.ndarray, actual: np.ndarray, buckets: np.ndarray) -> dict:
    """Return per-bucket-weighted mean stated, mean actual, and signed gap."""
    bw_stated, bw_actual = 0.0, 0.0
    counted = 0.0
    per_bucket = {}
    for b in BUCKETS_ORDER:
        mask = buckets == b
        if mask.sum() == 0:
            continue
        w = TEST_PROBS[b]
        bs = float(scores[mask].mean())
        ba = float(actual[mask].mean())
        bw_stated += w * bs
        bw_actual += w * ba
        counted += w
        per_bucket[b] = {"n": int(mask.sum()), "mean_stated": bs, "mean_actual": ba, "gap": bs - ba}
    # If some buckets are missing in this arm, renormalise
    if counted > 0:
        bw_stated /= counted
        bw_actual /= counted
    return {"reweighted_mean_stated": bw_stated, "reweighted_mean_actual": bw_actual,
            "reweighted_signed_gap": bw_stated - bw_actual, "per_bucket": per_bucket}


def bootstrap_gap_ci(scores: np.ndarray, actual: np.ndarray, buckets: np.ndarray,
                     B: int = 1000, seed: int = SEED) -> dict:
    """Stratified bootstrap CI on the test-reweighted signed gap.
    Audit-fix: resample WITHIN each bucket to original bucket size, then reweight.
    """
    rng = np.random.default_rng(seed)
    bucket_to_idx = {b: np.where(buckets == b)[0] for b in BUCKETS_ORDER if (buckets == b).any()}
    boots = []
    for _ in range(B):
        sampled = []
        for b, idx in bucket_to_idx.items():
            samp = rng.choice(idx, size=len(idx), replace=True)
            sampled.append(samp)
        idx = np.concatenate(sampled)
        rw = reweight_to_test_distribution(scores[idx], actual[idx], buckets[idx])
        boots.append(rw["reweighted_signed_gap"])
    boots = np.array(boots)
    point = reweight_to_test_distribution(scores, actual, buckets)["reweighted_signed_gap"]
    return {"point": float(point), "lo": float(np.quantile(boots, 0.025)),
            "hi": float(np.quantile(boots, 0.975)), "se": float(boots.std()), "B": B,
            "method": "stratified_bootstrap_within_bucket"}


def calibration_curve(scores: np.ndarray, actual: np.ndarray, n_bins: int):
    edges = np.quantile(scores, np.linspace(0, 1, n_bins + 1))
    edges[0] = 0.0
    edges[-1] = 1.0001
    bin_idx = np.digitize(scores, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = bin_idx == b
        if m.sum() == 0:
            continue
        out.append({"bin": int(b), "n": int(m.sum()),
                    "mean_pred": float(scores[m].mean()), "actual_rate": float(actual[m].mean())})
    return out


def load_arm(name: str):
    """Load an F-* arm. CANONICAL stated_intent_prob is the RAW (pre-guardrail)
    LLM output, per preregistration_v2.md §"Canonical stated_intent_prob".
    Audit-fix: previously loaded the post-guardrail `stated_intent` which
    mean-shrinks toward truth and would give F-arms an unearned gap reduction.
    """
    p = RESULTS / f"phase10_{name}_scores.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    raw = d["stated_intent_raw"].astype(float) if "stated_intent_raw" in d.files else d["stated_intent"].astype(float)
    return {
        "name": name, "cids": list(d["customer_id"]),
        "scores": raw,
        "scores_guardrailed": d["stated_intent"].astype(float),  # kept for sensitivity check
        "actual": d["actual"].astype(int),
        "buckets": d["activity_bucket"].astype(str),
        "n_tx": d["n_tx_pre_cutoff"].astype(int),
        "verbatim": list(d["verbatim"]),
        "reasoning": list(d["reasoning"]),
        "key_objection": list(d["key_objection"]),
    }


def load_d2_for_core(core_cids: list[str]):
    d = np.load(RESULTS / "phase4b_D2_scores.npz", allow_pickle=True)
    d_cids = list(d["customer_id"])
    d_scores = d["scores"].astype(float)
    d_actual = d["y_test"].astype(int)
    d_buckets = d["activity_bucket"].astype(str)
    lookup = {c: i for i, c in enumerate(d_cids)}
    matched_idx = [lookup[c] for c in core_cids if c in lookup]
    matched_core_cids = [c for c in core_cids if c in lookup]
    return {
        "name": "D2_on_core",
        "cids": matched_core_cids,
        "scores": d_scores[matched_idx],
        "actual": d_actual[matched_idx],
        "buckets": d_buckets[matched_idx],
        "overlap_n": len(matched_idx),
    }


def main():
    f_base = load_arm("F-base")
    f_nobase = load_arm("F-nobase")
    if f_base is None or f_nobase is None:
        raise SystemExit("[11] Need both F-base and F-nobase scores. Run Phase 10 first.")

    d2_core = load_d2_for_core(f_nobase["cids"])
    print(f"[11] D2-on-core overlap: {d2_core['overlap_n']}/{len(f_nobase['cids'])}")

    out: dict = {"arms": {}}

    for arm in [f_base, f_nobase, d2_core]:
        gap_ci = bootstrap_gap_ci(arm["scores"], arm["actual"], arm["buckets"], B=1000, seed=SEED)
        rw = reweight_to_test_distribution(arm["scores"], arm["actual"], arm["buckets"])
        cal = calibration_curve(arm["scores"], arm["actual"], n_bins=10)
        m = all_metrics(arm["actual"], arm["scores"], B=500, seed=SEED)
        out["arms"][arm["name"]] = {
            "n": len(arm["scores"]),
            "raw_mean_stated": float(arm["scores"].mean()),
            "raw_mean_actual": float(arm["actual"].mean()),
            "raw_signed_gap": float(arm["scores"].mean() - arm["actual"].mean()),
            "reweighted_mean_stated": rw["reweighted_mean_stated"],
            "reweighted_mean_actual": rw["reweighted_mean_actual"],
            "reweighted_signed_gap": rw["reweighted_signed_gap"],
            "reweighted_signed_gap_CI": gap_ci,
            "per_bucket_gap": rw["per_bucket"],
            "pr_auc": m["pr_auc"], "brier": m["brier"], "ece": m["ece"],
            "wasserstein_decile": m["wasserstein_decile"],
            "under_dispersion": m["under_dispersion"],
            "calibration_curve": cal,
        }

    # ---- H7 paired test: F-nobase vs D2_on_core (same customers) ----
    # Pair customers by cid; compare |stated - actual| per customer.
    f_nb_lookup = {c: i for i, c in enumerate(f_nobase["cids"])}
    d2_lookup = {c: i for i, c in enumerate(d2_core["cids"])}
    paired = [c for c in f_nobase["cids"] if c in d2_lookup]
    f_nb_err = np.array([abs(f_nobase["scores"][f_nb_lookup[c]] - f_nobase["actual"][f_nb_lookup[c]]) for c in paired])
    d2_err = np.array([abs(d2_core["scores"][d2_lookup[c]] - d2_core["actual"][d2_lookup[c]]) for c in paired])
    diffs = f_nb_err - d2_err  # negative = F-nobase has SMALLER gap (better)
    w_stat, w_p = wilcoxon(diffs, zero_method="wilcox", alternative="less")
    # Audit-fix: the prereg also requires |gap(F-nb)| <= |gap(D2)| - 0.05 margin.
    paired_fnb_scores = np.array([f_nobase["scores"][f_nb_lookup[c]] for c in paired])
    paired_fnb_actual = np.array([f_nobase["actual"][f_nb_lookup[c]] for c in paired])
    paired_d2_scores = np.array([d2_core["scores"][d2_lookup[c]] for c in paired])
    paired_d2_actual = np.array([d2_core["actual"][d2_lookup[c]] for c in paired])
    gap_fnb_paired = abs(paired_fnb_scores.mean() - paired_fnb_actual.mean())
    gap_d2_paired = abs(paired_d2_scores.mean() - paired_d2_actual.mean())
    margin = gap_d2_paired - gap_fnb_paired   # positive = F-nobase has smaller |gap|
    out["H7"] = {
        "paired_n": len(paired),
        "mean_abs_err_F_nobase": float(f_nb_err.mean()),
        "mean_abs_err_D2": float(d2_err.mean()),
        "mean_diff_F_nb_minus_D2": float(diffs.mean()),
        "abs_gap_F_nobase_paired": float(gap_fnb_paired),
        "abs_gap_D2_paired": float(gap_d2_paired),
        "abs_gap_margin_D2_minus_Fnb": float(margin),
        "wilcoxon_alt_less_p": float(w_p),
        "bonferroni_threshold": 0.025,
        "prereg_margin_threshold": 0.05,
        "verdict": (
            "CONFIRMED" if (w_p < 0.025 and margin >= 0.05) else
            ("PARTIAL: wilcoxon-significant but |gap| margin < 0.05" if (w_p < 0.025 and margin > 0) else
             "REFUTED_or_NS")
        ),
    }

    # ---- Base-rate-leakage decomposition (audit Addition 1) ----
    # Compare F-base vs F-nobase on the same customers; compare F-nobase vs D2 on overlap.
    fb_lookup = {c: i for i, c in enumerate(f_base["cids"])}
    paired_fb_fnb = [c for c in f_base["cids"] if c in f_nb_lookup]
    fb_gap = np.array([f_base["scores"][fb_lookup[c]] for c in paired_fb_fnb]).mean() - \
             np.array([f_base["actual"][fb_lookup[c]] for c in paired_fb_fnb]).mean()
    fnb_gap = np.array([f_nobase["scores"][f_nb_lookup[c]] for c in paired_fb_fnb]).mean() - \
              np.array([f_nobase["actual"][f_nb_lookup[c]] for c in paired_fb_fnb]).mean()
    d2_gap = (d2_core["scores"].mean() - d2_core["actual"].mean())
    out["base_rate_leakage_decomp"] = {
        "paired_fb_fnb_n": len(paired_fb_fnb),
        "gap_F_base": float(fb_gap),
        "gap_F_nobase": float(fnb_gap),
        "gap_D2_on_core": float(d2_gap),
        "delta_base_minus_nobase": float(fb_gap - fnb_gap),
        "delta_nobase_minus_D2": float(fnb_gap - d2_gap),
        "leakage_dominates": abs(fb_gap - fnb_gap) > abs(fnb_gap - d2_gap),
    }

    # ---- R1, R2 replication: report effect sizes ----
    out["R1_intent_inflation"] = {arm["name"]: float(arm["scores"].mean() - arm["actual"].mean())
                                   for arm in [f_base, f_nobase, d2_core]}
    out["R2_heterogeneous_gap"] = {}
    for arm in [f_base, f_nobase, d2_core]:
        rw = reweight_to_test_distribution(arm["scores"], arm["actual"], arm["buckets"])
        out["R2_heterogeneous_gap"][arm["name"]] = {b: pb["gap"] for b, pb in rw["per_bucket"].items()}

    # Save
    (RESULTS / "phase11_gap.json").write_text(json.dumps(out, indent=2, default=str))

    # Figure 1: per-bucket signed gap, three arms.
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.27
    x = np.arange(len(BUCKETS_ORDER))
    arms_for_plot = [("F-base", f_base), ("F-nobase", f_nobase), ("D2 (flat)", d2_core)]
    for i, (label, arm) in enumerate(arms_for_plot):
        rw = reweight_to_test_distribution(arm["scores"], arm["actual"], arm["buckets"])
        gaps = [rw["per_bucket"].get(b, {}).get("gap", float("nan")) for b in BUCKETS_ORDER]
        ax.bar(x + (i - 1) * width, gaps, width, label=label)
    ax.axhline(0, color="k", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(BUCKETS_ORDER)
    ax.set_xlabel("Activity bucket (pre-cutoff tx count)")
    ax.set_ylabel("Signed gap (E[stated] − E[actual])")
    ax.set_title("Say-do gap by regime — F-base vs F-nobase vs D2 flat")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(RESULTS / "phase11_gap_by_bucket.png", dpi=130)
    plt.close()

    # Figure 2: calibration curves overlay.
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, arm in arms_for_plot:
        cal = calibration_curve(arm["scores"], arm["actual"], n_bins=10)
        if not cal:
            continue
        xs = [c["mean_pred"] for c in cal]
        ys = [c["actual_rate"] for c in cal]
        ax.plot(xs, ys, "o-", label=label, alpha=0.85)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("Mean stated intent (binned)")
    ax.set_ylabel("Actual rate")
    ax.set_title("Calibration: stated intent vs revealed behavior")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "phase11_calibration.png", dpi=130)
    plt.close()

    print("\n[11] Headline results:")
    for arm_name, st in out["arms"].items():
        ci = st["reweighted_signed_gap_CI"]
        print(f"  {arm_name:<12} n={st['n']:<5} gap = {st['reweighted_signed_gap']:+.4f} [{ci['lo']:+.4f}, {ci['hi']:+.4f}], PR-AUC = {st['pr_auc']['point']:.4f}")

    print("\n[11] H7 (F-nobase < D2 in |stated - actual|, paired Wilcoxon, alpha=0.025):")
    print(f"   mean |err|: F-nobase={out['H7']['mean_abs_err_F_nobase']:.4f}, D2={out['H7']['mean_abs_err_D2']:.4f}, "
          f"diff={out['H7']['mean_diff_F_nb_minus_D2']:+.4f}, p={out['H7']['wilcoxon_alt_less_p']:.4g} → {out['H7']['verdict']}")

    print("\n[11] Base-rate-leakage decomposition:")
    bd = out["base_rate_leakage_decomp"]
    print(f"  gap(F-base)   = {bd['gap_F_base']:+.4f}")
    print(f"  gap(F-nobase) = {bd['gap_F_nobase']:+.4f}")
    print(f"  gap(D2_core)  = {bd['gap_D2_on_core']:+.4f}")
    print(f"  Δ base→nobase = {bd['delta_base_minus_nobase']:+.4f}   (leakage contribution)")
    print(f"  Δ nobase→D2   = {bd['delta_nobase_minus_D2']:+.4f}   (clean cognition contribution)")
    print(f"  LEAKAGE DOMINATES: {bd['leakage_dominates']}")

    print("\n[11] saved phase11_gap.json + 2 figures")


if __name__ == "__main__":
    main()
