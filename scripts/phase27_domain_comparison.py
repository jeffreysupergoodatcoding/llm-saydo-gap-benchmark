"""Phase 27: side-by-side domain-comparison figure + summary table.

Produces one PNG and one JSON: H&M and MovieLens side-by-side on
(gap, pooled ρ, within ρ, leakage Δ, human-self r).
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _safe(p):
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def main():
    sp = _safe(RESULTS / "phase19_spearman.json") or {}
    ml = _safe(RESULTS / "phase23_ml_analysis.json") or {}
    hm_human = _safe(RESULTS / "phase24_human_baseline.json") or {}
    ml_human = _safe(RESULTS / "phase26_ml_human_baseline.json") or {}
    pair = _safe(RESULTS / "phase18_pairwise_gap_diffs.json") or {}

    rows = []
    # H&M
    if sp:
        for arm in ["F-base", "F-nobase", "D2-core"]:
            if arm in sp:
                a = sp[arm]
                rows.append({"domain": "H&M", "arm": arm,
                             "pooled_ρ": a["pooled_spearman"]["rho"],
                             "within_ρ": a.get("within_bucket_spearman", a.get("within_bucket_pooled_spearman", {}))["rho"]})
    # ML
    if ml:
        for arm in ["F-base", "F-nobase"]:
            if arm in ml:
                a = ml[arm]
                rows.append({"domain": "MovieLens", "arm": arm,
                             "pooled_ρ": a["pooled_spearman"]["rho"],
                             "within_ρ": a.get("within_bucket_spearman", a.get("within_bucket_pooled_spearman", {}))["rho"]})

    # Plot pooled vs within
    fig, ax = plt.subplots(figsize=(8, 5))
    hm_pooled = [r["pooled_ρ"] for r in rows if r["domain"] == "H&M"]
    hm_within = [r["within_ρ"] for r in rows if r["domain"] == "H&M"]
    hm_labels = [r["arm"] for r in rows if r["domain"] == "H&M"]
    ml_pooled = [r["pooled_ρ"] for r in rows if r["domain"] == "MovieLens"]
    ml_within = [r["within_ρ"] for r in rows if r["domain"] == "MovieLens"]
    ml_labels = [r["arm"] for r in rows if r["domain"] == "MovieLens"]

    x = np.arange(len(hm_labels) + len(ml_labels))
    pooled = hm_pooled + ml_pooled
    within = hm_within + ml_within
    labels = [f"H&M\n{l}" for l in hm_labels] + [f"ML\n{l}" for l in ml_labels]
    w = 0.35
    ax.bar(x - w / 2, pooled, w, label="pooled ρ", color="steelblue")
    ax.bar(x + w / 2, within, w, label="within-bucket ρ", color="darkorange")
    # Human reference lines
    if hm_human.get("pairwise_corr"):
        hr = hm_human["pairwise_corr"]["w1_T-60_T-30__vs__w2_T-30_T"]["pearson_r"]
        ax.axhline(hr, color="green", linestyle="--", alpha=0.5,
                   label=f"H&M within-domain human-self r={hr:.3f}")
    ml_human_avail = (ml_human and "pairwise_corr" in ml_human
                     and "w1_T-60_T-30__vs__w2_T-30_T" in ml_human["pairwise_corr"]
                     and "error" not in ml_human["pairwise_corr"]["w1_T-60_T-30__vs__w2_T-30_T"])
    if ml_human_avail:
        mhr = ml_human["pairwise_corr"]["w1_T-60_T-30__vs__w2_T-30_T"]["pearson_r"]
        ax.axhline(mhr, color="purple", linestyle="--", alpha=0.5,
                   label=f"ML within-domain human-self r={mhr:.3f}")
    ax.axhline(0.53, color="red", linestyle=":", alpha=0.4,
               label="Sheeran 2002 cross-domain r=0.53")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Spearman ρ (intent vs revealed)")
    ax.set_title("Pooled vs within-bucket Spearman: H&M vs MovieLens")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(RESULTS / "phase27_domain_comparison.png", dpi=130)
    plt.close()

    summary = {
        "rows": rows,
        "hm_human_self_r": hm_human.get("pairwise_corr", {}).get("w1_T-60_T-30__vs__w2_T-30_T", {}).get("pearson_r"),
        "ml_human_self_r": (ml_human.get("pairwise_corr", {})
                            .get("w1_T-60_T-30__vs__w2_T-30_T", {}).get("pearson_r") if ml_human_avail else None),
        "sheeran_cross_domain": 0.53,
        "leakage_H&M": {
            "abs_Delta_F": pair.get("leakage_dominates_absdiff", {}).get("abs_leakage_contribution"),
            "abs_Delta_arch": pair.get("leakage_dominates_absdiff", {}).get("abs_arch_contribution"),
        },
        "leakage_ML": {
            "gap_Fbase_minus_gap_Fnobase": ml.get("paired_diff_F-base_minus_F-nobase", {}).get("diff_of_gaps"),
            "95CI": ml.get("paired_diff_F-base_minus_F-nobase", {}).get("95CI"),
        },
    }
    (RESULTS / "phase27_domain_comparison.json").write_text(json.dumps(summary, indent=2, default=str))
    print("Phase 27 saved: pooled vs within Spearman, side-by-side; reference human-self lines included.")


if __name__ == "__main__":
    main()
