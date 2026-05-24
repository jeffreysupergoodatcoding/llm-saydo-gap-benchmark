"""Phase 23: cross-domain analysis on MovieLens — replicate Phase 18 (paired
gap diffs) + Phase 19 (Spearman decomposition).

Both arms are evaluated on the SAME MovieLens test users, so the paired
structure is automatic. We don't have a D2-flat-on-ML arm, so the pairwise
test is F-base vs F-nobase only — the leakage-vs-architecture decomposition.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, wilcoxon

from src import SEED

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BUCKETS_ORDER = ["1", "2-5", "6-20", "21-100", "101+"]


def _load(name: str):
    p = RESULTS / f"phase22_ml_{name}_scores.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    return {
        "scores": d["stated_intent_raw"].astype(float),
        "actual": d["actual"].astype(int),
        "buckets": d["activity_bucket"].astype(str),
        "uids": list(d["user_id"]),
    }


def _bootstrap_spearman(scores, actual, B=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(scores)
    boots = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        rho, _ = spearmanr(scores[idx], actual[idx])
        if np.isfinite(rho):
            boots.append(rho)
    rho_p, p_ = spearmanr(scores, actual)
    return {"rho": float(rho_p) if np.isfinite(rho_p) else None,
            "p": float(p_) if np.isfinite(p_) else None,
            "lo": float(np.quantile(boots, 0.025)),
            "hi": float(np.quantile(boots, 0.975)), "B": B}


def main():
    fb = _load("F-base")
    fnb = _load("F-nobase")
    if fb is None or fnb is None:
        raise SystemExit("[23] Need both MovieLens arms.")
    # Both arms are run on the same userIds; verify
    assert fb["uids"] == fnb["uids"], "MovieLens user_id order should match across arms"

    out = {}

    # --- Per-arm gap + PR-AUC + Spearman ---
    for arm in [fb, fnb]:
        arm_name = "F-base" if arm is fb else "F-nobase"
        gap = float(arm["scores"].mean() - arm["actual"].mean())
        pooled = _bootstrap_spearman(arm["scores"], arm["actual"])
        # Within-bucket residual Spearman
        res_s = np.zeros_like(arm["scores"], dtype=float)
        res_a = np.zeros_like(arm["actual"], dtype=float)
        for b in BUCKETS_ORDER:
            m = arm["buckets"] == b
            if m.sum() < 2:
                continue
            res_s[m] = arm["scores"][m] - arm["scores"][m].mean()
            res_a[m] = arm["actual"][m] - arm["actual"][m].mean()
        within = _bootstrap_spearman(res_s, res_a)
        per_bucket = {}
        for b in BUCKETS_ORDER:
            m = arm["buckets"] == b
            if m.sum() < 10:
                continue
            per_bucket[b] = {"n": int(m.sum()),
                             "mean_stated": float(arm["scores"][m].mean()),
                             "mean_actual": float(arm["actual"][m].mean()),
                             "signed_gap": float(arm["scores"][m].mean() - arm["actual"][m].mean())}
        out[arm_name] = {
            "n": len(arm["scores"]),
            "mean_stated": float(arm["scores"].mean()),
            "mean_actual": float(arm["actual"].mean()),
            "signed_gap": gap,
            "pooled_spearman": pooled,
            "within_bucket_spearman": within,
            "per_bucket": per_bucket,
        }

    # --- Paired gap diff: F-base vs F-nobase ---
    diff_arr = fb["scores"] - fnb["scores"]
    actual = fb["actual"]
    gap_diff = (fb["scores"].mean() - actual.mean()) - (fnb["scores"].mean() - actual.mean())
    rng = np.random.default_rng(SEED)
    bucket_to_idx = {b: np.where(fb["buckets"] == b)[0] for b in BUCKETS_ORDER if (fb["buckets"] == b).any()}
    diffs_boot = []
    for _ in range(1000):
        sampled = np.concatenate([rng.choice(idx, len(idx), replace=True) for idx in bucket_to_idx.values()])
        d = (fb["scores"][sampled].mean() - actual[sampled].mean()) - (fnb["scores"][sampled].mean() - actual[sampled].mean())
        diffs_boot.append(d)
    diffs_boot = np.array(diffs_boot)
    err_fb = np.abs(fb["scores"] - actual)
    err_fnb = np.abs(fnb["scores"] - actual)
    wstat, wp = wilcoxon(err_fb - err_fnb, alternative="two-sided")
    out["paired_diff_F-base_minus_F-nobase"] = {
        "diff_of_gaps": gap_diff,
        "95CI": [float(np.quantile(diffs_boot, 0.025)), float(np.quantile(diffs_boot, 0.975))],
        "abs_leakage_contribution": abs(gap_diff),
        "wilcoxon_p_paired_abs_err": float(wp),
    }

    # --- Replication verdicts vs H&M ---
    out["replication_verdicts"] = {
        "leakage_pattern_present": (gap_diff < 0),  # F-base gap < F-nobase gap means table reduced inflation
        "pooled_vs_within_decomposition": {
            "F-base": {
                "pooled_ρ": out["F-base"]["pooled_spearman"]["rho"],
                "within_ρ": out["F-base"]["within_bucket_spearman"]["rho"],
                "ratio_within_over_pooled": (
                    out["F-base"]["within_bucket_spearman"]["rho"] / out["F-base"]["pooled_spearman"]["rho"]
                    if (out["F-base"]["pooled_spearman"]["rho"] and abs(out["F-base"]["pooled_spearman"]["rho"]) > 1e-6) else None
                ),
            },
            "F-nobase": {
                "pooled_ρ": out["F-nobase"]["pooled_spearman"]["rho"],
                "within_ρ": out["F-nobase"]["within_bucket_spearman"]["rho"],
                "ratio_within_over_pooled": (
                    out["F-nobase"]["within_bucket_spearman"]["rho"] / out["F-nobase"]["pooled_spearman"]["rho"]
                    if (out["F-nobase"]["pooled_spearman"]["rho"] and abs(out["F-nobase"]["pooled_spearman"]["rho"]) > 1e-6) else None
                ),
            },
        },
    }

    (RESULTS / "phase23_ml_analysis.json").write_text(json.dumps(out, indent=2, default=str))
    print("=== MovieLens cross-domain replication ===")
    for arm_name in ["F-base", "F-nobase"]:
        a = out[arm_name]
        print(f"\n{arm_name}: n={a['n']}, mean_stated={a['mean_stated']:.4f}, "
              f"mean_actual={a['mean_actual']:.4f}, gap={a['signed_gap']:+.4f}")
        ps = a['pooled_spearman']
        ws = a['within_bucket_spearman']
        print(f"  pooled ρ = {ps['rho']:+.3f} [{ps['lo']:+.3f}, {ps['hi']:+.3f}]   "
              f"within-bucket ρ = {ws['rho']:+.3f} [{ws['lo']:+.3f}, {ws['hi']:+.3f}]")
    print(f"\nPaired diff: gap(F-base) - gap(F-nobase) = {out['paired_diff_F-base_minus_F-nobase']['diff_of_gaps']:+.4f} "
          f"95% CI {out['paired_diff_F-base_minus_F-nobase']['95CI']}")
    print(f"Replication verdict — leakage pattern (table reduces inflation): "
          f"{out['replication_verdicts']['leakage_pattern_present']}")
    pvw = out["replication_verdicts"]["pooled_vs_within_decomposition"]
    print(f"Simpson's-paradox replication — F-base: pooled={pvw['F-base']['pooled_ρ']:.3f}, within={pvw['F-base']['within_ρ']:.3f}; "
          f"F-nobase: pooled={pvw['F-nobase']['pooled_ρ']:.3f}, within={pvw['F-nobase']['within_ρ']:.3f}")


if __name__ == "__main__":
    main()
