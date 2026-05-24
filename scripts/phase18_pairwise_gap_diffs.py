"""Phase 18: pairwise gap-difference paired bootstrap.

All three F-base, F-nobase, D2-core arms scored on the same core-1000 customers.
Compute paired bootstrap (stratified within bucket) on:
  - gap(F-base)   − gap(F-nobase)
  - gap(F-nobase) − gap(D2-core)
  - gap(F-base)   − gap(D2-core)
+ paired Wilcoxon on per-customer (stated_i − actual_i) differences.

This is the inferential support for "leakage dominates."
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

from src import SEED

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BUCKETS_ORDER = ["1", "2-5", "6-20", "21-100", "101+"]
B_BOOT = 1000


def _load(name: str):
    d = np.load(RESULTS / f"phase10_{name}_scores.npz", allow_pickle=True)
    return {
        "cids": list(d["customer_id"]),
        "scores": d["stated_intent_raw"].astype(float) if "stated_intent_raw" in d.files else d["stated_intent"].astype(float),
        "actual": d["actual"].astype(int),
        "buckets": d["activity_bucket"].astype(str),
    }


def _stratified_bootstrap_diff(scores_a, scores_b, actual, buckets, B=B_BOOT, seed=SEED):
    """gap(A) − gap(B) on paired customers. Stratified within bucket."""
    rng = np.random.default_rng(seed)
    bucket_to_idx = {b: np.where(buckets == b)[0] for b in BUCKETS_ORDER if (buckets == b).any()}
    diffs = []
    for _ in range(B):
        sampled = np.concatenate([
            rng.choice(idx, len(idx), replace=True) for idx in bucket_to_idx.values()
        ])
        gap_a = scores_a[sampled].mean() - actual[sampled].mean()
        gap_b = scores_b[sampled].mean() - actual[sampled].mean()
        diffs.append(gap_a - gap_b)
    diffs = np.array(diffs)
    point = (scores_a.mean() - actual.mean()) - (scores_b.mean() - actual.mean())
    return {"point": float(point), "lo": float(np.quantile(diffs, 0.025)),
            "hi": float(np.quantile(diffs, 0.975)), "se": float(diffs.std()), "B": B}


def main():
    fb = _load("F-base")
    fnb = _load("F-nobase")
    d2 = _load("D2-core")

    # Align all three on the intersection of customer_ids (same core-1000 in design).
    fb_idx = {c: i for i, c in enumerate(fb["cids"])}
    fnb_idx = {c: i for i, c in enumerate(fnb["cids"])}
    d2_idx = {c: i for i, c in enumerate(d2["cids"])}
    common = [c for c in fb["cids"] if c in fnb_idx and c in d2_idx]

    fb_s = np.array([fb["scores"][fb_idx[c]] for c in common])
    fnb_s = np.array([fnb["scores"][fnb_idx[c]] for c in common])
    d2_s = np.array([d2["scores"][d2_idx[c]] for c in common])
    actual = np.array([fb["actual"][fb_idx[c]] for c in common])
    buckets = np.array([fb["buckets"][fb_idx[c]] for c in common])

    out = {"n_paired": len(common), "bootstrap_B": B_BOOT}

    pairs = [
        ("F-base_minus_F-nobase", fb_s, fnb_s),
        ("F-nobase_minus_D2-core", fnb_s, d2_s),
        ("F-base_minus_D2-core", fb_s, d2_s),
    ]
    for name, a, b in pairs:
        boot = _stratified_bootstrap_diff(a, b, actual, buckets)
        # Wilcoxon on per-customer paired |err|
        wstat, wp = wilcoxon(np.abs(a - actual) - np.abs(b - actual), alternative="two-sided")
        out[name] = {
            "diff_of_gaps": boot["point"],
            "diff_of_gaps_95CI": [boot["lo"], boot["hi"]],
            "diff_of_gaps_significant": (boot["lo"] > 0) or (boot["hi"] < 0),
            "wilcoxon_paired_abs_err_p_two_sided": float(wp),
        }
        print(f"[18] {name}: Δgap={boot['point']:+.4f}  95% CI [{boot['lo']:+.4f}, {boot['hi']:+.4f}]  "
              f"Wilcoxon p={wp:.4g}")

    # Leakage-dominates inferential test: |Δ_F| vs |Δ_arch| with both as gap-diff CIs.
    leak = abs(out["F-base_minus_F-nobase"]["diff_of_gaps"])
    arch = abs(out["F-nobase_minus_D2-core"]["diff_of_gaps"])
    out["leakage_dominates_absdiff"] = {
        "abs_leakage_contribution": leak,
        "abs_arch_contribution": arch,
        "leakage_minus_arch": leak - arch,
        "leakage_dominates": leak > arch,
    }
    (RESULTS / "phase18_pairwise_gap_diffs.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"[18] |Δ_F|={leak:.4f}  |Δ_arch|={arch:.4f}  leakage_dominates={leak>arch}")


if __name__ == "__main__":
    main()
