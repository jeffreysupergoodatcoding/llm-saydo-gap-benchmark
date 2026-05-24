"""Phase 20: rank-invariance check under weighting + B audit."""

from __future__ import annotations
import json
import re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BUCKETS_ORDER = ["1", "2-5", "6-20", "21-100", "101+"]
# Test-pool bucket counts from phase1_summary.json
TEST_DIST = {"1": 9302, "2-5": 9754, "6-20": 9817, "21-100": 9717, "101+": 8275}


def _load(name: str):
    d = np.load(RESULTS / f"phase10_{name}_scores.npz", allow_pickle=True)
    return {
        "scores": d["stated_intent_raw"].astype(float) if "stated_intent_raw" in d.files else d["stated_intent"].astype(float),
        "actual": d["actual"].astype(int),
        "buckets": d["activity_bucket"].astype(str),
    }


def _gap_raw(scores, actual):
    return float(scores.mean() - actual.mean())


def _gap_test_reweighted(scores, actual, buckets):
    total = sum(TEST_DIST.values())
    w_stat, w_act = 0.0, 0.0
    for b in BUCKETS_ORDER:
        mask = buckets == b
        if mask.sum() == 0:
            continue
        w = TEST_DIST[b] / total
        w_stat += w * scores[mask].mean()
        w_act += w * actual[mask].mean()
    return float(w_stat - w_act)


def _gap_bucket_uniform(scores, actual, buckets):
    bs = []
    for b in BUCKETS_ORDER:
        mask = buckets == b
        if mask.sum() == 0:
            continue
        bs.append(scores[mask].mean() - actual[mask].mean())
    return float(np.mean(bs)) if bs else float("nan")


def main():
    arms = {}
    for name in ["F-base", "F-nobase", "D2-core"]:
        p = RESULTS / f"phase10_{name}_scores.npz"
        if p.exists():
            arms[name] = _load(name)

    out = {"arms": {}}
    for name, a in arms.items():
        out["arms"][name] = {
            "gap_raw": _gap_raw(a["scores"], a["actual"]),
            "gap_test_reweighted": _gap_test_reweighted(a["scores"], a["actual"], a["buckets"]),
            "gap_bucket_uniform": _gap_bucket_uniform(a["scores"], a["actual"], a["buckets"]),
        }

    # Rank invariance across weightings
    for w in ["gap_raw", "gap_test_reweighted", "gap_bucket_uniform"]:
        ranking = sorted(out["arms"].keys(), key=lambda n: out["arms"][n][w])
        out[f"rank_under_{w}"] = ranking

    out["rank_invariant"] = (out["rank_under_gap_raw"] == out["rank_under_gap_test_reweighted"] == out["rank_under_gap_bucket_uniform"])

    # B audit: grep all phase11/13/14/15/16/17/18/19 scripts for hard-coded B values
    scripts = list((ROOT / "scripts").glob("phase1[1-9]_*.py"))
    b_usage = {}
    for s in scripts:
        text = s.read_text()
        b_usage[s.name] = list(set(re.findall(r"B\s*=\s*(\d+)", text) + re.findall(r"n_resamples\s*=\s*(\d+)", text)))
    out["B_audit"] = b_usage

    (RESULTS / "phase20_reweighting_and_B.json").write_text(json.dumps(out, indent=2, default=str))
    print("[20] Per-arm gap under three weightings:")
    for name, vals in out["arms"].items():
        print(f"  {name}: raw={vals['gap_raw']:+.4f}  reweighted={vals['gap_test_reweighted']:+.4f}  bucket_uniform={vals['gap_bucket_uniform']:+.4f}")
    print(f"[20] Rank invariant across weightings: {out['rank_invariant']}")
    print(f"[20] B audit per script:")
    for s, vals in b_usage.items():
        print(f"    {s}: {vals}")


if __name__ == "__main__":
    main()
