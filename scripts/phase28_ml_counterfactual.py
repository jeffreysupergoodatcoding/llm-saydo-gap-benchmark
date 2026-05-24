"""Phase 28: MovieLens counterfactual perturbation (mirrors H&M Phase 11c).

50 users from the F-nobase MovieLens core; minimal perturb: swap one genre + one
recent movie title; rerun deliberation; compare mean |Δ stated_intent|.
"""

from __future__ import annotations
import json, copy, time
from pathlib import Path
import numpy as np
import polars as pl

from src import SEED
from src.movielens_data import T_TEST_CUTOFF, SPLITS, behavioral_trace_ml
from src.cognition_fragment.attention import rank_attention
from src.cognition_fragment.memory import retrieve_memories
from src.cognition_fragment.affect import compute_affect
from scripts.phase22_movielens_arms import deliberate_ml

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _perturb_ml(trace: dict, rng: np.random.Generator) -> dict:
    t = copy.deepcopy(trace)
    if not t["recent_purchases"]:
        return t
    idx = int(rng.integers(0, len(t["recent_purchases"])))
    genres_pool = ["Comedy", "Drama", "Thriller", "Romance", "Action", "Horror", "Sci-Fi"]
    titles_pool = ["Random Title (2018)", "Another Film (2019)", "Some Movie (2017)"]
    current_g = t["recent_purchases"][idx].get("garment_group", "")
    g_options = [g for g in genres_pool if g != current_g] or genres_pool
    t["recent_purchases"][idx]["garment_group"] = g_options[int(rng.integers(0, len(g_options)))]
    t_options = [tt for tt in titles_pool if tt != t["recent_purchases"][idx].get("prod_name", "")] or titles_pool
    t["recent_purchases"][idx]["prod_name"] = t_options[int(rng.integers(0, len(t_options)))]
    return t


def main():
    p = RESULTS / "phase22_ml_F-nobase_scores.npz"
    d = np.load(p, allow_pickle=True)
    uids = list(d["user_id"])
    orig_intent = d["stated_intent_raw"].astype(float)
    rng = np.random.default_rng(SEED + 200)
    sample_idx = rng.permutation(len(uids))[:50]
    sample_uids = [int(uids[i]) for i in sample_idx]
    sample_orig = [float(orig_intent[i]) for i in sample_idx]
    traces = behavioral_trace_ml(sample_uids, cutoff=T_TEST_CUTOFF)

    perturbed = []
    for i, uid in enumerate(sample_uids):
        if uid not in traces:
            continue
        pert = _perturb_ml(traces[uid], rng)
        a = rank_attention(pert); m = retrieve_memories(pert); af = compute_affect(pert)
        try:
            r = deliberate_ml(pert, a, m, af, include_base_rate=False, model="gemini-2.5-flash")
            new_intent = float(r["stated_intent_prob"])
        except Exception as e:
            print(f"err {uid}: {e}")
            continue
        delta = new_intent - sample_orig[i]
        perturbed.append({"uid": int(uid), "orig": sample_orig[i], "pert": new_intent, "delta": delta})
        if i < 5:
            print(f"  {uid}: orig={sample_orig[i]:.3f} pert={new_intent:.3f} Δ={delta:+.3f}")

    deltas = np.array([p["delta"] for p in perturbed])
    out = {"n": len(perturbed),
           "mean_abs_delta": float(np.abs(deltas).mean()),
           "median_abs_delta": float(np.median(np.abs(deltas))),
           "max_abs_delta": float(np.abs(deltas).max() if len(deltas) else 0.0),
           "samples": perturbed}
    (RESULTS / "phase28_ml_counterfactual.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"[28] mean |Δ| = {out['mean_abs_delta']:.4f}, n = {out['n']}")


if __name__ == "__main__":
    main()
