"""Phase 29: MovieLens noise floor — re-run same trace 3× with cache-busting nonces.
Mirrors H&M Phase 13.
"""

from __future__ import annotations
import json
import numpy as np
from pathlib import Path

from src import SEED
from src.movielens_data import T_TEST_CUTOFF, behavioral_trace_ml
from src.cognition_fragment.attention import rank_attention
from src.cognition_fragment.memory import retrieve_memories
from src.cognition_fragment.affect import compute_affect
from src.cognition_fragment.deliberation import build_prompt, parse_response
from scripts.phase22_movielens_arms import SYSTEM_PROMPT_ML, build_prompt as build_prompt_ml
from src.llm_client import call_llm

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main():
    d = np.load(RESULTS / "phase22_ml_F-nobase_scores.npz", allow_pickle=True)
    uids = list(d["user_id"])
    rng = np.random.default_rng(SEED + 300)
    sample = [int(uids[i]) for i in rng.permutation(len(uids))[:50]]
    traces = behavioral_trace_ml(sample, cutoff=T_TEST_CUTOFF)

    rows = []
    for i, uid in enumerate(sample):
        if uid not in traces:
            continue
        t = traces[uid]
        a = rank_attention(t); m = retrieve_memories(t); af = compute_affect(t)
        runs = []
        for r_i in range(3):
            prompt = build_prompt_ml(t, a, m, af, include_base_rate=False)
            busted = prompt + f"\n\n<!-- nonce {r_i}_{uid} -->"
            resp = call_llm("gemini-2.5-flash", busted, system=SYSTEM_PROMPT_ML, max_tokens=400)
            parsed = parse_response(resp["text"])
            runs.append(float(parsed["stated_intent_prob"]))
        spread = float(max(runs) - min(runs))
        std = float(np.std(runs))
        rows.append({"uid": uid, "runs": runs, "spread": spread, "std": std})
        if i < 5:
            print(f"  {uid}: runs={runs}, spread={spread:.4f}")

    spreads = np.array([r["spread"] for r in rows])
    stds = np.array([r["std"] for r in rows])
    out = {
        "n_users": len(rows), "n_repeats": 3,
        "mean_spread": float(spreads.mean()),
        "median_spread": float(np.median(spreads)),
        "mean_std": float(stds.mean()),
        "samples": rows[:20],
    }
    (RESULTS / "phase29_ml_noise_floor.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"[29] mean spread = {out['mean_spread']:.4f}, mean std = {out['mean_std']:.4f}")


if __name__ == "__main__":
    main()
