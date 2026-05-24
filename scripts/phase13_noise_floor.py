"""Phase 13: temporal noise-floor of the LLM's stated_intent_prob.

Re-run the SAME trace through the deliberation step three times with three
different cache-busting nonces and temperature=0. Because temperature=0 is
deterministic, the three runs should agree EXACTLY — any disagreement
indicates Gemini's own non-determinism. Compare the resulting |Δ| to the
counterfactual perturbation Δ from Phase 11c: if noise_floor ≈ perturbation_Δ,
the LLM is not actually reasoning about the specific trace.

We use a small `_nonce` field in the prompt (declared "ignore this") to bypass
the disk cache for the second and third pass.
"""

from __future__ import annotations
import json
import copy
from pathlib import Path
import numpy as np

from src import T_TEST_CUTOFF, SEED
from src.behavioral_trace import behavioral_trace
from src.cognition_fragment.attention import rank_attention
from src.cognition_fragment.memory import retrieve_memories
from src.cognition_fragment.affect import compute_affect
from src.cognition_fragment.deliberation import build_prompt, parse_response, SYSTEM_PROMPT_BASE
from src.llm_client import call_llm

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

N_CUSTOMERS = 50
N_REPEATS = 3


def _score_with_nonce(trace, attention, memories, affect, include_base_rate, nonce: str, model: str = "gemini-2.5-flash"):
    prompt = build_prompt(trace, attention, memories, affect, include_base_rate)
    # Cache-bust by appending an invisible-to-task nonce.
    busted = prompt + f"\n\n<!-- ignore this token, used to bypass cache: {nonce} -->"
    resp = call_llm(model, busted, system=SYSTEM_PROMPT_BASE, max_tokens=400)
    return parse_response(resp["text"]), resp.get("cost", 0.0)


def main():
    p = RESULTS / "phase10_F-nobase_scores.npz"
    if not p.exists():
        raise SystemExit("Run Phase 10 F-nobase first.")
    d = np.load(p, allow_pickle=True)
    cids = list(d["customer_id"])
    rng = np.random.default_rng(SEED)
    sample = [cids[i] for i in rng.permutation(len(cids))[:N_CUSTOMERS]]

    print(f"[13] computing traces for {len(sample)} customers")
    traces = behavioral_trace(sample, cutoff=T_TEST_CUTOFF)
    rows = []
    for i, cid in enumerate(sample):
        if cid not in traces:
            continue
        t = traces[cid]
        a = rank_attention(t)
        m = retrieve_memories(t)
        af = compute_affect(t)
        runs = []
        for r in range(N_REPEATS):
            parsed, cost = _score_with_nonce(t, a, m, af, include_base_rate=False,
                                              nonce=f"noise_{r}_{cid[:6]}")
            runs.append(parsed["stated_intent_prob"])
        spread = float(np.max(runs) - np.min(runs))
        std = float(np.std(runs))
        rows.append({"cid": cid[:12] + "…", "runs": runs, "spread": spread, "std": std})
        if i < 5:
            print(f"  {cid[:8]}…: runs={runs}, spread={spread:.4f}")

    spreads = np.array([r["spread"] for r in rows])
    stds = np.array([r["std"] for r in rows])
    out = {
        "n_customers": len(rows),
        "n_repeats_each": N_REPEATS,
        "mean_max_minus_min_spread": float(spreads.mean()),
        "median_spread": float(np.median(spreads)),
        "max_spread": float(spreads.max()),
        "mean_std_within_customer": float(stds.mean()),
        "samples": rows[:20],
    }
    (RESULTS / "phase13_noise_floor.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[13] mean max-min spread = {out['mean_max_minus_min_spread']:.4f}")
    print(f"[13] mean within-customer std = {out['mean_std_within_customer']:.4f}")
    print(f"[13] → counterfactual perturbation must EXCEED this noise floor to be meaningful.")


if __name__ == "__main__":
    main()
