"""Phase 11c: counterfactual trace perturbation (audit Addition 3).

For 50 random customers from the core-1000, generate a minimally perturbed
trace and re-run F-nobase. Report mean |Δ stated_intent_prob| and the
verbatim cosine shift. If both are small, the LLM is anchoring on global
priors not the specific trace.
"""

from __future__ import annotations
import json
import copy
import numpy as np
from pathlib import Path

from src import SEED, T_TEST_CUTOFF
from src.cognition_fragment.run import run_pipeline
from src.behavioral_trace import behavioral_trace
from src.cognition_fragment.attention import rank_attention
from src.cognition_fragment.memory import retrieve_memories
from src.cognition_fragment.affect import compute_affect
from src.cognition_fragment.deliberation import deliberate
from src.cognition_fragment.decision import blend_decision

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _perturb_trace(trace: dict, rng: np.random.Generator) -> dict:
    """Minimal perturbation: swap exactly one color and one product_type in one
    of the recent purchases. AUDIT FIX (Agent B): the previous version
    decremented total_orders and distinct_articles, which is *not* a minimal
    perturbation — it changes the aggregated summary stats that the LLM
    explicitly attends to. The minimal version only swaps surface attributes
    in one event."""
    t = copy.deepcopy(trace)
    if not t["recent_purchases"]:
        return t
    idx = int(rng.integers(0, len(t["recent_purchases"])))

    # Swap color
    palette = ["Black", "White", "Light Pink", "Dark Blue", "Beige", "Light Beige"]
    current_color = t["recent_purchases"][idx].get("color", "")
    options = [c for c in palette if c != current_color] or palette
    t["recent_purchases"][idx]["color"] = options[int(rng.integers(0, len(options)))]

    # Swap product_type (audit-required)
    pt_pool = ["T-shirt", "Trousers", "Dress", "Sweater", "Jacket", "Shorts", "Skirt"]
    current_pt = t["recent_purchases"][idx].get("product_type", "")
    pt_options = [p for p in pt_pool if p != current_pt] or pt_pool
    t["recent_purchases"][idx]["product_type"] = pt_options[int(rng.integers(0, len(pt_options)))]
    return t


def _run_one(trace: dict, include_base_rate: bool) -> dict:
    attention = rank_attention(trace)
    memories = retrieve_memories(trace)
    affect = compute_affect(trace)
    deliberation = deliberate(trace, attention, memories, affect,
                              include_base_rate=include_base_rate, model="gemini-2.5-flash")
    decision = blend_decision(deliberation, affect, trace)
    return {
        "stated_intent_prob": decision["stated_intent_prob_final"],
        "verbatim": deliberation["verbatim_reaction"],
        "raw": deliberation,
    }


def main():
    # Use the F-nobase core-1000 customers we already scored
    p = RESULTS / "phase10_F-nobase_scores.npz"
    if not p.exists():
        raise SystemExit("Run Phase 10 F-nobase first.")
    d = np.load(p, allow_pickle=True)
    cids = list(d["customer_id"])
    original_intents = d["stated_intent"].astype(float)
    original_verbatim = list(d["verbatim"])
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(cids))[:50]
    sample_cids = [cids[i] for i in perm]
    print(f"[11c] perturbing {len(sample_cids)} customers")

    # Build traces for the sample
    traces = behavioral_trace(sample_cids, cutoff=T_TEST_CUTOFF)

    perturbed = []
    for i, cid in enumerate(sample_cids):
        if cid not in traces:
            continue
        orig_t = traces[cid]
        pert_t = _perturb_trace(orig_t, rng)
        try:
            pert_out = _run_one(pert_t, include_base_rate=False)
        except Exception as e:
            print(f"[11c] error on {cid}: {e}")
            continue
        idx_in_arr = cids.index(cid)
        orig_intent = original_intents[idx_in_arr]
        orig_verb = original_verbatim[idx_in_arr]
        delta_intent = pert_out["stated_intent_prob"] - orig_intent
        perturbed.append({
            "cid": cid[:12] + "…",
            "orig_intent": float(orig_intent),
            "pert_intent": float(pert_out["stated_intent_prob"]),
            "delta_intent": float(delta_intent),
            "orig_verbatim": orig_verb[:120],
            "pert_verbatim": pert_out["verbatim"][:120],
        })
        if i < 5:
            print(f"  {cid[:8]}…: orig={orig_intent:.3f} pert={pert_out['stated_intent_prob']:.3f} Δ={delta_intent:+.3f}")

    deltas = np.array([p["delta_intent"] for p in perturbed])
    abs_deltas = np.abs(deltas)
    # AUDIT FIX (Agent B MAJOR): raise the anchoring-threshold to 0.05. Gemini rounds intent
    # to multiples of 0.05/0.10, so a 0.02 threshold is below the model's output resolution.
    out = {
        "n_perturbed": len(perturbed),
        "mean_abs_delta_intent": float(abs_deltas.mean()),
        "median_abs_delta_intent": float(np.median(abs_deltas)),
        "max_abs_delta_intent": float(abs_deltas.max()) if len(abs_deltas) > 0 else 0.0,
        "anchoring_signal_threshold": 0.05,
        "anchoring_to_priors": bool(abs_deltas.mean() < 0.05),
        "perturbation_type": "minimal: swap color + product_type on one recent purchase",
        "samples": perturbed,   # save ALL (fix from Iteration 1: was [:20])
    }
    (RESULTS / "phase11_counterfactual.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[11c] mean |Δ intent| = {out['mean_abs_delta_intent']:.4f} "
          f"(threshold for prior-anchoring: 0.02)")
    print(f"[11c] anchoring_to_priors: {out['anchoring_to_priors']}")
    print(f"[11c] saved phase11_counterfactual.json")


if __name__ == "__main__":
    main()
