"""End-to-end H&M Fragment-cognition orchestrator.

For each customer_id: behavioral_trace → attention → memory → affect →
deliberation (1 LLM call) → decision. Returns a dict per customer.
"""

from __future__ import annotations
from datetime import date

from ..behavioral_trace import behavioral_trace
from .attention import rank_attention
from .memory import retrieve_memories
from .affect import compute_affect
from .deliberation import deliberate
from .decision import blend_decision


def run_pipeline(customer_ids: list[str], cutoff: str | date,
                 include_base_rate: bool, model: str = "gemini-2.5-flash",
                 verbose: bool = False) -> dict[str, dict]:
    """Run the full cognition pipeline. Returns dict[cid] -> full result dict."""
    cutoff_d = date.fromisoformat(cutoff) if isinstance(cutoff, str) else cutoff
    traces = behavioral_trace(customer_ids, cutoff=cutoff_d)

    out: dict[str, dict] = {}
    for i, cid in enumerate(customer_ids):
        if cid not in traces:
            out[cid] = {"error": "no_trace", "stated_intent_prob": 0.5}
            continue
        trace = traces[cid]
        attention = rank_attention(trace)
        memories = retrieve_memories(trace)
        affect = compute_affect(trace)
        deliberation = deliberate(trace, attention, memories, affect,
                                  include_base_rate=include_base_rate, model=model)
        decision = blend_decision(deliberation, affect, trace)
        out[cid] = {
            "trace_summary": {
                "total_orders": trace["purchase_stats"]["total_orders"],
                "recency_days": trace["purchase_stats"]["recency_days"],
                "personality": trace["personality"],
                "primary_focus": attention["primary_focus"],
            },
            "affect": affect,
            "deliberation": {
                "reasoning": deliberation["reasoning"],
                "verbatim_reaction": deliberation["verbatim_reaction"],
                "key_objection": deliberation["key_objection"],
                "baseline_30d_buy_likelihood": deliberation["baseline_30d_buy_likelihood"],
                "stimulus_30d_buy_likelihood": deliberation["stimulus_30d_buy_likelihood"],
                "friction_score": deliberation["friction_score"],
                "confidence": deliberation["confidence"],
            },
            "decision": decision,
            "stated_intent_prob": decision["stated_intent_prob_final"],
            "cost": deliberation.get("_cost", 0.0),
        }
        if verbose and i < 3:
            print(f"[fragment] {cid[:8]}... raw={decision['stated_intent_prob_raw']:.3f} "
                  f"final={decision['stated_intent_prob_final']:.3f} "
                  f"verbatim={deliberation['verbatim_reaction'][:80]}")
    return out
