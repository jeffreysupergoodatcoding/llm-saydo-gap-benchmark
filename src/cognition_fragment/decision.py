"""Deterministic post-LLM decision blend + behavioral guardrails."""

from __future__ import annotations
from . import LLM_AFFECT_BLEND_RATIO


def blend_decision(deliberation_result: dict, affect: dict, trace: dict) -> dict:
    """Blend LLM friction (60%) with pre-computed affect friction (40%); apply
    behavioral guardrails; return final scoring dict.

    The 'final_intent_prob' is the audited stated_intent value after
    guardrail clamping. The unmodified 'stated_intent_prob' is preserved so
    that downstream metrics can use the raw LLM output if desired.
    """
    raw_intent = deliberation_result["stated_intent_prob"]
    llm_friction = deliberation_result["friction_score"] or 50.0
    affect_friction = affect["blended_friction"]

    blended_friction = LLM_AFFECT_BLEND_RATIO * llm_friction + (1 - LLM_AFFECT_BLEND_RATIO) * affect_friction
    flags = trace["derived_flags"]
    ps = trace["purchase_stats"]

    # Guardrails (lifted from Fragment defaults; not tuned on H&M):
    final_intent = raw_intent
    guardrail_notes = []

    # Lapsed customers should not exceed 0.25 stated intent (Fragment guardrail).
    if flags["is_lapsed"] and final_intent > 0.25:
        final_intent = 0.25
        guardrail_notes.append("lapsed-cap@0.25")

    # Single-purchase customers should not exceed 0.30.
    if flags["is_new_to_brand"] and final_intent > 0.30:
        final_intent = 0.30
        guardrail_notes.append("single-purchase-cap@0.30")

    # High-friction (>75 blended) should not exceed 0.40.
    if blended_friction > 75 and final_intent > 0.40:
        final_intent = 0.40
        guardrail_notes.append("high-friction-cap@0.40")

    # Very-active customers (101+ orders, recency<=30d) get a floor of 0.45.
    if ps["total_orders"] >= 101 and ps["recency_days"] <= 30 and final_intent < 0.45:
        final_intent = 0.45
        guardrail_notes.append("heavy-active-floor@0.45")

    return {
        "stated_intent_prob_raw": raw_intent,
        "stated_intent_prob_final": final_intent,
        "blended_friction": round(blended_friction, 1),
        "guardrail_notes": guardrail_notes,
    }
