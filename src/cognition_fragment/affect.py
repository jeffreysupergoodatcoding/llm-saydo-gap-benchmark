"""Deterministic pre-LLM friction estimate for H&M.

Six friction components weighted with the WIP defaults (FRICTION_WEIGHTS),
producing a 0..100 friction score that the deliberation stage will see and
the decision stage will blend with the LLM's own friction estimate.
"""

from __future__ import annotations
from . import FRICTION_WEIGHTS


def _component_scores(trace: dict) -> dict[str, float]:
    """Each component returns a friction value in [0, 100]; 0 = no friction,
    100 = strong friction against next-30-day purchase."""
    ps = trace["purchase_stats"]
    flags = trace["derived_flags"]
    sums = trace["product_summary"]

    # Price: higher AOV → less price friction (assumes affluent customer).
    # Normalised by H&M's typical AOV range ~0.005-0.08.
    if ps["aov"] >= 0.04:
        price_friction = 20.0
    elif ps["aov"] >= 0.02:
        price_friction = 40.0
    else:
        price_friction = 65.0

    # Trust: recency-driven; lapsed = high friction.
    if ps["recency_days"] <= 14:
        trust_friction = 10.0
    elif ps["recency_days"] <= 60:
        trust_friction = 25.0
    elif ps["recency_days"] <= 180:
        trust_friction = 55.0
    else:
        trust_friction = 80.0

    # Decision: ambivalence proxy. New-to-brand single-purchase customer has high decision friction.
    if flags["is_new_to_brand"]:
        decision_friction = 75.0
    elif ps["total_orders"] >= 10:
        decision_friction = 20.0
    else:
        decision_friction = 50.0

    # Channel: split-channel customers have moderate friction (no strong preference).
    if ps["channel2_share"] >= 0.9 or ps["channel2_share"] <= 0.1:
        channel_friction = 15.0  # clear channel preference
    else:
        channel_friction = 45.0

    # Memory: do recent purchases include similar items? Proxy: section_loyalty.
    if sums["sections_seen"] == 1:
        memory_friction = 15.0  # repeat-section will get same recommendations
    elif sums["sections_seen"] <= 3:
        memory_friction = 30.0
    else:
        memory_friction = 55.0  # diverse: less memory pull to any one section

    # Product relevance: AOV + diversity proxy.
    if flags["is_diverse_shopper"]:
        product_friction = 30.0
    elif sums["sections_seen"] >= 2:
        product_friction = 40.0
    else:
        product_friction = 55.0

    return {
        "price": price_friction,
        "trust": trust_friction,
        "decision": decision_friction,
        "channel": channel_friction,
        "memory": memory_friction,
        "product_relevance": product_friction,
    }


def compute_affect(trace: dict) -> dict:
    """Return {components, blended_friction (0-100), gut_reaction}"""
    comps = _component_scores(trace)
    blended = sum(comps[k] * FRICTION_WEIGHTS[k] for k in comps)

    if blended < 30:
        gut = "warm"
    elif blended < 55:
        gut = "neutral"
    elif blended < 75:
        gut = "cool"
    else:
        gut = "cold"

    return {
        "components": {k: round(v, 1) for k, v in comps.items()},
        "blended_friction": round(blended, 1),
        "gut_reaction": gut,
    }
