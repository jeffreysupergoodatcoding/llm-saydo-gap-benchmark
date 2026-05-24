"""Deterministic attention/salience scoring for H&M.

Ranks which features of the customer the LLM should attend to first.
Faithful adaptation of Fragment Labs' attention.py logic to the H&M trace schema.
"""

from __future__ import annotations


def rank_attention(trace: dict) -> dict:
    """Return ordered salience features and a primary/secondary focus."""
    ps = trace["purchase_stats"]
    sums = trace["product_summary"]
    flags = trace["derived_flags"]

    scores: dict[str, float] = {}

    # Recency dominates: fresh customer signals strong base rate.
    if ps["recency_days"] <= 14:
        scores["recency_fresh"] = 1.0
    elif ps["recency_days"] <= 60:
        scores["recency_recent"] = 0.7
    elif ps["recency_days"] <= 180:
        scores["recency_moderate"] = 0.4
    else:
        scores["recency_lapsed"] = 0.2

    # Frequency / volume.
    if ps["total_orders"] >= 50:
        scores["heavy_buyer"] = 1.0
    elif ps["total_orders"] >= 10:
        scores["regular_buyer"] = 0.7
    elif ps["total_orders"] >= 2:
        scores["light_buyer"] = 0.4
    else:
        scores["single_purchase"] = 0.2

    # Diversity vs concentration.
    if flags["is_diverse_shopper"]:
        scores["category_explorer"] = 0.6
    elif sums["sections_seen"] == 1:
        scores["section_loyalist"] = 0.6

    # AOV.
    if ps["aov"] >= 0.04:
        scores["premium_aov"] = 0.5
    elif ps["aov"] <= 0.015:
        scores["budget_aov"] = 0.5

    # Channel preference.
    if ps["channel2_share"] >= 0.9:
        scores["online_exclusive"] = 0.4
    elif ps["channel2_share"] <= 0.1:
        scores["store_exclusive"] = 0.4

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return {
        "ranked": ranked,
        "primary_focus": ranked[0][0] if ranked else None,
        "secondary_focus": ranked[1][0] if len(ranked) > 1 else None,
    }
