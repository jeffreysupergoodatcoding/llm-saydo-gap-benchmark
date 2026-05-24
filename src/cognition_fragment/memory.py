"""Top-k memory retrieval for H&M. Deterministic.

Ranks recent purchases + behavioral patterns by relevance to the next-purchase
prediction task, picks top k=5 to surface in the LLM prompt.
"""

from __future__ import annotations
from . import TOP_K_MEMORIES


def retrieve_memories(trace: dict, k: int = TOP_K_MEMORIES) -> list[dict]:
    """Top-k memories: latest purchases + behavioral pattern flags."""
    memories: list[dict] = []

    # Last few purchases (each is a memory)
    for r in trace["recent_purchases"][:k]:
        days_ago = r["days_ago"]
        # Recency-weighted relevance: half-life of 30 days.
        relevance = 0.5 ** (days_ago / 30.0)
        memories.append({
            "type": "purchase",
            "relevance": round(relevance, 3),
            "summary": f"{days_ago}d ago: {r['prod_name']} ({r['product_type']}, {r['color']}) "
                       f"in {r['section']} via {r['channel']}",
        })

    # Behavioral pattern memories
    flags = trace["derived_flags"]
    ps = trace["purchase_stats"]
    if flags["is_lapsed"]:
        memories.append({"type": "pattern", "relevance": 0.6,
                         "summary": f"Lapsed: no purchase in {ps['recency_days']} days."})
    if flags["is_new_to_brand"]:
        memories.append({"type": "pattern", "relevance": 0.7,
                         "summary": "Made exactly one purchase ever."})
    if trace["personality"] == "habit-driven":
        memories.append({"type": "pattern", "relevance": 0.5,
                         "summary": f"Habit-driven: returns to same {trace['product_summary']['top_section']} section."})
    if trace["personality"] == "novelty-seeking":
        memories.append({"type": "pattern", "relevance": 0.5,
                         "summary": f"Novelty-seeking: tried {trace['product_summary']['sections_seen']} different sections."})

    # Cadence memory
    avg = trace["timeline"].get("avg_inter_purchase_days")
    if avg is not None and avg < 30:
        memories.append({"type": "pattern", "relevance": 0.55,
                         "summary": f"Cadence: avg {avg:.1f} days between purchases (frequent)."})
    elif avg is not None and avg > 120:
        memories.append({"type": "pattern", "relevance": 0.4,
                         "summary": f"Cadence: avg {avg:.1f} days between purchases (infrequent)."})

    memories.sort(key=lambda m: -m["relevance"])
    return memories[:k]
