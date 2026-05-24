"""H&M-adapted behavioral trace builder.

Lifts the schema concept from Fragment Labs (purchase + product + timeline +
demographics + personality) but maps every field onto H&M-available data:
no subscription, no email engagement, no Klaviyo. Per the audit's
hyperparameter-freeze requirement, all weights/thresholds are fixed up-front
and never tuned on H&M test data.
"""

from __future__ import annotations
from datetime import date
from functools import wraps
import polars as pl

from . import T_TRAIN_CUTOFF, T_TEST_CUTOFF
from .data import load_transactions, load_articles, load_customers
from .features import cutoff_guard


def _personality_descriptor(distinct_articles: int, total_orders: int, sections_seen: int) -> str:
    """Deterministic personality label from purchase pattern.
    No tuning on H&M test data; these thresholds are intentionally coarse.
    """
    if total_orders == 0:
        return "no-history"
    diversity_ratio = distinct_articles / max(total_orders, 1)
    if diversity_ratio > 0.85 and sections_seen >= 3:
        return "novelty-seeking"
    if diversity_ratio < 0.30 or sections_seen == 1:
        return "habit-driven"
    return "balanced"


@cutoff_guard
def behavioral_trace(customer_ids: list[str], *, cutoff: date, n_recent: int = 20) -> dict[str, dict]:
    """Return dict[customer_id] -> trace dict, suitable for downstream Fragment-style cognition.

    Each trace contains:
        identity:        {age, postal_region}
        purchase_stats:  {total_orders, total_spend, recency_days, tenure_days, aov, channel2_share}
        product_summary: {top_section, top_garment_group, top_color, distinct_articles,
                          sections_seen, garment_groups_seen, colors_seen}
        timeline:        {first_tx_date, last_tx_date, avg_inter_purchase_days}
        recent_purchases: [{days_ago, prod_name, product_type, garment_group, color, section}, ...]
        personality:     descriptor
        derived_flags:   {is_new_to_brand, is_lapsed, is_diverse_shopper}
    """
    if len(customer_ids) == 0:
        return {}

    tx = (
        load_transactions()
        .filter((pl.col("t_dat") < cutoff) & (pl.col("customer_id").is_in(customer_ids)))
        .collect()
    )
    art_cols = ["article_id", "prod_name", "product_type_name", "garment_group_name",
                "colour_group_name", "section_name", "detail_desc"]
    art = load_articles().select(art_cols).collect()
    tx_j = tx.join(art, on="article_id", how="left")

    cust = load_customers().select(["customer_id", "age"]).collect()
    cust_d = {row["customer_id"]: row.get("age") for row in cust.iter_rows(named=True)}

    out: dict[str, dict] = {}
    for cid, group in tx_j.group_by("customer_id"):
        cid = cid[0] if isinstance(cid, tuple) else cid
        group = group.sort("t_dat")
        n_total = len(group)
        first = group["t_dat"].min()
        last = group["t_dat"].max()
        recency_days = (cutoff - last).days
        tenure_days = (cutoff - first).days
        total_spend = float(group["price"].sum())
        aov = float(group["price"].mean())
        channel2_share = float((group["sales_channel_id"] == 2).mean())
        distinct_articles = int(group["article_id"].n_unique())

        sec_counts = group.group_by("section_name").len().sort("len", descending=True)
        gg_counts = group.group_by("garment_group_name").len().sort("len", descending=True)
        color_counts = group.group_by("colour_group_name").len().sort("len", descending=True)
        pt_counts = group.group_by("product_type_name").len().sort("len", descending=True)

        top_section = sec_counts.head(1)["section_name"][0] if len(sec_counts) else None
        top_garment_group = gg_counts.head(1)["garment_group_name"][0] if len(gg_counts) else None
        top_color = color_counts.head(1)["colour_group_name"][0] if len(color_counts) else None
        top_product_type = pt_counts.head(1)["product_type_name"][0] if len(pt_counts) else None

        sections_seen = int(group["section_name"].n_unique())
        garment_groups_seen = int(group["garment_group_name"].n_unique())
        colors_seen = int(group["colour_group_name"].n_unique())

        # Inter-purchase cadence
        dates = sorted(set(group["t_dat"].to_list()))
        if len(dates) >= 2:
            diffs = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
            avg_inter_purchase_days = float(sum(diffs) / len(diffs))
        else:
            avg_inter_purchase_days = None

        # Recent purchases
        recent = group.tail(n_recent)
        recent_list = [
            {
                "days_ago": (cutoff - r["t_dat"]).days,
                "prod_name": r["prod_name"],
                "product_type": r["product_type_name"],
                "garment_group": r["garment_group_name"],
                "color": r["colour_group_name"],
                "section": r["section_name"],
                "price": r["price"],
                "channel": "store" if r["sales_channel_id"] == 1 else "online",
            }
            for r in recent.iter_rows(named=True)
        ]

        personality = _personality_descriptor(distinct_articles, n_total, sections_seen)
        is_new_to_brand = n_total == 1
        is_lapsed = recency_days > 180
        is_diverse_shopper = sections_seen >= 4 and (distinct_articles / max(n_total, 1)) > 0.7

        out[cid] = {
            "identity": {"age": cust_d.get(cid)},
            "purchase_stats": {
                "total_orders": n_total,
                "total_spend": round(total_spend, 4),
                "recency_days": int(recency_days),
                "tenure_days": int(tenure_days),
                "aov": round(aov, 4),
                "channel2_share": round(channel2_share, 3),
                "distinct_articles": distinct_articles,
            },
            "product_summary": {
                "top_section": top_section,
                "top_garment_group": top_garment_group,
                "top_color": top_color,
                "top_product_type": top_product_type,
                "sections_seen": sections_seen,
                "garment_groups_seen": garment_groups_seen,
                "colors_seen": colors_seen,
            },
            "timeline": {
                "first_tx_date": str(first),
                "last_tx_date": str(last),
                "avg_inter_purchase_days": avg_inter_purchase_days,
            },
            "recent_purchases": recent_list,
            "personality": personality,
            "derived_flags": {
                "is_new_to_brand": is_new_to_brand,
                "is_lapsed": is_lapsed,
                "is_diverse_shopper": is_diverse_shopper,
            },
        }
    return out
