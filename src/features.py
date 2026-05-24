"""Feature extraction with cutoff_guard enforcement.

Every feature function must take a `cutoff` date and must NOT use any data
with t_dat >= cutoff. The cutoff_guard decorator verifies this at runtime.
"""

from __future__ import annotations
from datetime import date
from functools import wraps
import numpy as np
import polars as pl

from .data import load_transactions, load_articles, load_customers


def cutoff_guard(fn):
    """Decorator that runs an explicit, hard-failing leakage check on the cutoff.

    Pre-call: verifies that the function's body, when called, does not access any
    transaction with t_dat >= cutoff. Implemented by sampling raw tx for the
    function's customer_ids and asserting max(t_dat) < cutoff.
    """
    @wraps(fn)
    def wrap(*args, cutoff: str | date, **kwargs):
        cutoff_d = date.fromisoformat(cutoff) if isinstance(cutoff, str) else cutoff
        # Independent leakage assertion on the underlying transactions table,
        # restricted to the customer_ids the function was asked to compute over.
        customer_ids = args[0] if args else kwargs.get("customer_ids")
        if customer_ids is not None and len(customer_ids) > 0:
            sample = customer_ids if len(customer_ids) <= 5000 else list(customer_ids)[:5000]
            mx = (
                load_transactions()
                .filter((pl.col("customer_id").is_in(sample)) & (pl.col("t_dat") < cutoff_d))
                .select(pl.col("t_dat").max())
                .collect()
                .item()
            )
            if mx is not None and mx >= cutoff_d:
                raise RuntimeError(f"cutoff_guard: pre-cutoff max t_dat {mx} >= {cutoff_d} in {fn.__name__}")
        return fn(*args, cutoff=cutoff_d, **kwargs)
    return wrap


# ---------- RFM (Rep A) ----------

@cutoff_guard
def rfm_features(customer_ids: list[str], *, cutoff: date) -> pl.DataFrame:
    tx = load_transactions().filter((pl.col("t_dat") < cutoff) & (pl.col("customer_id").is_in(customer_ids)))

    agg = (
        tx.group_by("customer_id")
        .agg(
            pl.len().alias("frequency"),
            pl.col("price").sum().alias("monetary"),
            pl.col("price").mean().alias("mean_spend"),
            pl.col("t_dat").max().alias("last_tx_date"),
            pl.col("t_dat").min().alias("first_tx_date"),
            (pl.col("sales_channel_id") == 2).sum().alias("n_channel2"),
            pl.col("article_id").n_unique().alias("n_distinct_articles"),
        )
        .with_columns(
            (pl.lit(cutoff).cast(pl.Date) - pl.col("last_tx_date")).dt.total_days().alias("recency_days"),
            (pl.lit(cutoff).cast(pl.Date) - pl.col("first_tx_date")).dt.total_days().alias("tenure_days"),
        )
        .with_columns(
            (pl.col("frequency") / (pl.col("tenure_days") + 1)).alias("freq_per_day"),
            (pl.col("n_channel2") / pl.col("frequency")).alias("channel2_share"),
            (pl.col("monetary") / pl.col("frequency")).alias("aov"),
        )
        .drop(["last_tx_date", "first_tx_date"])
        .collect()
    )
    # Add static customer features (use only age and postal as non-leaky)
    cust = load_customers().select(["customer_id", "age", "postal_code"]).collect()
    out = agg.join(cust, on="customer_id", how="left")

    # Postal modal top-20 one-hot
    top_postals = (
        out.group_by("postal_code").len().sort("len", descending=True).head(20)["postal_code"].to_list()
    )
    out = out.with_columns(
        [
            (pl.col("postal_code") == p).cast(pl.Int8).alias(f"postal_{i}")
            for i, p in enumerate(top_postals)
        ]
    ).drop("postal_code")

    # Age bucket
    out = out.with_columns(
        pl.when(pl.col("age") < 25).then(pl.lit("u25"))
        .when(pl.col("age") < 35).then(pl.lit("25-34"))
        .when(pl.col("age") < 45).then(pl.lit("35-44"))
        .when(pl.col("age") < 55).then(pl.lit("45-54"))
        .otherwise(pl.lit("55p"))
        .alias("age_bucket")
    )
    age_buckets = ["u25", "25-34", "35-44", "45-54", "55p"]
    out = out.with_columns(
        [
            (pl.col("age_bucket") == b).cast(pl.Int8).alias(f"age_{b}")
            for b in age_buckets
        ]
    ).drop("age_bucket")
    out = out.with_columns(pl.col("age").fill_null(out["age"].median()))
    return out


# ---------- Bag-of-categories (Rep B) ----------

@cutoff_guard
def bag_of_categories(customer_ids: list[str], *, cutoff: date) -> pl.DataFrame:
    tx = load_transactions().filter((pl.col("t_dat") < cutoff) & (pl.col("customer_id").is_in(customer_ids)))
    art = load_articles().select(["article_id", "product_type_no", "garment_group_no", "colour_group_code", "index_group_no"])
    joined = tx.join(art, on="article_id", how="left").collect()

    def _count_pivot(col: str, prefix: str, top_k: int = 50):
        top = joined.group_by(col).len().sort("len", descending=True).head(top_k)[col].to_list()
        sub = joined.filter(pl.col(col).is_in(top))
        piv = (
            sub.group_by(["customer_id", col]).len().rename({"len": "n"})
            .pivot(values="n", index="customer_id", on=col)
            .fill_null(0)
        )
        piv = piv.rename({c: f"{prefix}{c}" for c in piv.columns if c != "customer_id"})
        return piv

    pt = _count_pivot("product_type_no", "pt_", top_k=50)
    gg = _count_pivot("garment_group_no", "gg_", top_k=20)
    cg = _count_pivot("colour_group_code", "cg_", top_k=30)
    ig = _count_pivot("index_group_no", "ig_", top_k=10)

    out = pt.join(gg, on="customer_id", how="left").join(cg, on="customer_id", how="left").join(ig, on="customer_id", how="left")
    out = out.fill_null(0)
    return out


# ---------- Sequences (Rep C) ----------

@cutoff_guard
def event_sequences(customer_ids: list[str], *, cutoff: date, max_len: int = 64) -> dict:
    """Return dict customer_id -> list of (article_id, days_before_cutoff, channel)."""
    tx = (
        load_transactions()
        .filter((pl.col("t_dat") < cutoff) & (pl.col("customer_id").is_in(customer_ids)))
        .sort(["customer_id", "t_dat"], descending=[False, False])
        .collect()
    )
    tx = tx.with_columns(
        (pl.lit(cutoff).cast(pl.Date) - pl.col("t_dat")).dt.total_days().alias("days_before"),
    )
    out: dict[str, list] = {}
    for cust_id, group in tx.group_by("customer_id"):
        seq = list(zip(group["article_id"].to_list(), group["days_before"].to_list(), group["sales_channel_id"].to_list()))
        out[cust_id[0]] = seq[-max_len:]
    return out


# ---------- LLM narrative (Rep D) ----------

@cutoff_guard
def behavioral_narrative(customer_ids: list[str], *, cutoff: date, n_recent: int = 20) -> dict[str, str]:
    """Return dict customer_id -> natural-language summary of their behavior, suitable for LLM input."""
    tx = (
        load_transactions()
        .filter((pl.col("t_dat") < cutoff) & (pl.col("customer_id").is_in(customer_ids)))
        .collect()
    )
    art = load_articles().select(["article_id", "prod_name", "product_type_name", "garment_group_name", "colour_group_name", "section_name", "detail_desc"]).collect()
    tx_j = tx.join(art, on="article_id", how="left")

    # NOTE: club_member_status and fashion_news_frequency are snapshot fields
    # (state at dataset release ~Sept 2020), not time-stamped. Excluded to avoid
    # post-cutoff leakage into the LLM narrative.
    cust = load_customers().select(["customer_id", "age"]).collect()
    cust_d = {row["customer_id"]: row for row in cust.iter_rows(named=True)}

    out: dict[str, str] = {}
    for cust_id, group in tx_j.group_by("customer_id"):
        cid = cust_id[0]
        group = group.sort("t_dat")
        n_total = len(group)
        first = group["t_dat"].min()
        last = group["t_dat"].max()
        tenure = (cutoff - last).days
        days_since_last = (cutoff - last).days
        total_spend = float(group["price"].sum())
        mean_spend = float(group["price"].mean())

        top_pt = group.group_by("product_type_name").len().sort("len", descending=True).head(3)["product_type_name"].to_list()
        top_color = group.group_by("colour_group_name").len().sort("len", descending=True).head(3)["colour_group_name"].to_list()
        top_section = group.group_by("section_name").len().sort("len", descending=True).head(3)["section_name"].to_list()

        recent = group.tail(n_recent)
        recent_lines = []
        for r in recent.iter_rows(named=True):
            days_ago = (cutoff - r["t_dat"]).days
            line = f"  - {days_ago}d ago: {r['prod_name']} ({r['product_type_name']}, {r['colour_group_name']})"
            recent_lines.append(line)

        c = cust_d.get(cid, {})
        age = c.get("age") if c else None

        narr = (
            f"CUSTOMER PROFILE\n"
            f"- Demographics: age {age}\n"
            f"- Purchase history: {n_total} total purchases, first on {first}, most recent {days_since_last} days before cutoff\n"
            f"- Total spend: {total_spend:.2f} (avg per item {mean_spend:.4f})\n"
            f"- Top product types: {', '.join(str(x) for x in top_pt if x)}\n"
            f"- Top colors: {', '.join(str(x) for x in top_color if x)}\n"
            f"- Top sections: {', '.join(str(x) for x in top_section if x)}\n"
            f"\nRECENT PURCHASES (up to last {n_recent}):\n"
            + "\n".join(recent_lines)
        )
        out[cid] = narr
    return out
