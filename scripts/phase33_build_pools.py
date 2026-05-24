"""Phase 33: build M3 k-NN neighbour pool and M8 RAG history pool for v3 sandbox.

For each core-1000 customer, pick 5 RFM-nearest customers from val.parquet that
are NOT in the core-1000 itself. Persist as JSON dicts.

For M8 RAG, we additionally need the candidates' realized labels (post-cutoff
purchase) — which val.parquet already has (label_window is pre-test).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import polars as pl


ROOT = Path(__file__).resolve().parents[1]


def main():
    core = pl.read_parquet(ROOT / "results" / "phase31_core1000_v3.parquet")
    val = pl.read_parquet(ROOT / "data" / "splits" / "val.parquet")
    train = pl.read_parquet(ROOT / "data" / "splits" / "train.parquet")

    # We want examples with realized 30-day outcomes. val has them.
    # Sample 30k val customers and use them as the prior-customer pool.
    rng = np.random.default_rng(2026)
    n_pool = min(30000, len(val))
    pool_idx = rng.choice(len(val), size=n_pool, replace=False)
    pool = val[pool_idx.tolist()]

    # For each pool customer we need a "top_section" — pull from train transactions for that customer.
    # Build a lightweight summary: orders, recency, label
    print(f"Pool size: {len(pool)}")

    # Top section per pool customer (from train)
    from src.data import load_transactions, load_articles
    from datetime import date
    from src import T_TRAIN_CUTOFF
    cutoff = date.fromisoformat(T_TRAIN_CUTOFF)
    pool_ids = set(pool["customer_id"].to_list())
    tx = (
        load_transactions()
        .filter((pl.col("t_dat") < cutoff) & (pl.col("customer_id").is_in(list(pool_ids))))
        .collect()
    )
    art = load_articles().select(["article_id", "section_name"]).collect()
    tx_j = tx.join(art, on="article_id", how="left")
    top_sec = (
        tx_j.group_by(["customer_id", "section_name"]).len()
        .sort("len", descending=True)
        .group_by("customer_id").agg(pl.first("section_name").alias("top_section"))
    )
    pool = pool.join(top_sec, on="customer_id", how="left")

    pool_lite = pool.select([
        "customer_id", "n_tx_pre_cutoff", "label", "activity_bucket", "top_section",
        "n_tx_label_window",
    ]).rename({
        "n_tx_pre_cutoff": "total_orders",
    }).with_columns(
        pl.lit(30).alias("recency_days"),  # placeholder; pool customers' recency is not needed
    )

    # Add recency proxy from last_tx_date_pre
    if "last_tx_date_pre" in pool.columns:
        pool_lite = pool_lite.with_columns(
            ((cutoff - pool["last_tx_date_pre"]).dt.total_days()).alias("recency_days")
        )

    pool_dicts = pool_lite.to_dicts()
    # filter out null top_section
    pool_dicts = [p for p in pool_dicts if p.get("top_section")]
    print(f"Pool with valid top_section: {len(pool_dicts)}")

    # For each core customer, find 5 nearest by RFM
    core_ids = core["customer_id"].to_list()
    core_orders = core["n_tx_pre_cutoff"].to_list()

    neighbours: dict[str, list[dict]] = {}
    pool_arr = np.array([[p["total_orders"], p.get("recency_days", 30)] for p in pool_dicts], dtype=float)
    for i, cid in enumerate(core_ids):
        target = np.array([core_orders[i], 30.0])  # we don't have recency in core, use 30
        d = np.abs(pool_arr[:, 0] - target[0]) + 0.05 * np.abs(pool_arr[:, 1] - target[1])
        idxs = np.argsort(d)[:5]
        neighbours[cid] = [pool_dicts[k] for k in idxs.tolist()]

    out_nb = ROOT / "results" / "phase33_neighbours.json"
    out_nb.write_text(json.dumps(neighbours, indent=2))
    print(f"Wrote {out_nb}")

    # Full history pool for M8 RAG
    out_pool = ROOT / "results" / "phase33_history_pool.json"
    out_pool.write_text(json.dumps(pool_dicts, indent=2))
    print(f"Wrote {out_pool} ({len(pool_dicts)} examples)")


if __name__ == "__main__":
    main()
