"""Temporal customer split builder.

Two cutoffs, customer-disjoint. Stratified by activity bucket.
"""

from __future__ import annotations
from datetime import date, timedelta
import polars as pl
import numpy as np

from . import SEED, T_TRAIN_CUTOFF, T_TEST_CUTOFF, LABEL_WINDOW_DAYS, SAMPLE_N, ACTIVITY_BUCKETS
from .data import SPLITS, load_transactions, load_customers


def _activity_bucket_expr(col: str = "n_tx_pre_cutoff") -> pl.Expr:
    e = pl.when(pl.col(col) <= 1).then(pl.lit("1"))
    for lo, hi in ACTIVITY_BUCKETS[1:]:
        label = f"{lo}-{hi}" if hi < 10**8 else f"{lo}+"
        e = e.when(pl.col(col).is_between(lo, hi)).then(pl.lit(label))
    return e.otherwise(pl.lit("unknown"))


def build_split_for_cutoff(cutoff: str, label_window_days: int = LABEL_WINDOW_DAYS):
    """For a given T_cutoff date, compute per-customer:
    - n_tx_pre_cutoff (any txs strictly before cutoff)
    - label = 1 if ≥1 tx in [cutoff, cutoff + label_window_days)
    - activity_bucket
    Returns a polars dataframe.
    """
    cutoff_d = date.fromisoformat(cutoff)
    label_end = cutoff_d + timedelta(days=label_window_days)

    tx = load_transactions()

    pre = (
        tx.filter(pl.col("t_dat") < cutoff_d)
        .group_by("customer_id")
        .agg(
            pl.len().alias("n_tx_pre_cutoff"),
            pl.col("t_dat").max().alias("last_tx_date_pre"),
            pl.col("t_dat").min().alias("first_tx_date_pre"),
        )
    )
    post = (
        tx.filter((pl.col("t_dat") >= cutoff_d) & (pl.col("t_dat") < label_end))
        .group_by("customer_id")
        .agg(pl.len().alias("n_tx_label_window"))
    )
    df = (
        pre.join(post, on="customer_id", how="left")
        .with_columns(pl.col("n_tx_label_window").fill_null(0))
        .with_columns(pl.col("n_tx_label_window").gt(0).cast(pl.Int8).alias("label"))
        .filter(pl.col("n_tx_pre_cutoff") >= 1)  # condition on having pre-cutoff history
        .collect()
    )
    df = df.with_columns(_activity_bucket_expr().alias("activity_bucket"))
    return df


def stratified_subsample(df: pl.DataFrame, n: int, seed: int = SEED) -> pl.DataFrame:
    """Sample n rows stratified by activity bucket."""
    rng = np.random.default_rng(seed)
    buckets = df["activity_bucket"].unique().to_list()
    target_per_bucket = max(1, n // len(buckets))
    parts = []
    for b in buckets:
        sub = df.filter(pl.col("activity_bucket") == b)
        k = min(target_per_bucket, len(sub))
        idx = rng.choice(len(sub), size=k, replace=False)
        parts.append(sub[idx.tolist()])
    out = pl.concat(parts)
    if len(out) > n:
        idx = rng.choice(len(out), size=n, replace=False)
        out = out[idx.tolist()]
    return out


def build_train_test_splits():
    """Full split builder. Writes parquets to data/splits/."""
    SPLITS.mkdir(parents=True, exist_ok=True)

    train_df = build_split_for_cutoff(T_TRAIN_CUTOFF)
    test_df = build_split_for_cutoff(T_TEST_CUTOFF)

    # Stratified subsample of the test pool
    test_sample = stratified_subsample(test_df, SAMPLE_N)
    # Train pool: use train_df, partition 80/10/10 train/val/test_internal
    train_sample = stratified_subsample(train_df, SAMPLE_N)

    rng = np.random.default_rng(SEED)
    n = len(train_sample)
    perm = rng.permutation(n)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]

    train_part = train_sample[train_idx.tolist()].with_columns(pl.lit("train").alias("split"))
    val_part = train_sample[val_idx.tolist()].with_columns(pl.lit("val").alias("split"))

    # Ensure test customers are disjoint from train+val
    train_val_ids = set(train_part["customer_id"].to_list()) | set(val_part["customer_id"].to_list())
    test_part = test_sample.filter(~pl.col("customer_id").is_in(list(train_val_ids))).with_columns(
        pl.lit("test").alias("split")
    )

    train_part.write_parquet(SPLITS / "train.parquet")
    val_part.write_parquet(SPLITS / "val.parquet")
    test_part.write_parquet(SPLITS / "test.parquet")

    return {
        "train": train_part,
        "val": val_part,
        "test": test_part,
        "train_full": train_df,
        "test_full": test_df,
    }


def load_split(name: str) -> pl.DataFrame:
    return pl.read_parquet(SPLITS / f"{name}.parquet")
