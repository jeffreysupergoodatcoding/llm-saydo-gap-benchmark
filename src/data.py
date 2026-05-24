"""Data loading and Parquet conversion."""

from __future__ import annotations
from pathlib import Path
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"


def csv_to_parquet():
    """Convert raw CSVs to Parquet for fast re-reads."""
    PROCESSED.mkdir(parents=True, exist_ok=True)

    if not (PROCESSED / "articles.parquet").exists():
        articles = pl.read_csv(RAW / "articles.csv")
        articles = articles.with_columns(pl.col("article_id").cast(pl.Int64))
        articles.write_parquet(PROCESSED / "articles.parquet")

    if not (PROCESSED / "customers.parquet").exists():
        customers = pl.read_csv(RAW / "customers.csv")
        customers.write_parquet(PROCESSED / "customers.parquet")

    if not (PROCESSED / "transactions.parquet").exists():
        # Stream the 3.5 GB CSV
        tx = pl.read_csv(
            RAW / "transactions_train.csv",
            schema_overrides={
                "t_dat": pl.Utf8,
                "customer_id": pl.Utf8,
                "article_id": pl.Int64,
                "price": pl.Float64,
                "sales_channel_id": pl.Int64,
            },
        ).with_columns(pl.col("t_dat").str.to_date("%Y-%m-%d"))
        tx.write_parquet(PROCESSED / "transactions.parquet", compression="zstd")

    return {
        "articles": PROCESSED / "articles.parquet",
        "customers": PROCESSED / "customers.parquet",
        "transactions": PROCESSED / "transactions.parquet",
    }


def load_transactions() -> pl.LazyFrame:
    return pl.scan_parquet(PROCESSED / "transactions.parquet")


def load_articles() -> pl.LazyFrame:
    return pl.scan_parquet(PROCESSED / "articles.parquet")


def load_customers() -> pl.LazyFrame:
    return pl.scan_parquet(PROCESSED / "customers.parquet")
