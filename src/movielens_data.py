"""MovieLens 25M data loader + behavioral trace adapter.

Adapts the H&M pipeline (purchases → ratings≥4 in 30d window).
Same activity-bucket strata, same temporal-cutoff protocol.
"""

from __future__ import annotations
from datetime import date, datetime, timezone
from pathlib import Path
import polars as pl
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ML_DIR = ROOT / "data" / "movielens" / "ml-25m"
PROCESSED = ROOT / "data" / "movielens" / "processed"
SPLITS = ROOT / "data" / "movielens" / "splits"

# Use 2018-07-22 / 2018-08-22 cutoffs to mirror the H&M protocol.
T_TRAIN_CUTOFF = "2018-07-22"
T_TEST_CUTOFF = "2018-08-22"
LABEL_WINDOW_DAYS = 30
RATING_THRESHOLD = 4.0  # rating >= 4 = "positive engagement" (analog to H&M purchase)

ACTIVITY_BUCKETS = [(1, 1), (2, 5), (6, 20), (21, 100), (101, 10**9)]
SEED = 42


def csv_to_parquet():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    if not (PROCESSED / "ratings.parquet").exists():
        r = pl.read_csv(ML_DIR / "ratings.csv")
        # Convert UNIX timestamp → date
        r = r.with_columns(
            pl.col("timestamp").map_elements(
                lambda t: datetime.fromtimestamp(int(t), tz=timezone.utc).date(),
                return_dtype=pl.Date
            ).alias("t_dat")
        ).drop("timestamp")
        r.write_parquet(PROCESSED / "ratings.parquet")
    if not (PROCESSED / "movies.parquet").exists():
        m = pl.read_csv(ML_DIR / "movies.csv")
        m.write_parquet(PROCESSED / "movies.parquet")
    return {"ratings": PROCESSED / "ratings.parquet", "movies": PROCESSED / "movies.parquet"}


def load_ratings() -> pl.LazyFrame:
    return pl.scan_parquet(PROCESSED / "ratings.parquet")


def load_movies() -> pl.LazyFrame:
    return pl.scan_parquet(PROCESSED / "movies.parquet")


def _activity_bucket_expr(col: str = "n_pre") -> pl.Expr:
    e = pl.when(pl.col(col) <= 1).then(pl.lit("1"))
    for lo, hi in ACTIVITY_BUCKETS[1:]:
        label = f"{lo}-{hi}" if hi < 10**8 else f"{lo}+"
        e = e.when(pl.col(col).is_between(lo, hi)).then(pl.lit(label))
    return e.otherwise(pl.lit("unknown"))


def build_split_for_cutoff(cutoff: str):
    from datetime import timedelta
    cutoff_d = date.fromisoformat(cutoff)
    label_end = cutoff_d + timedelta(days=LABEL_WINDOW_DAYS)
    r = load_ratings()
    # Pre: any ratings (engagement); used for activity bucket.
    pre = (
        r.filter(pl.col("t_dat") < cutoff_d)
        .group_by("userId")
        .agg(
            pl.len().alias("n_pre"),
            pl.col("t_dat").max().alias("last_pre"),
            pl.col("t_dat").min().alias("first_pre"),
        )
    )
    # Post: ANY rating in label window = "engagement" (broader than ≥4 to match H&M's "any purchase").
    post = (
        r.filter((pl.col("t_dat") >= cutoff_d)
                  & (pl.col("t_dat") < label_end))
        .group_by("userId").agg(pl.len().alias("n_post"))
    )
    df = (
        pre.join(post, on="userId", how="left")
        .with_columns(pl.col("n_post").fill_null(0))
        .with_columns(pl.col("n_post").gt(0).cast(pl.Int8).alias("label"))
        .filter(pl.col("n_pre") >= 1)
        .collect()
    )
    df = df.with_columns(_activity_bucket_expr().alias("activity_bucket"))
    return df


def stratified_subsample(df: pl.DataFrame, n: int, seed: int = SEED) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    buckets = df["activity_bucket"].unique().to_list()
    per = max(1, n // len(buckets))
    parts = []
    for b in buckets:
        sub = df.filter(pl.col("activity_bucket") == b)
        k = min(per, len(sub))
        idx = rng.choice(len(sub), size=k, replace=False)
        parts.append(sub[idx.tolist()])
    out = pl.concat(parts)
    if len(out) > n:
        idx = rng.choice(len(out), size=n, replace=False)
        out = out[idx.tolist()]
    return out


def build_movielens_splits(n_per_bucket: int = 100):
    SPLITS.mkdir(parents=True, exist_ok=True)
    train_pool = build_split_for_cutoff(T_TRAIN_CUTOFF)
    test_pool = build_split_for_cutoff(T_TEST_CUTOFF)
    train_sample = stratified_subsample(train_pool, n_per_bucket * 5, seed=SEED)
    test_sample = stratified_subsample(test_pool, n_per_bucket * 5, seed=SEED + 1)
    test_sample = test_sample.filter(~pl.col("userId").is_in(train_sample["userId"].to_list()))
    train_sample.write_parquet(SPLITS / "train.parquet")
    test_sample.write_parquet(SPLITS / "test.parquet")
    return {"train_pool_full": len(train_pool), "test_pool_full": len(test_pool),
            "train_n": len(train_sample), "test_n": len(test_sample),
            "train_label_rate": float(train_sample["label"].mean()),
            "test_label_rate": float(test_sample["label"].mean()),
            "test_pool_label_rate": float(test_pool["label"].mean())}


def behavioral_trace_ml(user_ids: list[int], cutoff: str, n_recent: int = 15) -> dict[int, dict]:
    """MovieLens-adapted trace dict."""
    cutoff_d = date.fromisoformat(cutoff)
    r = (
        load_ratings()
        .filter((pl.col("t_dat") < cutoff_d) & (pl.col("userId").is_in(user_ids)))
        .collect()
    )
    m = load_movies().collect()
    rm = r.join(m, on="movieId", how="left")

    out: dict[int, dict] = {}
    for uid, group in rm.group_by("userId"):
        uid = uid[0] if isinstance(uid, tuple) else uid
        group = group.sort("t_dat")
        n_total = len(group)
        first = group["t_dat"].min()
        last = group["t_dat"].max()
        recency_days = (cutoff_d - last).days
        tenure_days = (cutoff_d - first).days
        mean_rating = float(group["rating"].mean())
        pct_4_plus = float((group["rating"] >= 4.0).mean())

        # Top genres (split by |)
        all_genres = "|".join([str(g) for g in group["genres"].to_list() if g])
        genre_counts = {}
        for g in all_genres.split("|"):
            if g and g != "(no genres listed)":
                genre_counts[g] = genre_counts.get(g, 0) + 1
        top_genres = sorted(genre_counts.items(), key=lambda x: -x[1])[:3]
        top_genres_str = ", ".join([g for g, _ in top_genres])
        n_distinct_movies = int(group["movieId"].n_unique())
        n_genres_seen = len(genre_counts)

        recent = group.tail(n_recent)
        recent_list = [
            {"days_ago": (cutoff_d - r["t_dat"]).days, "title": r["title"],
             "rating": r["rating"], "genres": r["genres"]}
            for r in recent.iter_rows(named=True)
        ]

        # Personality / flags
        if pct_4_plus > 0.8:
            personality = "positive-rater"
        elif pct_4_plus < 0.3:
            personality = "critical-rater"
        else:
            personality = "balanced-rater"

        out[uid] = {
            "identity": {"user_id": int(uid)},
            "purchase_stats": {  # keeping the H&M-key name for pipeline compatibility
                "total_orders": n_total,
                "total_spend": float(group["rating"].sum()),  # sum of ratings as a "spend" proxy
                "recency_days": int(recency_days),
                "tenure_days": int(tenure_days),
                "aov": mean_rating,
                "channel2_share": pct_4_plus,
                "distinct_articles": n_distinct_movies,
            },
            "product_summary": {
                "top_section": top_genres[0][0] if top_genres else None,
                "top_garment_group": top_genres[1][0] if len(top_genres) > 1 else None,
                "top_color": None,
                "top_product_type": None,
                "sections_seen": n_genres_seen,
                "garment_groups_seen": n_genres_seen,
                "colors_seen": 0,
                "top_genres": top_genres_str,
            },
            "timeline": {
                "first_tx_date": str(first), "last_tx_date": str(last),
                "avg_inter_purchase_days": (tenure_days / max(n_total - 1, 1)),
            },
            "recent_purchases": [
                {"days_ago": rr["days_ago"], "prod_name": rr["title"],
                 "product_type": rr["genres"], "garment_group": rr["genres"],
                 "color": "—", "section": rr["genres"], "price": float(rr["rating"]),
                 "channel": "rated"} for rr in recent_list
            ],
            "personality": personality,
            "derived_flags": {
                "is_new_to_brand": n_total <= 1,
                "is_lapsed": recency_days > 180,
                "is_diverse_shopper": n_genres_seen >= 5 and n_distinct_movies > 20,
            },
        }
    return out
