"""Phase 3: Data-volume scaling curves.

Two sweeps:
  - N_events: vary the most-recent-N events per customer used to compute features.
  - N_days: vary the look-back window in days before cutoff.

For each point, train RFM logistic and LightGBM on RFM+bag features. Bootstrap CIs.
"""

from __future__ import annotations
import json
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from src import SEED, T_TRAIN_CUTOFF, T_TEST_CUTOFF
from src.data import load_transactions, load_customers, load_articles
from src.splits import load_split
from src.eval import pr_auc, bootstrap_ci

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def limited_rfm(customer_ids: list[str], cutoff_str: str, *, n_events: int | None = None, n_days: int | None = None) -> pl.DataFrame:
    """Build RFM features using only first/last N events or last N days of history."""
    cutoff = date.fromisoformat(cutoff_str)
    tx = load_transactions().filter((pl.col("t_dat") < cutoff) & (pl.col("customer_id").is_in(customer_ids)))
    if n_days is not None:
        earliest = cutoff - timedelta(days=n_days)
        tx = tx.filter(pl.col("t_dat") >= earliest)
    if n_events is not None:
        # Take the most recent n_events per customer
        tx = (
            tx.sort(["customer_id", "t_dat"], descending=[False, True])
            .group_by("customer_id")
            .head(n_events)
        )
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
    cust = load_customers().select(["customer_id", "age"]).collect()
    out = agg.join(cust, on="customer_id", how="left")
    out = out.with_columns(pl.col("age").fill_null(out["age"].median()))
    return out


def align_features(split_df: pl.DataFrame, feat_df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    joined = split_df.join(feat_df, on="customer_id", how="left").fill_null(0)
    feature_cols = [c for c in feat_df.columns if c != "customer_id"]
    X = joined.select(feature_cols).to_numpy().astype(np.float32)
    y = joined["label"].to_numpy()
    return X, y


def train_eval_logistic(X_train, y_train, X_test, y_test):
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
    ])
    pipe.fit(X_train, y_train)
    p = pipe.predict_proba(X_test)[:, 1]
    m = bootstrap_ci({"y_true": y_test, "y_score": p}, lambda y_true, y_score: pr_auc(y_true, y_score), B=300, seed=SEED)
    return m, p


def train_eval_lgbm(X_train, y_train, X_test, y_test):
    model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, num_leaves=63,
        min_data_in_leaf=20, feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    model.fit(X_train, y_train)
    p = model.predict_proba(X_test)[:, 1]
    m = bootstrap_ci({"y_true": y_test, "y_score": p}, lambda y_true, y_score: pr_auc(y_true, y_score), B=300, seed=SEED)
    return m, p


def main():
    train = load_split("train")
    test = load_split("test")
    train_ids = train["customer_id"].to_list()
    test_ids = test["customer_id"].to_list()

    N_EVENTS = [1, 2, 4, 8, 16, 32, 64, None]
    N_DAYS = [7, 14, 30, 60, 90, 180, 365, None]

    results = {"by_n_events": [], "by_n_days": []}

    # Sweep N_events (with full N_days)
    print("\n=== N_events sweep (full N_days) ===")
    for ne in N_EVENTS:
        print(f"  N_events = {ne}", flush=True)
        f_train = limited_rfm(train_ids, T_TRAIN_CUTOFF, n_events=ne, n_days=None)
        f_test = limited_rfm(test_ids, T_TEST_CUTOFF, n_events=ne, n_days=None)
        X_tr, y_tr = align_features(train, f_train)
        X_te, y_te = align_features(test, f_test)
        m_lr, _ = train_eval_logistic(X_tr, y_tr, X_te, y_te)
        m_gb, _ = train_eval_lgbm(X_tr, y_tr, X_te, y_te)
        results["by_n_events"].append({
            "n_events": ne if ne is not None else -1,
            "logistic": m_lr,
            "lgbm": m_gb,
            "n_features": X_tr.shape[1],
        })
        print(f"    logistic PR-AUC = {m_lr['point']:.4f}, lgbm = {m_gb['point']:.4f}", flush=True)

    # Sweep N_days (with full N_events)
    print("\n=== N_days sweep (full N_events) ===")
    for nd in N_DAYS:
        print(f"  N_days = {nd}", flush=True)
        f_train = limited_rfm(train_ids, T_TRAIN_CUTOFF, n_events=None, n_days=nd)
        f_test = limited_rfm(test_ids, T_TEST_CUTOFF, n_events=None, n_days=nd)
        X_tr, y_tr = align_features(train, f_train)
        X_te, y_te = align_features(test, f_test)
        m_lr, _ = train_eval_logistic(X_tr, y_tr, X_te, y_te)
        m_gb, _ = train_eval_lgbm(X_tr, y_tr, X_te, y_te)
        results["by_n_days"].append({
            "n_days": nd if nd is not None else -1,
            "logistic": m_lr,
            "lgbm": m_gb,
            "n_features": X_tr.shape[1],
        })
        print(f"    logistic PR-AUC = {m_lr['point']:.4f}, lgbm = {m_gb['point']:.4f}", flush=True)

    (RESULTS / "phase3_scaling.json").write_text(json.dumps(results, indent=2, default=str))

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    def _x(rows, key):
        return [r[key] if r[key] != -1 else 100 for r in rows]

    rows = results["by_n_events"]
    x = _x(rows, "n_events")
    lr_pts = [r["logistic"]["point"] for r in rows]
    lr_lo = [r["logistic"]["lo"] for r in rows]
    lr_hi = [r["logistic"]["hi"] for r in rows]
    gb_pts = [r["lgbm"]["point"] for r in rows]
    gb_lo = [r["lgbm"]["lo"] for r in rows]
    gb_hi = [r["lgbm"]["hi"] for r in rows]
    axes[0].plot(x, lr_pts, "o-", label="RFM logistic")
    axes[0].fill_between(x, lr_lo, lr_hi, alpha=0.2)
    axes[0].plot(x, gb_pts, "s-", label="RFM LightGBM")
    axes[0].fill_between(x, gb_lo, gb_hi, alpha=0.2)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("N events (most recent per customer; 100 = all)")
    axes[0].set_ylabel("PR-AUC")
    axes[0].set_title("Scaling: events per customer")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    rows = results["by_n_days"]
    x = _x(rows, "n_days")
    lr_pts = [r["logistic"]["point"] for r in rows]
    lr_lo = [r["logistic"]["lo"] for r in rows]
    lr_hi = [r["logistic"]["hi"] for r in rows]
    gb_pts = [r["lgbm"]["point"] for r in rows]
    gb_lo = [r["lgbm"]["lo"] for r in rows]
    gb_hi = [r["lgbm"]["hi"] for r in rows]
    axes[1].plot(x, lr_pts, "o-", label="RFM logistic")
    axes[1].fill_between(x, lr_lo, lr_hi, alpha=0.2)
    axes[1].plot(x, gb_pts, "s-", label="RFM LightGBM")
    axes[1].fill_between(x, gb_lo, gb_hi, alpha=0.2)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("N days of history (100=all ~700d)")
    axes[1].set_ylabel("PR-AUC")
    axes[1].set_title("Scaling: history window")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS / "phase3_scaling.png", dpi=130)
    plt.close()

    # Inflection: smallest N where adding 50% more buys < 1 PR-AUC point
    def find_inflection(rows, key):
        pts = [(r[key], r["lgbm"]["point"]) for r in rows if r[key] != -1]
        pts.sort()
        for i in range(len(pts) - 1):
            n0, p0 = pts[i]
            n1, p1 = pts[i + 1]
            if n1 >= 1.5 * n0 and (p1 - p0) < 0.01:
                return {"n": n0, "pr_auc": p0, "next_n": n1, "delta": p1 - p0}
        return None

    infl_e = find_inflection(results["by_n_events"], "n_events")
    infl_d = find_inflection(results["by_n_days"], "n_days")
    results["inflection"] = {"events": infl_e, "days": infl_d}
    (RESULTS / "phase3_scaling.json").write_text(json.dumps(results, indent=2, default=str))
    print("\nInflection (events):", infl_e)
    print("Inflection (days):", infl_d)
    print("[phase3] Done.")


if __name__ == "__main__":
    main()
