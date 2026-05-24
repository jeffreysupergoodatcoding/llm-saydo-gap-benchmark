"""Phase 2: Baselines.

- Majority class (always-no)
- Popularity + recency heuristic: score = exp(-days_since_last/30) * log(1+n_tx_pre)
- RFM logistic regression
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import polars as pl

from src import SEED, T_TRAIN_CUTOFF, T_TEST_CUTOFF
from src.splits import load_split
from src.features import rfm_features
from src.models.tabular import train_logistic, predict_proba
from src.eval import all_metrics, paired_bootstrap_diff, pr_auc

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def main():
    train = load_split("train")
    test = load_split("test")

    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()

    # 1. Majority class — always predict prior probability (label_rate of train)
    rate = float(y_train.mean())
    s_majority = np.full(len(y_test), rate)

    # 2. Popularity + recency heuristic
    # need recency for the test customers as of T_TEST_CUTOFF
    from datetime import date
    cutoff = date.fromisoformat(T_TEST_CUTOFF)
    cutoff_np = np.datetime64(cutoff, "D")
    last_arr = test["last_tx_date_pre"].to_numpy().astype("datetime64[D]")
    test_recency = (cutoff_np - last_arr).astype(int)
    n_tx_pre = test["n_tx_pre_cutoff"].to_numpy()
    s_recency = np.exp(-test_recency / 30.0) * np.log1p(n_tx_pre)
    # rescale to [0,1]
    s_recency = (s_recency - s_recency.min()) / max(s_recency.max() - s_recency.min(), 1e-9)

    # 3. RFM logistic
    print("[phase2] Computing RFM features for train...")
    train_ids = train["customer_id"].to_list()
    rfm_train = rfm_features(train_ids, cutoff=T_TRAIN_CUTOFF)
    print("[phase2] Computing RFM features for test...")
    test_ids = test["customer_id"].to_list()
    rfm_test = rfm_features(test_ids, cutoff=T_TEST_CUTOFF)

    # Align with train/test labels
    train_full = train.join(rfm_train, on="customer_id", how="left")
    test_full = test.join(rfm_test, on="customer_id", how="left")

    feature_cols = [c for c in rfm_train.columns if c != "customer_id"]
    X_train = train_full.select(feature_cols).fill_null(0).to_numpy().astype(np.float32)
    X_test = test_full.select(feature_cols).fill_null(0).to_numpy().astype(np.float32)
    y_train_align = train_full["label"].to_numpy()
    y_test_align = test_full["label"].to_numpy()

    print(f"[phase2] X_train {X_train.shape}, X_test {X_test.shape}")

    model = train_logistic(X_train, y_train_align, seed=SEED)
    s_logistic = predict_proba(model, X_test)

    # Metrics
    results = {}
    for name, score, y_use in [
        ("majority", s_majority, y_test),
        ("popularity_recency", s_recency, y_test),
        ("rfm_logistic", s_logistic, y_test_align),
    ]:
        m = all_metrics(y_use, score, B=500, seed=SEED)
        results[name] = m
        print(f"[phase2] {name}: PR-AUC = {m['pr_auc']['point']:.4f} [{m['pr_auc']['lo']:.4f}, {m['pr_auc']['hi']:.4f}]")

    # Paired diff: rfm_logistic - popularity_recency
    diff = paired_bootstrap_diff(y_test_align, s_logistic, s_recency, pr_auc, B=500, seed=SEED)
    results["paired_diff_logistic_vs_recency"] = diff
    print(f"[phase2] Δ PR-AUC (logistic - recency): {diff['point']:.4f} [{diff['lo']:.4f}, {diff['hi']:.4f}], p={diff['p']:.4f}")

    # Save scores
    np.savez(RESULTS / "phase2_scores.npz",
             y_test=y_test_align,
             customer_id=np.array(test_full["customer_id"].to_list()),
             majority=s_majority[:len(y_test_align)],
             recency=s_recency[:len(y_test_align)],
             logistic=s_logistic)

    (RESULTS / "phase2_baselines.json").write_text(json.dumps(results, indent=2, default=str))
    print("[phase2] Done.")


if __name__ == "__main__":
    main()
