"""Phase 1: Build splits + EDA report."""

from __future__ import annotations
import json
from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt
import numpy as np

from src.splits import build_train_test_splits

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def main():
    print("[phase1] Building splits...")
    splits = build_train_test_splits()
    train = splits["train"]
    val = splits["val"]
    test = splits["test"]
    train_full = splits["train_full"]
    test_full = splits["test_full"]

    out = {
        "train_pool_full": len(train_full),
        "test_pool_full": len(test_full),
        "train_label_rate_full": float(train_full["label"].mean()),
        "test_label_rate_full": float(test_full["label"].mean()),
        "train_n": len(train),
        "val_n": len(val),
        "test_n": len(test),
        "train_label_rate": float(train["label"].mean()),
        "val_label_rate": float(val["label"].mean()),
        "test_label_rate": float(test["label"].mean()),
    }

    # Activity bucket breakdown
    bucket_breakdown = {}
    for split_name, frame in [("train", train), ("val", val), ("test", test)]:
        bucket_breakdown[split_name] = (
            frame.group_by("activity_bucket")
            .agg(pl.len().alias("n"), pl.col("label").mean().alias("label_rate"))
            .sort("activity_bucket")
            .to_dicts()
        )
    out["bucket_breakdown"] = bucket_breakdown

    # n_tx histogram for EDA figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(np.log10(train["n_tx_pre_cutoff"].to_numpy() + 1), bins=50)
    axes[0].set_xlabel("log10(n_tx_pre_cutoff + 1)")
    axes[0].set_ylabel("customers")
    axes[0].set_title("Train: pre-cutoff transaction count")
    # label rate by bucket
    bb = bucket_breakdown["train"]
    axes[1].bar([b["activity_bucket"] for b in bb], [b["label_rate"] for b in bb])
    axes[1].set_ylabel("30d repeat-purchase rate")
    axes[1].set_title("Train: label rate by activity bucket")
    plt.tight_layout()
    plt.savefig(RESULTS / "phase1_eda.png", dpi=120)
    plt.close()

    (RESULTS / "phase1_summary.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))
    print("[phase1] Done.")


if __name__ == "__main__":
    main()
