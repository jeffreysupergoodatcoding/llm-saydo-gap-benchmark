"""Phase 31: select v3 sandbox core-1000, customer-disjoint from v2 core-1000.

200 customers per activity bucket × 5 buckets. Stratified, deterministic seed=2026.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import polars as pl


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase31_core1000_v3.parquet"
LIST_OUT = ROOT / "results" / "phase31_core1000_v3.json"


def main():
    test = pl.read_parquet(ROOT / "data" / "splits" / "test.parquet")
    # Load v2 IDs to exclude (raw H&M customer_ids are already SHA256-style hashes)
    v2 = np.load(ROOT / "results" / "phase10_F-base_scores.npz", allow_pickle=True)
    v2_ids = set(v2["customer_id"].tolist())
    test = test.filter(~pl.col("customer_id").is_in(list(v2_ids)))
    print(f"After excluding v2 core-1000: {len(test)} candidates")

    buckets = ["1", "2-5", "6-20", "21-100", "101+"]
    rng = np.random.default_rng(2026)

    picks = []
    for b in buckets:
        pool = test.filter(pl.col("activity_bucket") == b)
        n_pool = len(pool)
        if n_pool < 200:
            print(f"  WARNING bucket {b}: only {n_pool} available, taking all")
            chosen_idx = np.arange(n_pool)
        else:
            chosen_idx = rng.choice(n_pool, size=200, replace=False)
        sub = pool[chosen_idx.tolist()]
        picks.append(sub)
        print(f"  bucket {b}: {len(sub)} picked (label_rate={sub['label'].mean():.3f})")

    core = pl.concat(picks)
    print(f"Total core-1000: {len(core)}  overall label_rate={core['label'].mean():.3f}")

    core.write_parquet(OUT)
    LIST_OUT.write_text(json.dumps({
        "n": len(core),
        "seed": 2026,
        "customer_ids": core["customer_id"].to_list(),
        "by_bucket": {b: int((core["activity_bucket"] == b).sum()) for b in buckets},
        "label_rate_overall": float(core["label"].mean()),
        "label_rate_by_bucket": {
            b: float(core.filter(pl.col("activity_bucket") == b)["label"].mean()) for b in buckets
        },
    }, indent=2))
    print(f"Wrote {OUT}\nWrote {LIST_OUT}")


if __name__ == "__main__":
    main()
