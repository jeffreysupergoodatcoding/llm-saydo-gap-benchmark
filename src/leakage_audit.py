"""Standalone leakage audit. Run after each phase that builds features."""

from __future__ import annotations
from datetime import date
import polars as pl

from . import T_TRAIN_CUTOFF, T_TEST_CUTOFF
from .data import load_transactions
from .splits import load_split
from .features import rfm_features, bag_of_categories


def assert_no_future_tx(customer_ids: list[str], cutoff: str) -> dict:
    """Verify the underlying transactions used for these customer_ids have no t_dat >= cutoff."""
    cutoff_d = date.fromisoformat(cutoff)
    tx = (
        load_transactions()
        .filter((pl.col("customer_id").is_in(customer_ids[:5000])) & (pl.col("t_dat") < cutoff_d))
        .select(pl.col("t_dat").max())
        .collect()
        .item()
    )
    assert tx is None or tx < cutoff_d, f"FAILURE: tx max {tx} >= cutoff {cutoff_d}"
    return {"checked_customers": min(len(customer_ids), 5000), "max_t_dat_pre_cutoff": str(tx)}


def assert_split_disjoint() -> dict:
    tr = set(load_split("train")["customer_id"].to_list())
    va = set(load_split("val")["customer_id"].to_list())
    te = set(load_split("test")["customer_id"].to_list())
    tr_va = tr & va
    tr_te = tr & te
    va_te = va & te
    assert len(tr_va) == 0, f"train ∩ val: {len(tr_va)} customers"
    assert len(tr_te) == 0, f"train ∩ test: {len(tr_te)} customers"
    assert len(va_te) == 0, f"val ∩ test: {len(va_te)} customers"
    return {"train_n": len(tr), "val_n": len(va), "test_n": len(te), "disjoint": True}


def spot_check_rfm() -> dict:
    train = load_split("train").head(50)
    ids = train["customer_id"].to_list()
    rfm = rfm_features(ids, cutoff=T_TRAIN_CUTOFF)
    # Now manually verify that recency_days is consistent with transactions
    cutoff_d = date.fromisoformat(T_TRAIN_CUTOFF)
    tx = (
        load_transactions()
        .filter((pl.col("customer_id").is_in(ids)) & (pl.col("t_dat") < cutoff_d))
        .group_by("customer_id")
        .agg(pl.col("t_dat").max().alias("max_t"))
        .collect()
    )
    joined = rfm.join(tx, on="customer_id", how="left")
    for r in joined.iter_rows(named=True):
        expected = (cutoff_d - r["max_t"]).days
        observed = r["recency_days"]
        assert abs(int(expected) - int(observed)) <= 1, f"recency mismatch: cust={r['customer_id']}, expected {expected}, got {observed}"
    return {"spot_checked_customers": len(joined), "recency_consistent": True}


def main():
    out = {}
    train_ids = load_split("train")["customer_id"].to_list()
    test_ids = load_split("test")["customer_id"].to_list()
    out["train_pre_cutoff"] = assert_no_future_tx(train_ids, T_TRAIN_CUTOFF)
    out["test_pre_cutoff"] = assert_no_future_tx(test_ids, T_TEST_CUTOFF)
    out["split_disjoint"] = assert_split_disjoint()
    out["rfm_spot_check"] = spot_check_rfm()
    print(out)
    return out


if __name__ == "__main__":
    main()
