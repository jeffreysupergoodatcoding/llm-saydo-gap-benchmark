"""Phase 32: smoke test the v3 sandbox on n=5 with all 8 methods.

Validates the runner + each method end-to-end, prints per-customer trace.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
from datetime import date

import polars as pl

from src import T_TEST_CUTOFF
from src.behavioral_trace import behavioral_trace
from src.sandbox.runner import run_session
from src.sandbox.methods import METHOD_REGISTRY


ROOT = Path(__file__).resolve().parents[1]


def main():
    core = pl.read_parquet(ROOT / "results" / "phase31_core1000_v3.parquet")
    # Pick 5 across buckets
    sample = (
        pl.concat([
            core.filter(pl.col("activity_bucket") == b).head(1)
            for b in ["1", "2-5", "6-20", "21-100", "101+"]
        ])
    )
    cids = sample["customer_id"].to_list()
    labels = dict(zip(cids, sample["label"].to_list()))
    buckets = dict(zip(cids, sample["activity_bucket"].to_list()))

    print(f"Building traces for {len(cids)} customers...")
    traces = behavioral_trace(cids, cutoff=date.fromisoformat(T_TEST_CUTOFF))
    print(f"Got traces for {len(traces)}")

    results = {}
    for mname, mclass in METHOD_REGISTRY.items():
        print(f"\n=== {mname} ({mclass.__name__}) ===")
        m = mclass()
        per_cust = []
        for cid in cids:
            if cid not in traces:
                print(f"  {cid[:10]}... NO TRACE, skipping")
                continue
            trace = traces[cid]
            try:
                outcome = run_session(cid, trace, m)
                per_cust.append({
                    "cid": cid,
                    "bucket": buckets[cid],
                    "actual": labels[cid],
                    "purchased": outcome.purchased,
                    "n_purchases": len(outcome.weekly_purchases),
                    "n_dp_calls": outcome.n_dp_calls,
                    "weeks_engaged": sum(1 for wa in outcome.weekly_actions if wa.get("dp1") and wa["dp1"]["action"] == "ENGAGE"),
                })
                print(f"  {cid[:10]}... bucket={buckets[cid]} actual={labels[cid]} llm_purchased={outcome.purchased} dp_calls={outcome.n_dp_calls}")
            except Exception as e:
                print(f"  {cid[:10]}... ERROR: {e}")
                per_cust.append({"cid": cid, "error": str(e)})
        results[mname] = per_cust

    out = ROOT / "results" / "phase32_smoke.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
