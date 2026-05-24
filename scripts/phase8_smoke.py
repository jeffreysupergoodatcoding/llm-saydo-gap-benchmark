"""Phase 8 smoke test: run Fragment cognition pipeline on n=5 customers."""

from __future__ import annotations
import json
from src.splits import load_split
from src.cognition_fragment.run import run_pipeline
from src import T_TEST_CUTOFF


def main():
    test = load_split("test").head(5)
    cids = test["customer_id"].to_list()
    print(f"[8 smoke] running F-base (include_base_rate=True) on n={len(cids)}...", flush=True)
    out_base = run_pipeline(cids, cutoff=T_TEST_CUTOFF, include_base_rate=True, verbose=True)

    print(f"\n[8 smoke] running F-nobase (include_base_rate=False) on same n={len(cids)}...", flush=True)
    out_nobase = run_pipeline(cids, cutoff=T_TEST_CUTOFF, include_base_rate=False, verbose=True)

    # Compare side-by-side
    print("\n[8 smoke] Per-customer F-base vs F-nobase:")
    print(f"  {'cid[:8]':<10} {'orders':<7} {'recency':<8} {'F-base':<8} {'F-nobase':<10} {'actual':<7}")
    actual = {row["customer_id"]: int(row["label"]) for row in test.iter_rows(named=True)}
    for cid in cids:
        b = out_base[cid].get("stated_intent_prob", 0.5)
        nb = out_nobase[cid].get("stated_intent_prob", 0.5)
        ts = out_base[cid].get("trace_summary", {})
        print(f"  {cid[:8]:<10} {ts.get('total_orders','?'):<7} {ts.get('recency_days','?'):<8} {b:<8.3f} {nb:<10.3f} {actual[cid]:<7}")

    print("\n[8 smoke] sample verbatims (F-base):")
    for cid in cids[:3]:
        v = out_base[cid].get("deliberation", {}).get("verbatim_reaction", "")
        print(f"  {cid[:8]}...: {v}")


if __name__ == "__main__":
    main()
