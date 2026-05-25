"""Phase 44: validate the real sandbox v2 with stochastic dynamics + inventory.

Runs M1 zero-shot through sandbox v2 (Gemini) on the core-200 customers, with:
- Stochastic stimulus arrival (Bernoulli per p_arrive)
- Shared inventory across customers
- Post-purchase satisfaction reward signal
- Recency rolls forward each week
- Purchase fatigue

Compares to sandbox v1 (deterministic) M1 result on the same customers.
"""
from __future__ import annotations
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from src import T_TEST_CUTOFF
from src.behavioral_trace import behavioral_trace
from src.sandbox.methods import M1_ZeroShot
from src.sandbox import env as sandbox_env
from src.sandbox_v2.runner import run_world_session
from src.sandbox_v2.world import SharedInventory


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase44_world_sandbox_M1.jsonl"


def build_initial_inventory(scale: float = 0.5) -> SharedInventory:
    """Initial stock for each article = floor(scale * pre-cutoff popularity)."""
    sandbox_env._ensure_popularity_indices()
    init = {}
    for sec, items in sandbox_env._POPULAR_BY_SECTION.items():
        for it in items:
            init[str(it["article_id"])] = max(1, int(scale * (it.get("popularity") or 1)))
    # Reduce a fraction to make depletion meaningful: items in the top decile
    # by popularity get capped so they actually run out under heavy demand.
    sorted_items = sorted(init.items(), key=lambda kv: -kv[1])
    cap = max(2, int(len(sorted_items) * 0.0001))  # ~top 0.01% capped to 2 stock
    for a, _ in sorted_items[:cap]:
        init[a] = 2
    return SharedInventory(init_stock=init)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max", type=int, default=200)
    args = ap.parse_args()

    # Use the same core-200 sample as phase42 for direct comparison.
    inp = json.loads((ROOT / "results" / "phase42_claude_proper_input.json").read_text())
    labels = json.loads((ROOT / "results" / "phase42_actual_labels.json").read_text())
    if args.max:
        inp = inp[: args.max]
    cids = [c["customer_id"] for c in inp]
    buckets = {c["customer_id"]: c["bucket"] for c in inp}

    print(f"Building traces for {len(cids)} customers...", flush=True)
    t0 = time.time()
    traces = behavioral_trace(cids, cutoff=date.fromisoformat(T_TEST_CUTOFF))
    print(f"Traces ready: {len(traces)} ({time.time()-t0:.1f}s)", flush=True)

    print("Initializing shared inventory (this may take a few seconds)...", flush=True)
    inventory = build_initial_inventory(scale=0.05)
    print(f"Inventory pool: {len(inventory.stock)} unique articles", flush=True)

    if OUT.exists():
        OUT.unlink()

    write_lock = threading.Lock()

    def _run_one(cid):
        if cid not in traces:
            return None
        m = M1_ZeroShot()
        try:
            outcome = run_world_session(cid, traces[cid], m, inventory)
            return {
                "customer_id": cid,
                "method": "M1_world_v2",
                "actual": labels.get(cid, 0),
                "bucket": buckets[cid],
                "purchased": int(outcome.purchased),
                "n_purchases": len(outcome.weekly_purchases),
                "n_dp_calls": outcome.n_dp_calls,
                "weekly_actions": outcome.weekly_actions,
            }
        except Exception as e:
            return {"customer_id": cid, "method": "M1_world_v2",
                    "error": str(e), "actual": labels.get(cid, 0),
                    "bucket": buckets[cid]}

    n_done = 0
    print(f"\n=== Running sandbox v2 M1 with {args.workers} workers ===", flush=True)
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_run_one, cid): cid for cid in cids}
        for fut in as_completed(futures):
            rec = fut.result()
            if rec:
                with write_lock:
                    with OUT.open("a") as f:
                        f.write(json.dumps(rec, default=str) + "\n")
                n_done += 1
                if n_done % 25 == 0:
                    elapsed = time.time() - t_start
                    rate = n_done / max(elapsed, 1e-6)
                    eta = (len(cids) - n_done) / max(rate, 1e-6)
                    print(f"  {n_done}/{len(cids)} rate={rate:.2f}/s ETA={eta:.0f}s "
                          f"stim_arrivals_so_far={sum(1 for cid, aid in inventory.purchases)}",
                          flush=True)

    # Final summary
    recs = []
    for line in OUT.read_text().splitlines():
        try:
            recs.append(json.loads(line))
        except Exception:
            pass
    n = len(recs)
    purch = sum(r.get("purchased", 0) for r in recs)
    actual = sum(r.get("actual", 0) for r in recs)
    print(f"\n=== DONE n={n} ===")
    print(f"  Sandbox-v2 M1 funnel rate: {purch/n:.3f}")
    print(f"  Actual rate:              {actual/n:.3f}")
    print(f"  Signed gap (sandbox-v2):  {(purch-actual)/n:+.3f}")
    print(f"  Inventory consumed: {sum(1 for _ in inventory.purchases)} purchases across customers")


if __name__ == "__main__":
    main()
