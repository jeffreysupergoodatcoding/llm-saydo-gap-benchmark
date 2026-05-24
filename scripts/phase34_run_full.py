"""Phase 34: full sandbox run — core-1000 × 8 methods, with incremental persistence.

Saves per-method per-customer outcomes to JSONL. Resumable: skips customers
already in the per-method JSONL.

Run with: PYTHONPATH=. uv run python scripts/phase34_run_full.py [--method M1] [--max N]
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import polars as pl

from src import T_TEST_CUTOFF
from src.behavioral_trace import behavioral_trace
from src.sandbox.runner import run_session
from src.sandbox.methods import (
    METHOD_REGISTRY, M3_KNN, M8_RAG,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "phase34_sandbox"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_neighbours():
    p = ROOT / "results" / "phase33_neighbours.json"
    return json.loads(p.read_text())


def load_history_pool():
    p = ROOT / "results" / "phase33_history_pool.json"
    return json.loads(p.read_text())


def already_done(method_name: str) -> set[str]:
    fn = OUT_DIR / f"{method_name}.jsonl"
    if not fn.exists():
        return set()
    done = set()
    for line in fn.read_text().splitlines():
        try:
            done.add(json.loads(line)["customer_id"])
        except Exception:
            pass
    return done


def make_method(method_name: str, neighbours, history_pool):
    cls = METHOD_REGISTRY[method_name]
    if method_name == "M3":
        return M3_KNN(neighbours=neighbours)
    if method_name == "M8":
        return M8_RAG(history_pool=history_pool)
    return cls()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", action="append", default=None,
                    help="Restrict to method names. Default: all 8.")
    ap.add_argument("--max", type=int, default=None,
                    help="Max customers per method.")
    ap.add_argument("--scalar", action="store_true",
                    help="Also compute per-method scalar_prob alongside sandbox.")
    args = ap.parse_args()

    methods = args.method or list(METHOD_REGISTRY.keys())
    core = pl.read_parquet(ROOT / "results" / "phase31_core1000_v3.parquet")
    cids = core["customer_id"].to_list()
    labels = dict(zip(cids, core["label"].to_list()))
    buckets = dict(zip(cids, core["activity_bucket"].to_list()))
    if args.max:
        cids = cids[: args.max]

    print(f"Loading neighbours/history pool...")
    neighbours = load_neighbours()
    history_pool = load_history_pool()

    print(f"Building traces for {len(cids)} customers (may take a minute)...")
    t0 = time.time()
    traces = behavioral_trace(cids, cutoff=date.fromisoformat(T_TEST_CUTOFF))
    print(f"Traces ready: {len(traces)}  ({time.time()-t0:.1f}s)")

    for mname in methods:
        done = already_done(mname)
        todo = [c for c in cids if c not in done and c in traces]
        print(f"\n=== Method {mname}: {len(done)} already done, {len(todo)} to run ===")
        m = make_method(mname, neighbours, history_pool)
        fn = OUT_DIR / f"{mname}.jsonl"
        t_start = time.time()
        ok = 0
        err = 0
        for i, cid in enumerate(todo):
            trace = traces[cid]
            try:
                outcome = run_session(cid, trace, m)
                rec = {
                    "customer_id": cid,
                    "method": mname,
                    "actual": labels[cid],
                    "bucket": buckets[cid],
                    "purchased": int(outcome.purchased),
                    "n_purchases": len(outcome.weekly_purchases),
                    "purchased_items": outcome.weekly_purchases,
                    "weekly_actions": outcome.weekly_actions,
                    "declared_max_purchases": outcome.final_state.declared_max_purchases,
                    "n_dp_calls": outcome.n_dp_calls,
                }
                # Compute scalar if requested
                if args.scalar:
                    try:
                        sc = m.scalar_prob({**trace, "_cid": cid})
                        rec["scalar_prob"] = sc.get("scalar_prob", 0.5)
                    except Exception as e:
                        rec["scalar_prob_error"] = str(e)
                with fn.open("a") as f:
                    f.write(json.dumps(rec, default=str) + "\n")
                ok += 1
            except Exception as e:
                err += 1
                err_rec = {"customer_id": cid, "method": mname, "error": str(e),
                           "actual": labels[cid], "bucket": buckets[cid]}
                with fn.open("a") as f:
                    f.write(json.dumps(err_rec) + "\n")
                print(f"  ERR {cid[:10]}... {e}")
            if (i + 1) % 50 == 0 or (i + 1) == len(todo):
                elapsed = time.time() - t_start
                rate = (i + 1) / max(elapsed, 1e-6)
                eta = (len(todo) - i - 1) / max(rate, 1e-6)
                print(f"  [{mname}] {i+1}/{len(todo)} ok={ok} err={err}  rate={rate:.2f}/s  ETA={eta:.0f}s")


if __name__ == "__main__":
    main()
