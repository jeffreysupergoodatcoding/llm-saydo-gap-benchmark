"""Phase 39: 3-seed stimulus sensitivity for M1 + S2 + S4.

Reviewer red flag #4: single seed for sandbox stimuli. Re-run M1, S2, S4 with
seeds 2027 and 2028 (in addition to the headline seed 2026) on a 200-customer
sub-sample of the core-1000 (stratified, deterministic).

Output: results/phase39_seed_sensitivity/{method}_seed{seed}.jsonl
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
from src.sandbox import env as sandbox_env
from src.sandbox.runner import run_session
from src.sandbox.methods import M1_ZeroShot, S2_OutcomeConditioned, S4_Commitment


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "phase39_seed_sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHODS = {"M1": M1_ZeroShot, "S2": S2_OutcomeConditioned, "S4": S4_Commitment}
SEEDS = [2027, 2028]
N_SUBSAMPLE = 200  # 40 per bucket × 5 buckets


def make_subsample(seed: int = 2026) -> tuple[list[str], dict, dict]:
    core = pl.read_parquet(ROOT / "results" / "phase31_core1000_v3.parquet")
    rng = np.random.default_rng(seed)
    parts = []
    for b in ["1", "2-5", "6-20", "21-100", "101+"]:
        sub = core.filter(pl.col("activity_bucket") == b)
        idx = rng.choice(len(sub), size=40, replace=False)
        parts.append(sub[idx.tolist()])
    sub = pl.concat(parts)
    return sub["customer_id"].to_list(), \
           dict(zip(sub["customer_id"].to_list(), sub["label"].to_list())), \
           dict(zip(sub["customer_id"].to_list(), sub["activity_bucket"].to_list()))


_write_lock = threading.Lock()


def _write(fn: Path, rec: dict):
    with _write_lock:
        with fn.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")


def run_one(cid, trace, method_obj, label, bucket, mname, seed):
    # Monkey-patch the seed function for this thread context — actually we just rely
    # on the closure-style _seed_for to change globally. Simpler: replace _seed_for.
    try:
        outcome = run_session(cid, trace, method_obj)
        return {
            "customer_id": cid, "method": mname, "seed": seed,
            "actual": label, "bucket": bucket,
            "purchased": int(outcome.purchased),
            "n_purchases": len(outcome.weekly_purchases),
            "n_dp_calls": outcome.n_dp_calls,
        }
    except Exception as e:
        return {"customer_id": cid, "method": mname, "seed": seed,
                "error": str(e), "actual": label, "bucket": bucket}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    cids, labels, buckets = make_subsample()
    print(f"Subsample n={len(cids)}", flush=True)
    print("Building traces...", flush=True)
    traces = behavioral_trace(cids, cutoff=date.fromisoformat(T_TEST_CUTOFF))
    print(f"Traces ready: {len(traces)}", flush=True)

    for seed in SEEDS:
        # Override the seed function
        original_seed_for = sandbox_env._seed_for

        def new_seed_for(customer_id: str, week: int, _s=seed) -> int:
            from hashlib import sha256
            h = sha256(f"{customer_id}|{week}|{_s}".encode()).digest()
            return int.from_bytes(h[:8], "big") % (2**31 - 1)

        sandbox_env._seed_for = new_seed_for
        try:
            for mname, cls in METHODS.items():
                fn = OUT_DIR / f"{mname}_seed{seed}.jsonl"
                done_cids = set()
                if fn.exists():
                    for line in fn.read_text().splitlines():
                        try:
                            done_cids.add(json.loads(line)["customer_id"])
                        except: pass
                todo = [c for c in cids if c not in done_cids and c in traces]
                print(f"\n=== seed={seed} method={mname}: {len(done_cids)} done, {len(todo)} to run ===", flush=True)
                m = cls()
                t0 = time.time()
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    futures = {
                        ex.submit(run_one, cid, traces[cid], m, labels[cid], buckets[cid], mname, seed): cid
                        for cid in todo
                    }
                    n_done = 0
                    for fut in as_completed(futures):
                        rec = fut.result()
                        _write(fn, rec)
                        n_done += 1
                        if n_done % 25 == 0:
                            elapsed = time.time() - t0
                            rate = n_done / max(elapsed, 1e-6)
                            print(f"  [seed={seed} {mname}] {n_done}/{len(todo)} rate={rate:.2f}/s", flush=True)
        finally:
            sandbox_env._seed_for = original_seed_for


if __name__ == "__main__":
    main()
