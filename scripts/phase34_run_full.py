"""Phase 34: full sandbox run — core-1000 × 8 methods, parallelized.

Uses ThreadPoolExecutor to run multiple (customer, method) sessions concurrently
since the bottleneck is network latency to Gemini, not CPU.

Saves per-method per-customer outcomes to JSONL. Resumable: skips customers
already in the per-method JSONL.

Run: PYTHONPATH=. uv run python scripts/phase34_run_full.py [--method M1] [--max N] [--workers 16]
"""
from __future__ import annotations
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import polars as pl

from src import T_TEST_CUTOFF
from src.behavioral_trace import behavioral_trace
from src.sandbox.runner import run_session
from src.sandbox.methods import METHOD_REGISTRY, M3_KNN, M8_RAG, M2_RandomICL, M7_Hybrid, M8a_RAG_NoLabel


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "phase34_sandbox"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_write_locks: dict[str, threading.Lock] = {}


def _get_lock(method: str) -> threading.Lock:
    if method not in _write_locks:
        _write_locks[method] = threading.Lock()
    return _write_locks[method]


def load_neighbours():
    return json.loads((ROOT / "results" / "phase33_neighbours.json").read_text())


def load_history_pool():
    return json.loads((ROOT / "results" / "phase33_history_pool.json").read_text())


def _record_has_dp_errors(rec: dict) -> bool:
    """Return True if any DP in weekly_actions has an error field (indicates LLM call failed at DP level)."""
    if "error" in rec:
        return True
    for wa in rec.get("weekly_actions", []):
        for k in ("dp1", "dp2", "dp3"):
            v = wa.get(k)
            if isinstance(v, dict) and "error" in v:
                return True
    return False


def already_done(method_name: str) -> set[str]:
    """Customers we should SKIP — only those with clean records (no DP-level errors)."""
    fn = OUT_DIR / f"{method_name}.jsonl"
    if not fn.exists():
        return set()
    done = set()
    for line in fn.read_text().splitlines():
        try:
            rec = json.loads(line)
            if not _record_has_dp_errors(rec):
                done.add(rec["customer_id"])
        except Exception:
            pass
    return done


def rewrite_jsonl_dropping_bad(method_name: str):
    """De-duplicate by customer_id, preferring clean records over DP-errored ones.

    Strategy: for each customer, keep the FIRST clean (no DP error) record; if no
    clean record exists, drop them entirely (they need to be re-run).
    """
    fn = OUT_DIR / f"{method_name}.jsonl"
    if not fn.exists():
        return 0, 0
    clean: dict[str, str] = {}
    dirty_only: set[str] = set()
    dropped_dupes = 0
    dropped_bad = 0
    for line in fn.read_text().splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            dropped_bad += 1
            continue
        cid = rec.get("customer_id")
        if not cid:
            dropped_bad += 1
            continue
        if _record_has_dp_errors(rec):
            dirty_only.add(cid)
            dropped_bad += 1
            continue
        if cid in clean:
            dropped_dupes += 1
            continue
        clean[cid] = line
    # If a cid is in both clean and dirty_only, that's fine — clean wins.
    fn.write_text("\n".join(clean.values()) + ("\n" if clean else ""))
    return len(clean), dropped_bad + dropped_dupes


def _load_optional(p: Path):
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def make_method(method_name: str, neighbours, history_pool):
    cls = METHOD_REGISTRY[method_name]
    if method_name == "M3":
        return M3_KNN(neighbours=neighbours)
    if method_name == "M2":
        random_nb = _load_optional(ROOT / "results" / "phase38_neighbours_random.json") or {}
        return M2_RandomICL(neighbours_random=random_nb)
    if method_name == "M8":
        return M8_RAG(history_pool=history_pool)
    if method_name == "M8a":
        return M8a_RAG_NoLabel(history_pool=history_pool)
    if method_name == "M7":
        lgbm = _load_optional(ROOT / "results" / "phase38_lgbm_preds.json") or {}
        return M7_Hybrid(lgbm_preds=lgbm)
    return cls()


def run_one(cid, trace, method_name, method_obj, label, bucket, want_scalar: bool):
    try:
        outcome = run_session(cid, trace, method_obj)
        rec = {
            "customer_id": cid,
            "method": method_name,
            "actual": label,
            "bucket": bucket,
            "purchased": int(outcome.purchased),
            "n_purchases": len(outcome.weekly_purchases),
            "purchased_items": outcome.weekly_purchases,
            "weekly_actions": outcome.weekly_actions,
            "declared_max_purchases": outcome.final_state.declared_max_purchases,
            "n_dp_calls": outcome.n_dp_calls,
        }
        if want_scalar:
            try:
                sc = method_obj.scalar_prob({**trace, "_cid": cid})
                rec["scalar_prob"] = sc.get("scalar_prob", 0.5)
            except Exception as e:
                rec["scalar_prob_error"] = str(e)
        return rec
    except Exception as e:
        return {"customer_id": cid, "method": method_name, "error": str(e),
                "actual": label, "bucket": bucket}


def write_record(method: str, rec: dict):
    fn = OUT_DIR / f"{method}.jsonl"
    with _get_lock(method):
        with fn.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", action="append", default=None)
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--scalar", action="store_true")
    args = ap.parse_args()

    methods = args.method or list(METHOD_REGISTRY.keys())

    core = pl.read_parquet(ROOT / "results" / "phase31_core1000_v3.parquet")
    cids = core["customer_id"].to_list()
    labels = dict(zip(cids, core["label"].to_list()))
    buckets = dict(zip(cids, core["activity_bucket"].to_list()))
    if args.max:
        cids = cids[: args.max]

    print(f"Loading neighbours/history pool...", flush=True)
    neighbours = load_neighbours()
    history_pool = load_history_pool()

    print(f"Building traces for {len(cids)} customers...", flush=True)
    t0 = time.time()
    traces = behavioral_trace(cids, cutoff=date.fromisoformat(T_TEST_CUTOFF))
    print(f"Traces ready: {len(traces)}  ({time.time()-t0:.1f}s)", flush=True)

    for mname in methods:
        # Drop existing records with DP-level errors so we can re-run them cleanly.
        good_n, bad_n = rewrite_jsonl_dropping_bad(mname)
        if bad_n:
            print(f"\n[{mname}] dropped {bad_n} previously errored records "
                  f"(kept {good_n} clean records)", flush=True)
        done = already_done(mname)
        todo = [c for c in cids if c not in done and c in traces]
        print(f"\n=== Method {mname}: {len(done)} already done, {len(todo)} to run "
              f"with {args.workers} workers ===", flush=True)
        m = make_method(mname, neighbours, history_pool)
        t_start = time.time()
        ok = err = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(run_one, cid, traces[cid], mname, m,
                          labels[cid], buckets[cid], args.scalar): cid
                for cid in todo
            }
            done_count = 0
            for fut in as_completed(futures):
                cid = futures[fut]
                try:
                    rec = fut.result()
                except Exception as e:
                    rec = {"customer_id": cid, "method": mname, "error": f"future-exc: {e}",
                           "actual": labels[cid], "bucket": buckets[cid]}
                write_record(mname, rec)
                if "error" in rec:
                    err += 1
                else:
                    ok += 1
                done_count += 1
                if done_count % 25 == 0 or done_count == len(todo):
                    elapsed = time.time() - t_start
                    rate = done_count / max(elapsed, 1e-6)
                    eta = (len(todo) - done_count) / max(rate, 1e-6)
                    print(f"  [{mname}] {done_count}/{len(todo)} ok={ok} err={err} "
                          f"rate={rate:.2f}/s ETA={eta:.0f}s", flush=True)


if __name__ == "__main__":
    main()
