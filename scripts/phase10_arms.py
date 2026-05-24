"""Phase 10: run F-base and F-nobase arms on the core-1000.

Single-script entry; --arm selects which.
"""

from __future__ import annotations
import argparse
import json
import numpy as np
import polars as pl
from datetime import datetime
from pathlib import Path

from src import T_TEST_CUTOFF, SEED
from src.splits import load_split
from src.cognition_fragment.run import run_pipeline
from src.eval import all_metrics

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def stratified_core_sample(test_df: pl.DataFrame, per_bucket: int, seed: int = SEED) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    buckets = sorted(test_df["activity_bucket"].unique().to_list())
    parts = []
    for b in buckets:
        sub = test_df.filter(pl.col("activity_bucket") == b)
        k = min(per_bucket, len(sub))
        idx = rng.choice(len(sub), size=k, replace=False)
        parts.append(sub[idx.tolist()])
    return pl.concat(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["F-base", "F-nobase"], required=True)
    ap.add_argument("--per-bucket", type=int, default=200)
    ap.add_argument("--model", default="gemini-2.5-flash")
    args = ap.parse_args()

    test = load_split("test")
    core = stratified_core_sample(test, per_bucket=args.per_bucket, seed=SEED)
    print(f"[10/{args.arm}] core-{len(core)} ({args.per_bucket}/bucket × {core['activity_bucket'].n_unique()} buckets)", flush=True)
    cids = core["customer_id"].to_list()

    include_base_rate = (args.arm == "F-base")
    print(f"[10/{args.arm}] running pipeline, include_base_rate={include_base_rate}, model={args.model}", flush=True)
    t0 = datetime.utcnow()
    out = run_pipeline(cids, cutoff=T_TEST_CUTOFF, include_base_rate=include_base_rate,
                       model=args.model, verbose=False)
    elapsed_s = (datetime.utcnow() - t0).total_seconds()
    print(f"[10/{args.arm}] done in {elapsed_s:.0f}s", flush=True)

    # Save scores + verbatims for downstream analysis
    scores = np.array([out[c].get("stated_intent_prob", 0.5) for c in cids])
    raw_scores = np.array([
        out[c].get("decision", {}).get("stated_intent_prob_raw", out[c].get("stated_intent_prob", 0.5))
        for c in cids
    ])
    actual = core["label"].to_numpy()
    verbatims = [out[c].get("deliberation", {}).get("verbatim_reaction", "") for c in cids]
    objections = [out[c].get("deliberation", {}).get("key_objection", "") for c in cids]
    reasonings = [out[c].get("deliberation", {}).get("reasoning", "") for c in cids]
    confs = [out[c].get("deliberation", {}).get("confidence", 50.0) for c in cids]
    frictions = [out[c].get("deliberation", {}).get("friction_score", 50.0) for c in cids]
    buckets_arr = core["activity_bucket"].to_numpy()
    n_tx_arr = core["n_tx_pre_cutoff"].to_numpy()

    np.savez(RESULTS / f"phase10_{args.arm}_scores.npz",
             customer_id=np.array(cids),
             stated_intent=scores,
             stated_intent_raw=raw_scores,
             actual=actual,
             activity_bucket=buckets_arr,
             n_tx_pre_cutoff=n_tx_arr,
             verbatim=np.array(verbatims, dtype=object),
             key_objection=np.array(objections, dtype=object),
             reasoning=np.array(reasonings, dtype=object),
             confidence=np.array(confs),
             friction=np.array(frictions))

    # Headline metrics
    mean_stated = float(scores.mean())
    mean_actual = float(actual.mean())
    signed_gap = mean_stated - mean_actual
    m = all_metrics(actual, scores, B=500, seed=SEED)
    per_bucket = {}
    for b in sorted(set(buckets_arr.tolist())):
        mask = buckets_arr == b
        if mask.sum() < 20:
            continue
        per_bucket[b] = {
            "n": int(mask.sum()),
            "mean_stated": float(scores[mask].mean()),
            "mean_actual": float(actual[mask].mean()),
            "signed_gap": float(scores[mask].mean() - actual[mask].mean()),
        }

    summary = {
        "arm": args.arm,
        "model": args.model,
        "n": len(cids),
        "elapsed_s": elapsed_s,
        "mean_stated_intent_prob": mean_stated,
        "mean_actual_label": mean_actual,
        "signed_gap_E_stated_minus_E_actual": signed_gap,
        "pr_auc": m["pr_auc"],
        "roc_auc": m["roc_auc"],
        "brier": m["brier"],
        "ece": m["ece"],
        "wasserstein_decile": m["wasserstein_decile"],
        "under_dispersion": m["under_dispersion"],
        "per_bucket_gap": per_bucket,
    }
    (RESULTS / f"phase10_{args.arm}_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(f"\n[10/{args.arm}] mean_stated = {mean_stated:.4f}, mean_actual = {mean_actual:.4f}, "
          f"signed_gap = {signed_gap:+.4f}")
    print(f"[10/{args.arm}] PR-AUC = {m['pr_auc']['point']:.4f} [{m['pr_auc']['lo']:.4f}, {m['pr_auc']['hi']:.4f}]")
    print(f"[10/{args.arm}] per-bucket signed-gap:")
    for b, st in per_bucket.items():
        print(f"   bucket {b:>6}: gap = {st['signed_gap']:+.3f}  (n={st['n']}, stated={st['mean_stated']:.3f}, actual={st['mean_actual']:.3f})")


if __name__ == "__main__":
    main()
