"""Phase 22: MovieLens F-base / F-nobase arms.

Cross-domain replication of the H&M Iteration-1-4 result. Same cognition
pipeline (attention/memory/affect/deliberation/decision) adapted by
swapping H&M behavioral_trace for the MovieLens-adapted trace.
"""

from __future__ import annotations
import argparse, json, time
import numpy as np
import polars as pl
from pathlib import Path

from src import SEED
from src.movielens_data import (
    behavioral_trace_ml, T_TEST_CUTOFF, SPLITS as ML_SPLITS,
)
from src.cognition_fragment.attention import rank_attention
from src.cognition_fragment.memory import retrieve_memories
from src.cognition_fragment.affect import compute_affect
from src.cognition_fragment.deliberation import (
    build_prompt as _build_prompt_hm,
    SYSTEM_PROMPT_BASE, parse_response, HM_BASE_RATES,
)
from src.cognition_fragment.decision import blend_decision
from src.llm_client import call_llm
from src.eval import all_metrics

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# MovieLens-specific per-bucket base rates (from phase22 split build).
ML_BASE_RATES = {
    "1": 0.667,    # tiny n=3; effectively noise — kept for prompt completeness
    "2-5": 0.667,
    "6-20": 0.005,
    "21-100": 0.005,
    "101+": 0.020,
}


def build_prompt(trace, attention, memories, affect, include_base_rate: bool) -> str:
    # Re-use the H&M deliberation prompt body; swap in MovieLens base-rate table.
    from src.cognition_fragment.deliberation import _trace_block, _format_base_rate_table
    body = _trace_block(trace, attention, memories, affect)
    pieces = [body]
    if include_base_rate:
        rows = "\n".join(f"  - prior-rating-count {k}: {v*100:.1f}%" for k, v in ML_BASE_RATES.items())
        pieces.append(
            "Empirical base rates from MovieLens 25M test data (use as calibration anchor):\n" + rows
        )
    pieces.append(
        "Question: in the next 30 days starting from your last rating date + 1, "
        "will you rate any new movie on MovieLens?\n"
        "Reason briefly first-person, then output JSON with the required keys."
    )
    return "\n\n".join(pieces)


SYSTEM_PROMPT_ML = SYSTEM_PROMPT_BASE.replace("H&M customer", "MovieLens user") \
    .replace("Heavy buyers (101+ prior tx) should land near 60-80%.",
             "Heavy raters (101+ ratings) should land near 1-5% — most MovieLens users rate sparsely.") \
    .replace("Single-purchase or lapsed customers should land below 10%.",
             "Single-rating or lapsed users should land below 1%.")


def deliberate_ml(trace, attention, memories, affect, include_base_rate: bool,
                  model: str = "gemini-2.5-flash") -> dict:
    prompt = build_prompt(trace, attention, memories, affect, include_base_rate)
    resp = call_llm(model, prompt, system=SYSTEM_PROMPT_ML, max_tokens=400)
    parsed = parse_response(resp["text"])
    parsed["_cost"] = resp.get("cost", 0.0)
    return parsed


def run_pipeline_ml(user_ids, include_base_rate: bool, model: str = "gemini-2.5-flash"):
    traces = behavioral_trace_ml(user_ids, cutoff=T_TEST_CUTOFF)
    out = {}
    for uid in user_ids:
        if uid not in traces:
            out[uid] = {"error": "no_trace", "stated_intent_prob": 0.5}
            continue
        t = traces[uid]
        a = rank_attention(t)
        m = retrieve_memories(t)
        af = compute_affect(t)
        delib = deliberate_ml(t, a, m, af, include_base_rate=include_base_rate, model=model)
        dec = blend_decision(delib, af, t)
        out[uid] = {
            "verbatim_reaction": delib["verbatim_reaction"],
            "stated_intent_prob": delib["stated_intent_prob"],
            "stated_intent_prob_raw": delib["stated_intent_prob"],
            "stated_intent_prob_final": dec["stated_intent_prob_final"],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["F-base", "F-nobase"], required=True)
    args = ap.parse_args()
    test = pl.read_parquet(ML_SPLITS / "test.parquet")
    uids = test["userId"].to_list()
    print(f"[22/{args.arm}] running on n={len(uids)} MovieLens users", flush=True)
    t0 = time.time()
    out = run_pipeline_ml(uids, include_base_rate=(args.arm == "F-base"))
    elapsed = time.time() - t0
    print(f"[22/{args.arm}] done in {elapsed:.0f}s")

    scores = np.array([out[u].get("stated_intent_prob_raw", 0.5) for u in uids])
    scores_final = np.array([out[u].get("stated_intent_prob_final", 0.5) for u in uids])
    actual = test["label"].to_numpy()
    buckets = test["activity_bucket"].to_numpy()
    verbatims = [out[u].get("verbatim_reaction", "") for u in uids]

    np.savez(RESULTS / f"phase22_ml_{args.arm}_scores.npz",
             user_id=np.array(uids),
             stated_intent_raw=scores,
             stated_intent_final=scores_final,
             actual=actual,
             activity_bucket=buckets,
             verbatim=np.array(verbatims, dtype=object))

    m = all_metrics(actual, scores, B=500, seed=SEED)
    gap = float(scores.mean() - actual.mean())
    per_bucket = {}
    for b in sorted(set(buckets.tolist())):
        mask = buckets == b
        if mask.sum() < 10:
            continue
        per_bucket[b] = {"n": int(mask.sum()), "mean_stated": float(scores[mask].mean()),
                         "mean_actual": float(actual[mask].mean()),
                         "signed_gap": float(scores[mask].mean() - actual[mask].mean())}
    summary = {"arm": args.arm, "n": len(uids), "elapsed_s": elapsed,
               "mean_stated": float(scores.mean()), "mean_actual": float(actual.mean()),
               "signed_gap": gap, "pr_auc": m["pr_auc"], "brier": m["brier"],
               "per_bucket": per_bucket}
    (RESULTS / f"phase22_ml_{args.arm}_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[22/{args.arm}] mean_stated={scores.mean():.4f} actual={actual.mean():.4f} gap={gap:+.4f}")
    print(f"[22/{args.arm}] PR-AUC = {m['pr_auc']['point']:.4f} [{m['pr_auc']['lo']:.4f}, {m['pr_auc']['hi']:.4f}]")


if __name__ == "__main__":
    main()
