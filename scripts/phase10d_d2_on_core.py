"""Phase 10d: D2 flat prompt on the SAME core-1000 customers as F-base / F-nobase.

This brings the H7 paired test to full n=1000 (instead of ~107 chance-overlap
between F-nobase and the original D2 5000-customer sample). Pure statistical
power addition; no architectural change.

Uses the same Gemini 2.5 Flash + the same flat narrative prompt as the v1 D2 arm.
"""

from __future__ import annotations
import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import polars as pl

from src import T_TEST_CUTOFF, SEED
from src.splits import load_split
from src.features import behavioral_narrative
from src.llm_client import call_llm
from src.eval import all_metrics

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

SYSTEM_PROMPT = (
    "You are an expert at predicting customer purchase behavior. "
    "Given a behavioral profile, you output a single probability between 0 and 1 "
    "representing the chance the customer will make at least one purchase in the next 30 days. "
    "Output a single line of raw JSON with one field 'p' (a float between 0 and 1). "
    "Example output: {\"p\": 0.43}. "
    "Do NOT use code fences, do NOT add commentary, do NOT include any other text."
)


def build_prompt(narrative: str) -> str:
    return (
        f"{narrative}\n\n"
        f"Will this customer make any purchase in the next 30 days? "
        f"Output JSON: {{\"p\": <float 0-1>}}"
    )


def parse_p(text: str) -> float:
    if not text:
        return 0.5
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    try:
        return float(json.loads(s)["p"])
    except Exception:
        pass
    m = re.search(r'"?p"?\s*:\s*([0-9.]+)', text)
    if m:
        return float(m.group(1))
    m = re.search(r"\b(0?\.\d+|1\.0+|0|1)\b", text)
    if m:
        return max(0.0, min(1.0, float(m.group(1))))
    return 0.5


def stratified_core_sample(test_df: pl.DataFrame, per_bucket: int, seed: int = SEED) -> pl.DataFrame:
    """SAME function as phase10_arms.py to guarantee identical core-1000."""
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
    ap.add_argument("--per-bucket", type=int, default=200)
    ap.add_argument("--model", default="gemini-2.5-flash")
    args = ap.parse_args()

    test = load_split("test")
    core = stratified_core_sample(test, per_bucket=args.per_bucket, seed=SEED)
    cids = core["customer_id"].to_list()
    print(f"[10d] D2-flat on the same core-{len(core)} customers", flush=True)

    print("[10d] building narratives...", flush=True)
    narratives = behavioral_narrative(cids, cutoff=T_TEST_CUTOFF, n_recent=20)
    print(f"[10d] built {len(narratives)} narratives", flush=True)

    scores = np.zeros(len(cids))
    t0 = time.time()
    for i, cid in enumerate(cids):
        narr = narratives.get(cid, "")
        if not narr:
            scores[i] = 0.5
            continue
        prompt = build_prompt(narr)
        try:
            resp = call_llm(args.model, prompt, system=SYSTEM_PROMPT, max_tokens=80)
            scores[i] = parse_p(resp["text"])
        except Exception as e:
            print(f"[10d] error on {cid}: {e}", flush=True)
            scores[i] = 0.5
        if (i + 1) % 200 == 0:
            print(f"[10d] {i+1}/{len(cids)}  ({(i+1)/(time.time()-t0):.1f}/sec)", flush=True)

    actual = core["label"].to_numpy()
    buckets_arr = core["activity_bucket"].to_numpy()
    n_tx_arr = core["n_tx_pre_cutoff"].to_numpy()

    np.savez(RESULTS / "phase10_D2-core_scores.npz",
             customer_id=np.array(cids),
             stated_intent=scores,
             stated_intent_raw=scores,   # flat-prompt has no guardrail
             actual=actual,
             activity_bucket=buckets_arr,
             n_tx_pre_cutoff=n_tx_arr,
             verbatim=np.array(["" for _ in cids], dtype=object),
             key_objection=np.array(["" for _ in cids], dtype=object),
             reasoning=np.array(["" for _ in cids], dtype=object),
             confidence=np.full(len(cids), 50.0),
             friction=np.full(len(cids), 50.0))

    mean_stated = float(scores.mean())
    mean_actual = float(actual.mean())
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
        "arm": "D2-core",
        "model": args.model,
        "n": len(cids),
        "elapsed_s": time.time() - t0,
        "mean_stated_intent_prob": mean_stated,
        "mean_actual_label": mean_actual,
        "signed_gap_E_stated_minus_E_actual": mean_stated - mean_actual,
        "pr_auc": m["pr_auc"], "brier": m["brier"], "ece": m["ece"],
        "wasserstein_decile": m["wasserstein_decile"],
        "under_dispersion": m["under_dispersion"],
        "per_bucket_gap": per_bucket,
    }
    (RESULTS / "phase10_D2-core_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[10d] mean_stated={mean_stated:.4f} actual={mean_actual:.4f} gap={mean_stated-mean_actual:+.4f}")
    print(f"[10d] PR-AUC = {m['pr_auc']['point']:.4f} [{m['pr_auc']['lo']:.4f}, {m['pr_auc']['hi']:.4f}]")


if __name__ == "__main__":
    main()
