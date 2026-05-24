"""Phase 4b: LLM digital twin (Rep D).

For a stratified 5000-customer subsample of test:
  1. Generate behavioral narrative
  2. Send to gpt-4o-mini with a Park-2024-style prompt
  3. Extract P(repeat purchase 30d) from response

Supports 4 ablation conditions D1..D4 (controlled by --variant flag).
"""

from __future__ import annotations
import argparse
import json
import re
import time
from pathlib import Path
import numpy as np
import polars as pl

from src import SEED, T_TEST_CUTOFF
from src.splits import load_split
from src.features import behavioral_narrative
from src.llm_client import call_llm, remaining_budget_usd
call_openai = call_llm  # backwards-compat alias
from src.eval import all_metrics

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


SYSTEM_PROMPT = (
    "You are an expert at predicting customer purchase behavior. "
    "Given a behavioral profile, you output a single probability between 0 and 1 "
    "representing the chance the customer will make at least one purchase in the next 30 days. "
    "Output a single line of raw JSON with one field 'p' (a float between 0 and 1). "
    "Example output: {\"p\": 0.43}. "
    "Do NOT use code fences, do NOT add commentary, do NOT include any other text."
)


def build_prompt_d1(narrative: str) -> str:
    """D1: raw events only (no summary stats, no reflection)."""
    # Strip out the summary lines, keep only "RECENT PURCHASES"
    parts = narrative.split("RECENT PURCHASES")
    if len(parts) > 1:
        # also keep age from CUSTOMER PROFILE
        head = parts[0]
        age_line = next((l for l in head.splitlines() if "age" in l.lower()), "")
        return (
            f"{age_line}\nRECENT PURCHASES{parts[1]}\n\n"
            f"Will this customer make any purchase in the next 30 days? "
            f"Output JSON: {{\"p\": <float 0-1>}}"
        )
    return narrative


def build_prompt_d2(narrative: str) -> str:
    """D2: D1 + summary stats (the full narrative)."""
    return (
        f"{narrative}\n\n"
        f"Will this customer make any purchase in the next 30 days? "
        f"Output JSON: {{\"p\": <float 0-1>}}"
    )


def build_prompt_d3(narrative: str, reflection: str) -> str:
    """D3: D2 + reflection step output included."""
    return (
        f"{narrative}\n\n"
        f"Inferred shopper persona: {reflection}\n\n"
        f"Given the inferred persona and behavior, will this customer make any purchase in the next 30 days? "
        f"Output JSON: {{\"p\": <float 0-1>}}"
    )


def build_reflection_prompt(narrative: str) -> str:
    return (
        f"{narrative}\n\n"
        f"In one sentence, describe what kind of shopper this person is "
        f"(e.g., 'bargain-hunter, infrequent, seasonal' or 'loyal heavy buyer of basics'). "
        f"Just output the description, nothing else."
    )


def parse_p(text: str) -> float:
    """Parse {p:..} from response. Strips code fences. Falls back to regex."""
    if not text:
        return 0.5
    # Strip code fences if present
    s = text.strip()
    if s.startswith("```"):
        # remove first and last fenced lines
        lines = s.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    # Try strict JSON
    try:
        obj = json.loads(s)
        return float(obj["p"])
    except Exception:
        pass
    # Try to extract any JSON-like {"p": ...}
    m = re.search(r'"?p"?\s*:\s*([0-9.]+)', text)
    if m:
        return float(m.group(1))
    # Try to find a stand-alone number
    m = re.search(r"\b(0?\.\d+|1\.0+|0|1)\b", text)
    if m:
        v = float(m.group(1))
        return max(0.0, min(1.0, v))
    return 0.5


def score_one(narrative: str, variant: str, model: str = "gpt-4o-mini") -> tuple[float, dict]:
    if variant == "D1":
        prompt = build_prompt_d1(narrative)
        resp = call_openai(model, prompt, system=SYSTEM_PROMPT, max_tokens=80)
        p = parse_p(resp["text"])
        return p, {"used_tokens": resp["in"] + resp["out"], "cost": resp["cost"], "reflection": None}

    if variant == "D2":
        prompt = build_prompt_d2(narrative)
        resp = call_openai(model, prompt, system=SYSTEM_PROMPT, max_tokens=80)
        p = parse_p(resp["text"])
        return p, {"used_tokens": resp["in"] + resp["out"], "cost": resp["cost"], "reflection": None}

    if variant == "D3":
        ref_resp = call_openai(model, build_reflection_prompt(narrative), system=None, max_tokens=80, temperature=0.0)
        reflection = (ref_resp["text"] or "").strip()
        prompt = build_prompt_d3(narrative, reflection)
        resp = call_openai(model, prompt, system=SYSTEM_PROMPT, max_tokens=80)
        p = parse_p(resp["text"])
        return p, {"used_tokens": ref_resp["in"] + ref_resp["out"] + resp["in"] + resp["out"], "cost": ref_resp["cost"] + resp["cost"], "reflection": reflection}

    raise ValueError(variant)


def stratified_llm_subsample(test_df: pl.DataFrame, n: int, seed: int = SEED) -> pl.DataFrame:
    """Sample n customers stratified by activity bucket."""
    rng = np.random.default_rng(seed)
    buckets = test_df["activity_bucket"].unique().to_list()
    per = max(1, n // len(buckets))
    parts = []
    for b in buckets:
        sub = test_df.filter(pl.col("activity_bucket") == b)
        k = min(per, len(sub))
        idx = rng.choice(len(sub), size=k, replace=False)
        parts.append(sub[idx.tolist()])
    out = pl.concat(parts)
    if len(out) > n:
        idx = rng.choice(len(out), size=n, replace=False)
        out = out[idx.tolist()]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["D1", "D2", "D3"], default="D2")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--model", default="gemini-2.5-flash")
    args = ap.parse_args()

    test = load_split("test")
    sub = stratified_llm_subsample(test, args.n)
    print(f"[4b/{args.variant}] subsample n = {len(sub)}", flush=True)

    sub_ids = sub["customer_id"].to_list()
    print("[4b] building narratives...", flush=True)
    narratives = behavioral_narrative(sub_ids, cutoff=T_TEST_CUTOFF, n_recent=20)
    print(f"[4b] built {len(narratives)} narratives. budget remaining ${remaining_budget_usd():.2f}", flush=True)

    scores = np.zeros(len(sub_ids))
    meta_log = []
    cost_total = 0.0
    start = time.time()
    for i, cid in enumerate(sub_ids):
        narr = narratives.get(cid, "")
        if not narr:
            scores[i] = 0.5
            continue
        try:
            p, meta = score_one(narr, args.variant, model=args.model)
        except Exception as e:
            print(f"[4b] error on {cid}: {e}", flush=True)
            p = 0.5
            meta = {"error": str(e), "cost": 0.0}
        scores[i] = p
        cost_total += meta.get("cost", 0.0)
        meta_log.append({"cid": cid, "p": p, **{k: v for k, v in meta.items() if k != "reflection"}})
        if i < 3:
            print(f"[4b] sample {i}: cid={cid[:8]}... p={p:.3f}", flush=True)
        if (i + 1) % 200 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / max(elapsed, 1e-6)
            print(f"[4b] {i+1}/{len(sub_ids)}  ({rate:.1f}/sec)  budget=${remaining_budget_usd():.2f}", flush=True)

    y = sub["label"].to_numpy()
    m = all_metrics(y, scores, B=500, seed=SEED)
    print(f"[4b/{args.variant}] PR-AUC = {m['pr_auc']['point']:.4f}  [{m['pr_auc']['lo']:.4f}, {m['pr_auc']['hi']:.4f}]", flush=True)

    out_path = RESULTS / f"phase4b_{args.variant}.json"
    out_path.write_text(json.dumps({
        "variant": args.variant,
        "n": len(sub_ids),
        "model": args.model,
        "metrics": m,
        "budget_remaining_usd": remaining_budget_usd(),
    }, indent=2, default=str))

    np.savez(RESULTS / f"phase4b_{args.variant}_scores.npz",
             customer_id=np.array(sub_ids),
             y_test=y,
             scores=scores,
             activity_bucket=np.array(sub["activity_bucket"].to_list()),
             n_tx_pre_cutoff=sub["n_tx_pre_cutoff"].to_numpy())
    print("[4b] Done.")


if __name__ == "__main__":
    main()
