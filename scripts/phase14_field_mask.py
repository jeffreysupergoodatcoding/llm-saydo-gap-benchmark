"""Phase 14: field-masking ablation.

For a 50-customer subsample, rerun F-nobase deliberation under 4 conditions:
(A) full trace
(B) demographics masked (age, postal_region removed)
(C) recent purchases masked (recent_purchases list replaced with "[redacted]")
(D) personality + derived_flags masked
(E) product summary masked (top_section/top_garment_group/etc. removed)

Report mean |Δ stated_intent_prob| per condition. Whichever drops the most
attribution to the trace tells us which input the LLM was using.

Output: results/phase14_field_mask.json
"""

from __future__ import annotations
import json
import copy
from pathlib import Path
import numpy as np

from src import T_TEST_CUTOFF, SEED
from src.behavioral_trace import behavioral_trace
from src.cognition_fragment.attention import rank_attention
from src.cognition_fragment.memory import retrieve_memories
from src.cognition_fragment.affect import compute_affect
from src.cognition_fragment.deliberation import build_prompt, parse_response, SYSTEM_PROMPT_BASE
from src.llm_client import call_llm

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

N_CUSTOMERS = 50


def _mask(trace: dict, condition: str) -> dict:
    t = copy.deepcopy(trace)
    if condition == "mask_demographics":
        t["identity"]["age"] = None
    elif condition == "mask_recent_purchases":
        t["recent_purchases"] = []
    elif condition == "mask_personality":
        t["personality"] = "redacted"
        t["derived_flags"] = {k: False for k in t["derived_flags"]}
    elif condition == "mask_product_summary":
        for k in t["product_summary"]:
            if isinstance(t["product_summary"][k], str):
                t["product_summary"][k] = None
    return t


def _score(t, include_base_rate: bool, nonce: str, model: str = "gemini-2.5-flash"):
    attention = rank_attention(t)
    memories = retrieve_memories(t)
    affect = compute_affect(t)
    prompt = build_prompt(t, attention, memories, affect, include_base_rate)
    busted = prompt + f"\n\n<!-- nonce {nonce} -->"
    resp = call_llm(model, busted, system=SYSTEM_PROMPT_BASE, max_tokens=400)
    return parse_response(resp["text"])


def main():
    p = RESULTS / "phase10_F-nobase_scores.npz"
    if not p.exists():
        raise SystemExit("Run Phase 10 F-nobase first.")
    d = np.load(p, allow_pickle=True)
    cids = list(d["customer_id"])
    rng = np.random.default_rng(SEED + 1)
    sample = [cids[i] for i in rng.permutation(len(cids))[:N_CUSTOMERS]]

    print(f"[14] computing traces for {len(sample)} customers")
    traces = behavioral_trace(sample, cutoff=T_TEST_CUTOFF)

    rows = []
    conditions = ["full", "mask_demographics", "mask_recent_purchases", "mask_personality",
                  "mask_product_summary"]
    for i, cid in enumerate(sample):
        if cid not in traces:
            continue
        per_condition = {}
        for cond in conditions:
            t_use = traces[cid] if cond == "full" else _mask(traces[cid], cond)
            parsed = _score(t_use, include_base_rate=False, nonce=f"{cond}_{cid[:6]}")
            per_condition[cond] = float(parsed["stated_intent_prob"])
        rows.append({"cid": cid[:12] + "…", **per_condition})
        if i < 3:
            print(f"  {cid[:8]}…: {per_condition}")

    deltas = {cond: [] for cond in conditions if cond != "full"}
    for r in rows:
        for cond in deltas:
            deltas[cond].append(abs(r[cond] - r["full"]))

    out = {
        "n_customers": len(rows),
        "conditions": conditions,
        "mean_abs_delta_vs_full": {cond: float(np.mean(deltas[cond])) for cond in deltas},
        "median_abs_delta_vs_full": {cond: float(np.median(deltas[cond])) for cond in deltas},
        "samples": rows[:20],
    }
    (RESULTS / "phase14_field_mask.json").write_text(json.dumps(out, indent=2, default=str))
    print("\n[14] Mean |Δ stated_intent| when each field is masked (larger = field was load-bearing):")
    for cond, m in sorted(out["mean_abs_delta_vs_full"].items(), key=lambda kv: -kv[1]):
        print(f"  {cond:<30} → mean |Δ| = {m:.4f}")


if __name__ == "__main__":
    main()
