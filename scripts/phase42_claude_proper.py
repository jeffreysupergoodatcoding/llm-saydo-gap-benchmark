"""Phase 42: prep leakage-free input + dispatch infrastructure for proper per-DP
Claude sandbox arm.

Differences from phase40:
1. `actual_label` is REDACTED from the input file the agent reads — strict leakage
   prevention. The agent literally cannot use the ground truth.
2. The agent prompt forbids writing Python heuristic scripts. Each DP decision
   must come from the agent's own LLM reasoning over the customer's trace + state.
3. Output to phase42_claude_proper_M1.jsonl (etc), not phase40.

Output:
  results/phase42_claude_proper_input.json  — n=200 customers, NO actual_label
  results/phase42_actual_labels.json         — sidecar with cid -> actual, for analysis only
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import polars as pl
from datetime import date

from src import T_TEST_CUTOFF
from src.behavioral_trace import behavioral_trace
from src.sandbox.env import generate_stimulus_menu
from src.sandbox.env import SandboxState
from src.sandbox.env import render_state_for_prompt


ROOT = Path(__file__).resolve().parents[1]
OUT_INPUT = ROOT / "results" / "phase42_claude_proper_input.json"
OUT_LABELS = ROOT / "results" / "phase42_actual_labels.json"
N_PER_BUCKET = 40  # 200 total for the per-DP arm (matches seed-sensitivity sub-sample size)


def main():
    core = pl.read_parquet(ROOT / "results" / "phase31_core1000_v3.parquet")
    rng = np.random.default_rng(2026)
    parts = []
    for b in ["1", "2-5", "6-20", "21-100", "101+"]:
        sub = core.filter(pl.col("activity_bucket") == b)
        idx = rng.choice(len(sub), size=N_PER_BUCKET, replace=False)
        parts.append(sub[idx.tolist()])
    sub = pl.concat(parts)
    cids = sub["customer_id"].to_list()
    labels = dict(zip(cids, sub["label"].to_list()))
    buckets = dict(zip(cids, sub["activity_bucket"].to_list()))
    print(f"Sample n={len(cids)}, label_rate={sum(labels.values())/len(labels):.3f}")

    traces = behavioral_trace(cids, cutoff=date.fromisoformat(T_TEST_CUTOFF))

    batch = []
    for cid in cids:
        if cid not in traces:
            continue
        trace = traces[cid]
        weekly_menus = []
        for w in range(4):
            menu = generate_stimulus_menu(cid, w, trace)
            weekly_menus.append([
                {"label": ["A", "B", "C"][i], "product_type": c.product_type,
                 "garment_group": c.garment_group, "colour": c.colour,
                 "section": c.section, "in_or_out": c.label}
                for i, c in enumerate(menu)
            ])
        state = SandboxState(customer_id=cid, week_t=0, trace_snapshot=trace)
        trace_text = render_state_for_prompt(state, trace)
        # CRITICAL: NO actual_label in batch entries; only customer_id + trace + menus + bucket
        batch.append({
            "customer_id": cid,
            "bucket": buckets[cid],
            "trace_text": trace_text,
            "weekly_menus": weekly_menus,
        })

    OUT_INPUT.write_text(json.dumps(batch, indent=2))
    OUT_LABELS.write_text(json.dumps({c: labels[c] for c in cids if c in traces}, indent=2))
    print(f"Wrote leakage-free input: {OUT_INPUT}  ({len(batch)} customers)")
    print(f"Wrote sidecar labels:     {OUT_LABELS}  (for analysis only)")
    print(f"\nNext: split into batches and dispatch agents that DO PER-DP REASONING.")

    # Split into 8 batches of 25 for parallel dispatch
    bs = 25
    n_batches = math.ceil(len(batch) / bs)
    for i in range(n_batches):
        sub_b = batch[i*bs:(i+1)*bs]
        fn = ROOT / "results" / f"phase42_claude_proper_batch_{i}.json"
        fn.write_text(json.dumps(sub_b, indent=2))
        print(f"  Batch {i}: {len(sub_b)} customers -> {fn.name}")


if __name__ == "__main__":
    main()
