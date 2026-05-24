"""Phase 40: prepare a Claude Code subagent cross-provider sandbox arm.

This script builds the input JSON for a Claude Code subagent to read and process.
The subagent will simulate the M1 zero-shot sandbox protocol on 100 stratified
customers (20 per bucket × 5 buckets) and emit predictions.

Output:
  results/phase40_claude_batch_input.json  — 100 customer traces + menus
  results/phase40_claude_predictions.jsonl — to be filled by the subagent

The subagent prompt (in PROMPT below) is what we pass to Agent(prompt=...) from
the parent Claude Code session. This file just preps the inputs.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import polars as pl
from datetime import date

from src import T_TEST_CUTOFF
from src.behavioral_trace import behavioral_trace
from src.sandbox.env import generate_stimulus_menu, render_state_for_prompt
from src.sandbox.env import SandboxState


ROOT = Path(__file__).resolve().parents[1]
OUT_INPUT = ROOT / "results" / "phase40_claude_batch_input.json"
N_PER_BUCKET = 20


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
    print(f"Traces ready: {len(traces)}")

    batch = []
    for cid in cids:
        if cid not in traces:
            continue
        trace = traces[cid]
        # Build the 4-week stimulus menus deterministically with the same seed function
        weekly_menus = []
        for w in range(4):
            menu = generate_stimulus_menu(cid, w, trace)
            weekly_menus.append([
                {"label": ["A", "B", "C"][i], "product_type": c.product_type,
                 "garment_group": c.garment_group, "colour": c.colour,
                 "section": c.section, "in_or_out": c.label}
                for i, c in enumerate(menu)
            ])
        # Render a compact pre-window trace summary
        state = SandboxState(customer_id=cid, week_t=0, trace_snapshot=trace)
        trace_text = render_state_for_prompt(state, trace)
        batch.append({
            "customer_id": cid,
            "bucket": buckets[cid],
            "actual_label": labels[cid],  # for evaluation only, NOT for the LLM
            "trace_text": trace_text,
            "weekly_menus": weekly_menus,
        })

    OUT_INPUT.write_text(json.dumps(batch, indent=2))
    print(f"Wrote {OUT_INPUT} with {len(batch)} customers")
    print(f"Next step: spawn Claude Code Agent to read this file and emit "
          f"results/phase40_claude_predictions.jsonl")


if __name__ == "__main__":
    main()
