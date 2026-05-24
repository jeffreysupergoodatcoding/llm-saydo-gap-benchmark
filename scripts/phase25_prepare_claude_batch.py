"""Phase 25-prep: build a batch JSON file of 50 stratified H&M customer narratives.
The companion Phase 25 step (Claude Code subagent invocation, fired from the
parent Claude Code session) will score all 50 in a single agent call and
return a JSON array of {customer_id, p}.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import polars as pl

from src import T_TEST_CUTOFF, SEED
from src.features import behavioral_narrative

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
N_BATCH = 50


def main():
    # Pull the existing H&M F-base/F-nobase 1000-customer core (commit guarantees same seed).
    fb_npz = np.load(RESULTS / "phase10_F-base_scores.npz", allow_pickle=True)
    cids = list(fb_npz["customer_id"])
    buckets = list(fb_npz["activity_bucket"])
    actual = fb_npz["actual"]

    # Stratified subsample: 10 per bucket × 5 buckets = 50.
    rng = np.random.default_rng(SEED + 100)
    by_bucket = {}
    for i, b in enumerate(buckets):
        by_bucket.setdefault(b, []).append(i)
    sample_idx = []
    for b, idxs in by_bucket.items():
        k = min(10, len(idxs))
        chosen = rng.choice(idxs, size=k, replace=False)
        sample_idx.extend(chosen.tolist())
    sample_cids = [cids[i] for i in sample_idx]
    print(f"[25-prep] {len(sample_cids)} customers across {len(by_bucket)} buckets")

    print("[25-prep] building narratives...")
    narratives = behavioral_narrative(sample_cids, cutoff=T_TEST_CUTOFF, n_recent=15)
    batch = []
    for i, cid in enumerate(sample_cids):
        narr = narratives.get(cid, "")
        if not narr:
            continue
        batch.append({"slot": i, "cid_short": cid[:12] + "…", "cid": cid,
                      "bucket": buckets[sample_idx[i]] if i < len(sample_idx) else "?",
                      "actual": int(actual[sample_idx[i]]),
                      "narrative": narr})

    out_path = RESULTS / "phase25_claude_batch_input.json"
    out_path.write_text(json.dumps(batch, indent=2, default=str))
    print(f"[25-prep] wrote {out_path}; {len(batch)} customers ready for Claude-subagent scoring")


if __name__ == "__main__":
    main()
