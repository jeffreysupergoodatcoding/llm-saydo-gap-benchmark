"""Phase 38: build pools needed for the added M2 / M7 methods.

Outputs:
  results/phase38_neighbours_random.json  — M2: 5 random val customers per core customer (seed=2026)
  results/phase38_lgbm_preds.json         — M7: per-core-1000-customer LGBM prediction
                                            from existing Phase 4a tabular model.

The LGBM scores are reused unchanged from `phase4a_tabular_scores.npz`.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import polars as pl


ROOT = Path(__file__).resolve().parents[1]


def main():
    core = pl.read_parquet(ROOT / "results" / "phase31_core1000_v3.parquet")
    core_ids = set(core["customer_id"].to_list())
    print(f"core-1000 v3: {len(core_ids)} customer_ids")

    # ---- M2: random 5-customer ICL ----
    val = pl.read_parquet(ROOT / "data" / "splits" / "val.parquet")
    pool_dicts = json.loads((ROOT / "results" / "phase33_history_pool.json").read_text())
    rng = np.random.default_rng(2026)
    random_neighbours = {}
    pool_idx = np.arange(len(pool_dicts))
    for cid in core_ids:
        sel = rng.choice(pool_idx, size=5, replace=False)
        random_neighbours[cid] = [pool_dicts[i] for i in sel.tolist()]
    out_r = ROOT / "results" / "phase38_neighbours_random.json"
    out_r.write_text(json.dumps(random_neighbours, indent=2))
    print(f"Wrote {out_r}  (n={len(random_neighbours)})")

    # ---- M7: LightGBM preds for core-1000 ----
    # Phase 4a was trained on the v2 train pool; predicting on v3 core-1000 should
    # work because train/val/test are all customer-disjoint and the model is
    # population-trained.
    data = np.load(ROOT / "results" / "phase4a_tabular_scores.npz", allow_pickle=True)
    print(f"phase4a_tabular_scores keys: {list(data.keys())}")
    # Find lightgbm scores
    if "lightgbm" in data:
        lgbm_test_scores = data["lightgbm"]
    else:
        # Iterate keys looking for lgbm
        candidates = [k for k in data.keys() if 'lgbm' in k.lower() or 'lightgbm' in k.lower()]
        if candidates:
            lgbm_test_scores = data[candidates[0]]
            print(f"  using key: {candidates[0]}")
        else:
            print("  no lgbm key found — using first non-customer_id key")
            cand = [k for k in data.keys() if k != "customer_id" and k != "labels"]
            lgbm_test_scores = data[cand[0]]

    test_ids = data["customer_id"].tolist()
    id_to_score = dict(zip(test_ids, lgbm_test_scores.tolist() if hasattr(lgbm_test_scores, 'tolist') else lgbm_test_scores))
    core_preds = {cid: float(id_to_score.get(cid, 0.21)) for cid in core_ids}
    n_found = sum(1 for v in core_preds.values() if v != 0.21)
    print(f"LGBM preds: {n_found}/{len(core_preds)} core customers had a prediction "
          f"(remaining default to test rate 0.21)")
    out_l = ROOT / "results" / "phase38_lgbm_preds.json"
    out_l.write_text(json.dumps(core_preds, indent=2))
    print(f"Wrote {out_l}")


if __name__ == "__main__":
    main()
