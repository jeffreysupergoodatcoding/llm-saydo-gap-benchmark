"""Phase 26: MovieLens within-domain human-self benchmark.

Same protocol as Phase 24 (H&M test-retest) — three adjacent 30-day windows
ending at the cutoff; binary indicator = any rating in window; report Pearson
+ Spearman autocorrelation.
"""

from __future__ import annotations
import json
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import polars as pl
from scipy.stats import pearsonr, spearmanr

from src.movielens_data import T_TEST_CUTOFF, load_ratings, SPLITS

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main():
    test = pl.read_parquet(SPLITS / "test.parquet")
    uids = test["userId"].to_list()
    cutoff = date.fromisoformat(T_TEST_CUTOFF)
    windows = [
        ("w0_T-90_T-60", cutoff - timedelta(days=90), cutoff - timedelta(days=60)),
        ("w1_T-60_T-30", cutoff - timedelta(days=60), cutoff - timedelta(days=30)),
        ("w2_T-30_T", cutoff - timedelta(days=30), cutoff),
    ]
    indicators = {}
    for name, start, end in windows:
        rt = (load_ratings()
              .filter((pl.col("userId").is_in(uids))
                       & (pl.col("t_dat") >= start) & (pl.col("t_dat") < end))
              .group_by("userId").agg(pl.len().alias("n"))
              .collect())
        lookup = dict(zip(rt["userId"].to_list(), rt["n"].to_list()))
        ind = np.array([1 if lookup.get(u, 0) > 0 else 0 for u in uids], dtype=int)
        indicators[name] = ind

    out = {"domain": "MovieLens", "windows": [w[0] for w in windows],
           "n_users": len(uids), "per_window_rate": {n: float(ind.mean()) for n, ind in indicators.items()}}

    out["pairwise_corr"] = {}
    keys = list(indicators.keys())
    for i in range(len(keys) - 1):
        a, b = keys[i], keys[i + 1]
        # Need at least one variant value for correlations
        try:
            pr, pp = pearsonr(indicators[a], indicators[b])
            sr, sp = spearmanr(indicators[a], indicators[b])
            out["pairwise_corr"][f"{a}__vs__{b}"] = {
                "pearson_r": float(pr), "pearson_p": float(pp),
                "spearman_rho": float(sr), "spearman_p": float(sp),
            }
        except Exception as e:
            out["pairwise_corr"][f"{a}__vs__{b}"] = {"error": str(e)}

    last_two_avg = (indicators["w0_T-90_T-60"] + indicators["w1_T-60_T-30"]) / 2.0
    try:
        pr, pp = pearsonr(last_two_avg, indicators["w2_T-30_T"])
        sr, sp = spearmanr(last_two_avg, indicators["w2_T-30_T"])
        out["past_two_predicts_current"] = {
            "pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_rho": float(sr), "spearman_p": float(sp),
        }
    except Exception as e:
        out["past_two_predicts_current"] = {"error": str(e)}

    (RESULTS / "phase26_ml_human_baseline.json").write_text(json.dumps(out, indent=2, default=str))
    print("=== MovieLens within-domain human-self benchmark ===")
    for n, rate in out["per_window_rate"].items():
        print(f"  {n}: rate = {rate:.3f}")
    print("Adjacent-window correlations:")
    for k, v in out["pairwise_corr"].items():
        if "error" in v:
            print(f"  {k}: {v['error']}")
        else:
            print(f"  {k}: Pearson={v['pearson_r']:+.3f}  Spearman={v['spearman_rho']:+.3f}")
    p2 = out.get("past_two_predicts_current", {})
    if "error" not in p2:
        print(f"Past-2-windows avg → current:  Pearson={p2['pearson_r']:+.3f}  Spearman={p2['spearman_rho']:+.3f}")


if __name__ == "__main__":
    main()
