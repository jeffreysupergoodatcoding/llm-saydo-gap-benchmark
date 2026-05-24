"""Phase 24: Domain-specific human-self benchmark via test-retest autocorrelation.

For the H&M test pool, compute the same-customer label autocorrelation across
adjacent 30-day windows. Sheeran's r ≈ 0.53 is a cross-domain meta-analytic
comparator; this is the *within-H&M-domain* "human predicts their own next
month from their last month" benchmark. The comparison is much more
apples-to-apples than the Sheeran citation alone allows.

For each H&M customer with ≥1 transaction across (T-90, T-60), (T-60, T-30),
(T-30, T):
  - Build 3 binary indicators: bought_in_window[i] for i in 0..2
  - Compute autocorrelation: Pearson + Spearman between adjacent windows
  - This is the "human predicts the human" comparator: Park-style normalized-
    accuracy denominator.
"""

from __future__ import annotations
import json
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import polars as pl
from scipy.stats import spearmanr, pearsonr

from src import T_TEST_CUTOFF
from src.data import load_transactions
from src.splits import load_split

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
WINDOW = 30


def main():
    test = load_split("test")
    cids = test["customer_id"].to_list()
    cutoff = date.fromisoformat(T_TEST_CUTOFF)
    # Windows: w0 = (T-90, T-60), w1 = (T-60, T-30), w2 = (T-30, T)
    windows = [
        ("w0_T-90_T-60", cutoff - timedelta(days=90), cutoff - timedelta(days=60)),
        ("w1_T-60_T-30", cutoff - timedelta(days=60), cutoff - timedelta(days=30)),
        ("w2_T-30_T", cutoff - timedelta(days=30), cutoff),
    ]
    indicators = {}
    for name, start, end in windows:
        tx = (load_transactions()
              .filter((pl.col("customer_id").is_in(cids))
                       & (pl.col("t_dat") >= start) & (pl.col("t_dat") < end))
              .group_by("customer_id").agg(pl.len().alias("n"))
              .collect())
        lookup = dict(zip(tx["customer_id"].to_list(), tx["n"].to_list()))
        ind = np.array([1 if lookup.get(c, 0) > 0 else 0 for c in cids], dtype=int)
        indicators[name] = ind

    out = {"windows": [w[0] for w in windows], "n_customers": len(cids), "per_window_rate": {}}
    for name, ind in indicators.items():
        out["per_window_rate"][name] = float(ind.mean())

    # Pairwise correlations
    out["pairwise_corr"] = {}
    keys = list(indicators.keys())
    for i in range(len(keys) - 1):
        a, b = keys[i], keys[i + 1]
        pr, pp = pearsonr(indicators[a], indicators[b])
        sr, sp = spearmanr(indicators[a], indicators[b])
        out["pairwise_corr"][f"{a}__vs__{b}"] = {
            "pearson_r": float(pr), "pearson_p": float(pp),
            "spearman_rho": float(sr), "spearman_p": float(sp),
        }

    # Same customer, all three windows → 1-vs-(2,3) "predicts next month from past 2"
    last_two_avg = (indicators["w0_T-90_T-60"] + indicators["w1_T-60_T-30"]) / 2.0
    pr, pp = pearsonr(last_two_avg, indicators["w2_T-30_T"])
    sr, sp = spearmanr(last_two_avg, indicators["w2_T-30_T"])
    out["past_two_predicts_current"] = {
        "pearson_r": float(pr), "pearson_p": float(pp),
        "spearman_rho": float(sr), "spearman_p": float(sp),
        "method": "use avg of two preceding 30-day window indicators as the 'human-self stated intent' analog"
    }

    # Sheeran ref
    out["sheeran_2002_meta_r"] = 0.53
    out["interpretation"] = (
        "The within-H&M test-retest correlation (a customer's buying in past 30 days predicting their buying in "
        "next 30 days) is the apples-to-apples 'human self predicts human self' benchmark. "
        "Sheeran's r ≈ 0.53 is for stated intent vs subsequent behavior across health/voting/exercise domains; the "
        "within-domain analog reported here lets us compare the LLM's within-bucket Spearman (0.22-0.28 in §4.3.1) "
        "to a same-task numerical comparator."
    )

    (RESULTS / "phase24_human_baseline.json").write_text(json.dumps(out, indent=2, default=str))
    print("=== Domain-specific human-self benchmark (H&M test pool) ===")
    for name, rate in out["per_window_rate"].items():
        print(f"  {name}: rate = {rate:.3f}")
    print("\nAdjacent-window correlations:")
    for k, v in out["pairwise_corr"].items():
        print(f"  {k}: Pearson={v['pearson_r']:+.3f}  Spearman={v['spearman_rho']:+.3f}")
    p2 = out["past_two_predicts_current"]
    print(f"\nPast-two-windows avg → current-window:  Pearson={p2['pearson_r']:+.3f}  Spearman={p2['spearman_rho']:+.3f}")
    print(f"\nSheeran 2002 cross-domain meta r = {out['sheeran_2002_meta_r']}")


if __name__ == "__main__":
    main()
