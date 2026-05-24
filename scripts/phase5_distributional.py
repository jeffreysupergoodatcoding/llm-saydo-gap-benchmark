"""Phase 5: Distributional metrics + Park-style normalized accuracy + error analysis.

Loads phase4 scores, computes per-rep:
- All standard metrics (PR-AUC, ROC, Brier, ECE, Wasserstein-1 decile, under-dispersion)
- Park normalized accuracy via adjacent-window test-retest
- Per-segment Spearman ρ
"""

from __future__ import annotations
import json
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from src import SEED, T_TRAIN_CUTOFF, T_TEST_CUTOFF
from src.data import load_transactions
from src.splits import load_split
from src.eval import all_metrics, park_normalized_accuracy, under_dispersion, wasserstein_decile, calibration_curve

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def compute_prev_window_labels(customer_ids: list[str], test_cutoff: str) -> np.ndarray:
    """For each customer, compute 1[purchased in [T-60, T-30)] as the test-retest analog."""
    cutoff = date.fromisoformat(test_cutoff)
    prev_start = cutoff - timedelta(days=60)
    prev_end = cutoff - timedelta(days=30)
    tx = (
        load_transactions()
        .filter((pl.col("customer_id").is_in(customer_ids)) & (pl.col("t_dat") >= prev_start) & (pl.col("t_dat") < prev_end))
        .group_by("customer_id")
        .agg(pl.len().alias("n"))
        .collect()
    )
    s = dict(zip(tx["customer_id"].to_list(), tx["n"].to_list()))
    out = np.array([1 if s.get(c, 0) > 0 else 0 for c in customer_ids], dtype=int)
    return out


def main():
    test = load_split("test")

    # Load Phase 4a scores
    a = np.load(RESULTS / "phase4a_scores.npz", allow_pickle=True)
    cids = a["customer_id"]
    y = a["y_test"]
    buckets = a["activity_bucket"]

    classical = {
        "A_logistic": a["A_logistic"],
        "A_lgbm": a["A_lgbm"],
        "B_lgbm": a["B_lgbm"],
        "C_gru4rec": a["C_gru4rec"],
        "C_sasrec": a["C_sasrec"],
        "C_bert4rec": a["C_bert4rec"],
    }

    # Park normalized accuracy: need prev-window labels
    print("[5] computing previous-window labels for test-retest analog...", flush=True)
    label_prev = compute_prev_window_labels(list(cids), T_TEST_CUTOFF)
    print(f"[5] prev-window positive rate: {label_prev.mean():.3f}", flush=True)

    out = {}
    for name, score in classical.items():
        m = all_metrics(y, score, B=500, seed=SEED)
        m["park_normalized_accuracy"] = park_normalized_accuracy(y, label_prev, score)
        m["under_dispersion"] = under_dispersion(y, score)
        m["wasserstein_decile"] = wasserstein_decile(y, score)
        # Per-bucket Spearman
        per_bucket_rho = {}
        for b in sorted(set(buckets.tolist())):
            mask = buckets == b
            if mask.sum() < 30:
                continue
            r, p = spearmanr(score[mask], y[mask])
            per_bucket_rho[b] = {"rho": float(r), "p": float(p), "n": int(mask.sum())}
        m["per_bucket_spearman"] = per_bucket_rho
        out[name] = m

    # Load LLM rep if present
    p4b = RESULTS / "phase4b_D2_scores.npz"
    if p4b.exists():
        d = np.load(p4b, allow_pickle=True)
        d_cids = d["customer_id"]
        d_y = d["y_test"]
        d_scores = d["scores"]
        # prev labels for LLM customers
        d_prev = compute_prev_window_labels(list(d_cids), T_TEST_CUTOFF)
        m = all_metrics(d_y, d_scores, B=500, seed=SEED)
        m["park_normalized_accuracy"] = park_normalized_accuracy(d_y, d_prev, d_scores)
        m["under_dispersion"] = under_dispersion(d_y, d_scores)
        m["wasserstein_decile"] = wasserstein_decile(d_y, d_scores)
        out["D_D2"] = m

    # Save
    (RESULTS / "phase5_metrics.json").write_text(json.dumps(out, indent=2, default=str))

    # H3 / H4 summary
    print("\n[5] Pre-registered hypotheses:")
    print(f"H3 (under-dispersion: Var(pred)/Var(obs) < 1):")
    for name, m in out.items():
        ud = m["under_dispersion"]
        ok = ud["ratio"] < 1
        print(f"  {name}: ratio = {ud['ratio']:.3f}  (Levene p = {ud['levene_p']:.2e})  {'PASS' if ok else 'FAIL'}")
    print(f"\nH4 (Wasserstein vs PR-AUC rank inversion):")
    names = list(out.keys())
    pr_aucs = {n: out[n]["pr_auc"]["point"] for n in names}
    wassers = {n: out[n]["wasserstein_decile"] for n in names}
    # Best PR-AUC and best Wasserstein (lower=better)
    pr_rank = sorted(names, key=lambda n: -pr_aucs[n])
    ws_rank = sorted(names, key=lambda n: wassers[n])
    print(f"  PR-AUC rank: {pr_rank}")
    print(f"  Wasserstein rank (best first, lower=better): {ws_rank}")
    # Spearman between PR-AUC and -wasserstein
    from scipy.stats import spearmanr as sp
    if len(names) >= 2:
        rho, p = sp([pr_aucs[n] for n in names], [-wassers[n] for n in names])
        print(f"  Spearman ρ(PR-AUC, -Wasserstein) = {rho:.3f}, p = {p:.3f}  (H4 expects negative)")

    # Calibration plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, m in out.items():
        cc = m["calibration_curve"]
        if not cc:
            continue
        axes[0].plot([c["pred_mean"] for c in cc], [c["actual_rate"] for c in cc], "o-", label=name, alpha=0.7)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_xlabel("Predicted P")
    axes[0].set_ylabel("Actual rate")
    axes[0].set_title("Calibration (reliability diagram, 10 bins)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Under-dispersion bar plot
    names_l = list(out.keys())
    ratios = [out[n]["under_dispersion"]["ratio"] for n in names_l]
    axes[1].barh(names_l, ratios)
    axes[1].axvline(1.0, color="r", linestyle="--")
    axes[1].set_xlabel("Var(pred) / Var(observed)")
    axes[1].set_title("Under-dispersion (ratio < 1 means under-dispersed)")
    plt.tight_layout()
    plt.savefig(RESULTS / "phase5_calibration.png", dpi=130)
    plt.close()

    print("\n[5] Done.")


if __name__ == "__main__":
    main()
