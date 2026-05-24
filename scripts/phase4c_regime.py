"""Phase 4c: Regime analysis (activity bucket × history length) head-to-head win-map.

Combines:
- Phase 4a scores (full 46k test, reps A_logistic, A_lgbm, B_lgbm, C_gru4rec, C_sasrec, C_bert4rec)
- Phase 4b scores (5k LLM subsample, Rep D_D2)

For overall comparison: report PR-AUC, CI, win-map.
For per-bucket comparison: compute PR-AUC per (activity_bucket × n_tx_pre_cutoff_bin).
For Rep D vs classical: restrict to the LLM subsample so comparison is fair.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from src import SEED
from src.eval import pr_auc, bootstrap_ci, paired_bootstrap_diff, holm_bonferroni

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

BUCKETS_ORDER = ["1", "2-5", "6-20", "21-100", "101+"]


def main():
    # Load Phase 4a scores
    a = np.load(RESULTS / "phase4a_scores.npz", allow_pickle=True)
    cid_a = a["customer_id"]
    y_a = a["y_test"]
    buckets_a = a["activity_bucket"]
    classical_scores = {
        "A_logistic": a["A_logistic"],
        "A_lgbm": a["A_lgbm"],
        "B_lgbm": a["B_lgbm"],
        "C_gru4rec": a["C_gru4rec"],
        "C_sasrec": a["C_sasrec"],
        "C_bert4rec": a["C_bert4rec"],
    }

    # Try to load Phase 4b (LLM) — D2 primary, D3 if exists
    llm_scores = {}
    for variant in ["D2", "D3", "D1"]:
        p = RESULTS / f"phase4b_{variant}_scores.npz"
        if p.exists():
            d = np.load(p, allow_pickle=True)
            llm_scores[variant] = {
                "customer_id": d["customer_id"],
                "y_test": d["y_test"],
                "scores": d["scores"],
                "activity_bucket": d["activity_bucket"],
            }
            print(f"[4c] loaded LLM variant {variant}: n={len(d['scores'])}")

    cid_to_idx = {cid: i for i, cid in enumerate(cid_a)}

    out = {"overall": {}, "by_bucket": {}, "by_history_bin": {}, "pairwise": {}}

    # ----- Overall metrics (full 46k test) -----
    print("\n[4c] Overall metrics on full test (46k):")
    for name, score in classical_scores.items():
        m = bootstrap_ci({"y_true": y_a, "y_score": score}, lambda y_true, y_score: pr_auc(y_true, y_score), B=500, seed=SEED)
        out["overall"][name] = m
        print(f"  {name}: PR-AUC {m['point']:.4f} [{m['lo']:.4f}, {m['hi']:.4f}]")

    # ----- LLM rep on its 5k subsample, restricted to overlap with 4a -----
    aligned_llm = {}
    aligned_classical_on_llm = {}
    for variant, d in llm_scores.items():
        # Align LLM customers with phase4a (same customers since they're a subset of test)
        sub_cids = d["customer_id"]
        sub_y = d["y_test"]
        # Index into phase4a arrays
        idx = np.array([cid_to_idx[c] for c in sub_cids if c in cid_to_idx])
        sub_mask = np.array([c in cid_to_idx for c in sub_cids])
        aligned_classical_on_llm[variant] = {name: s[idx] for name, s in classical_scores.items()}
        aligned_classical_on_llm[variant]["y"] = y_a[idx]
        aligned_classical_on_llm[variant]["bucket"] = buckets_a[idx]
        aligned_llm[variant] = {
            "scores": d["scores"][sub_mask],
            "y": sub_y[sub_mask],
            "bucket": d["activity_bucket"][sub_mask],
        }
        m = bootstrap_ci({"y_true": aligned_llm[variant]["y"], "y_score": aligned_llm[variant]["scores"]}, lambda y_true, y_score: pr_auc(y_true, y_score), B=500, seed=SEED)
        out["overall"][f"D_{variant}"] = m
        print(f"  D_{variant}: PR-AUC {m['point']:.4f} [{m['lo']:.4f}, {m['hi']:.4f}] (n={len(aligned_llm[variant]['scores'])})")

        # Also recompute classical models PR-AUC on the same 5k subset for fair comparison
        print(f"  Classical reps on D_{variant}'s 5k subsample:")
        for name, sub_s in aligned_classical_on_llm[variant].items():
            if name in ("y", "bucket"):
                continue
            mc = bootstrap_ci({"y_true": aligned_classical_on_llm[variant]["y"], "y_score": sub_s}, lambda y_true, y_score: pr_auc(y_true, y_score), B=500, seed=SEED)
            print(f"    {name}: {mc['point']:.4f} [{mc['lo']:.4f}, {mc['hi']:.4f}]")
            out["overall"][f"{name}_on_{variant}_sub"] = mc

    # ----- Pairwise paired bootstrap on full 46k (classical only) -----
    print("\n[4c] Pairwise paired bootstrap (classical, full 46k):")
    names = list(classical_scores.keys())
    diffs = {}
    p_values = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ni, nj = names[i], names[j]
            d = paired_bootstrap_diff(y_a, classical_scores[ni], classical_scores[nj], pr_auc, B=500, seed=SEED)
            diffs[f"{ni}_vs_{nj}"] = d
            p_values.append(d["p"])
            print(f"  {ni} - {nj}: Δ={d['point']:+.4f}, p={d['p']:.4f}")
    hb = holm_bonferroni(p_values, alpha=0.05)
    out["pairwise"] = {"diffs": {k: v for k, v in diffs.items()}, "holm_bonferroni": hb}

    # ----- LLM vs best classical (on LLM subsample) -----
    if "D2" in aligned_llm:
        print("\n[4c] LLM D2 vs classical (on D2's 5k subsample):")
        for name, s in aligned_classical_on_llm["D2"].items():
            if name in ("y", "bucket"):
                continue
            d = paired_bootstrap_diff(aligned_llm["D2"]["y"], aligned_llm["D2"]["scores"], s, pr_auc, B=500, seed=SEED)
            print(f"  D2 - {name}: Δ={d['point']:+.4f}, p={d['p']:.4f}")
            out["pairwise"][f"D2_vs_{name}_on_5k"] = d

    # ----- Per-bucket analysis -----
    print("\n[4c] Per-activity-bucket PR-AUC (classical, full 46k):")
    per_bucket = {b: {} for b in BUCKETS_ORDER}
    for b in BUCKETS_ORDER:
        mask = buckets_a == b
        if mask.sum() < 30:
            continue
        for name, s in classical_scores.items():
            try:
                m = pr_auc(y_a[mask], s[mask])
                per_bucket[b][name] = {"pr_auc": m, "n": int(mask.sum())}
            except Exception:
                per_bucket[b][name] = {"pr_auc": float("nan"), "n": int(mask.sum())}
        print(f"  bucket={b} (n={mask.sum()}): " + ", ".join(f"{k}={v['pr_auc']:.3f}" for k, v in per_bucket[b].items()))
    # Add LLM
    if "D2" in aligned_llm:
        print("\n[4c] Per-activity-bucket PR-AUC (LLM D2 + classical on 5k subsample):")
        for b in BUCKETS_ORDER:
            mask = aligned_llm["D2"]["bucket"] == b
            if mask.sum() < 30:
                continue
            entries = per_bucket.get(b, {})
            entries[f"D_D2"] = {"pr_auc": float(pr_auc(aligned_llm["D2"]["y"][mask], aligned_llm["D2"]["scores"][mask])), "n": int(mask.sum())}
            for name, s in aligned_classical_on_llm["D2"].items():
                if name in ("y", "bucket"):
                    continue
                entries[f"{name}_on_5k"] = {"pr_auc": float(pr_auc(aligned_classical_on_llm["D2"]["y"][mask], s[mask])), "n": int(mask.sum())}
            per_bucket[b] = entries
            print(f"  bucket={b} (n={mask.sum()}): D_D2={entries['D_D2']['pr_auc']:.3f}, B_lgbm={entries.get('B_lgbm_on_5k',{}).get('pr_auc',float('nan')):.3f}, C_sasrec={entries.get('C_sasrec_on_5k',{}).get('pr_auc',float('nan')):.3f}")
    out["by_bucket"] = per_bucket

    # Win-map figure: per-bucket bar chart of best classical vs LLM D2
    if "D2" in aligned_llm:
        labels = []
        d2_vals = []
        best_classical_vals = []
        best_classical_names = []
        for b in BUCKETS_ORDER:
            entries = per_bucket.get(b, {})
            if f"D_D2" not in entries:
                continue
            labels.append(b)
            d2_vals.append(entries["D_D2"]["pr_auc"])
            classical_keys = [k for k in entries if k.endswith("_on_5k")]
            classical_vals = [(k, entries[k]["pr_auc"]) for k in classical_keys]
            classical_vals.sort(key=lambda x: -x[1])
            if classical_vals:
                best_classical_names.append(classical_vals[0][0])
                best_classical_vals.append(classical_vals[0][1])
            else:
                best_classical_names.append("?")
                best_classical_vals.append(float("nan"))

        x = np.arange(len(labels))
        w = 0.35
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - w / 2, d2_vals, w, label="LLM Digital Twin (D2)")
        ax.bar(x + w / 2, best_classical_vals, w, label="Best classical")
        for xi, name in zip(x, best_classical_names):
            ax.text(xi + w / 2, 0.02, name.replace("_on_5k", ""), rotation=90, fontsize=8, ha="center", va="bottom", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Activity bucket (pre-cutoff tx count)")
        ax.set_ylabel("PR-AUC")
        ax.set_title("Per-regime: LLM Digital Twin vs Best Classical (test 5k stratified subsample)")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS / "phase4c_regime_winmap.png", dpi=130)
        plt.close()

    # History-length bins (n_tx_pre_cutoff bins): finer-grain regime
    n_tx = a["n_tx_pre_cutoff"]
    history_bins = [(1, 1, "1"), (2, 4, "2-4"), (5, 10, "5-10"), (11, 25, "11-25"), (26, 60, "26-60"), (61, 200, "61-200"), (201, 10**8, "201+")]
    by_hist = {}
    print("\n[4c] Per-history-bin PR-AUC (classical, full 46k):")
    for lo, hi, lbl in history_bins:
        mask = (n_tx >= lo) & (n_tx <= hi)
        if mask.sum() < 30:
            continue
        entries = {}
        for name, s in classical_scores.items():
            try:
                entries[name] = {"pr_auc": float(pr_auc(y_a[mask], s[mask])), "n": int(mask.sum())}
            except Exception:
                entries[name] = {"pr_auc": float("nan"), "n": int(mask.sum())}
        by_hist[lbl] = entries
        print(f"  hist={lbl} (n={mask.sum()}): " + ", ".join(f"{k.split('_')[0]}={v['pr_auc']:.3f}" for k, v in entries.items()))

    # Add LLM
    if "D2" in aligned_llm:
        n_tx_llm = np.array([n_tx[cid_to_idx[c]] if c in cid_to_idx else 0 for c in llm_scores["D2"]["customer_id"]])
        for lo, hi, lbl in history_bins:
            mask = (n_tx_llm >= lo) & (n_tx_llm <= hi)
            if mask.sum() < 30:
                continue
            entries = by_hist.get(lbl, {})
            entries["D_D2"] = {"pr_auc": float(pr_auc(llm_scores["D2"]["y_test"][mask], llm_scores["D2"]["scores"][mask])), "n": int(mask.sum())}
            by_hist[lbl] = entries

    out["by_history_bin"] = by_hist

    (RESULTS / "phase4c_regime.json").write_text(json.dumps(out, indent=2, default=str))
    print("\n[4c] Done.")


if __name__ == "__main__":
    main()
