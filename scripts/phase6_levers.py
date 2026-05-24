"""Phase 6: Cost-accuracy Pareto + 3 levers.

L1: class-weighted/focal loss for LightGBM.
L2: stacked ensemble of all reps.
L3: D2 → D3 reflection ablation (run separately via phase4b --variant D3).

This script computes:
- Pareto frontier: $/prediction vs PR-AUC for each rep
- L1, L2 deltas (L3 is run if D3 file exists)
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from src import SEED, T_TRAIN_CUTOFF, T_TEST_CUTOFF
from src.splits import load_split
from src.features import rfm_features, bag_of_categories
from src.eval import pr_auc, paired_bootstrap_diff, bootstrap_ci

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# Approximate cost per prediction (USD)
# Tabular models: amortized GPU/CPU time @ $0.10/CPU-hour, ~1ms per prediction → ~$3e-8
# LLM: ~$0.0002 per call for gpt-4o-mini at our prompt size
COST_PER_PRED = {
    "majority": 1e-10,
    "recency": 1e-10,
    "A_logistic": 3e-8,
    "A_lgbm": 5e-8,
    "B_lgbm": 8e-8,
    "C_gru4rec": 3e-7,
    "C_sasrec": 5e-7,
    "C_bert4rec": 5e-7,
    "D_D1": 1e-4,
    "D_D2": 2e-4,
    "D_D3": 4e-4,
}


def lever_1_focal(train, val, test):
    """L1: class-weighted LGBM vs unweighted (using Rep B = bag-of-categories + RFM features)."""
    print("[6/L1] computing B features (focal-weighted LGBM)...", flush=True)
    rfm_train = rfm_features(train["customer_id"].to_list(), cutoff=T_TRAIN_CUTOFF)
    bag_train = bag_of_categories(train["customer_id"].to_list(), cutoff=T_TRAIN_CUTOFF)
    rfm_val = rfm_features(val["customer_id"].to_list(), cutoff=T_TRAIN_CUTOFF)
    bag_val = bag_of_categories(val["customer_id"].to_list(), cutoff=T_TRAIN_CUTOFF)
    rfm_test = rfm_features(test["customer_id"].to_list(), cutoff=T_TEST_CUTOFF)
    bag_test = bag_of_categories(test["customer_id"].to_list(), cutoff=T_TEST_CUTOFF)
    comb_train = rfm_train.join(bag_train, on="customer_id", how="left")
    comb_val = rfm_val.join(bag_val, on="customer_id", how="left")
    comb_test = rfm_test.join(bag_test, on="customer_id", how="left")

    def to_xy(split, feats):
        joined = split.join(feats, on="customer_id", how="left").fill_null(0)
        feature_cols = [c for c in feats.columns if c != "customer_id"]
        X = joined.select(feature_cols).to_numpy().astype(np.float32)
        y = joined["label"].to_numpy()
        return X, y

    X_tr, y_tr = to_xy(train, comb_train)
    X_va, y_va = to_xy(val, comb_val)
    X_te, y_te = to_xy(test, comb_test)

    res = {}
    for label, kwargs in [
        ("baseline_no_weight", {}),
        ("balanced", {"class_weight": "balanced"}),
        ("scale_pos_weight_4", {"scale_pos_weight": 4.0}),
    ]:
        model = lgb.LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=63,
            min_data_in_leaf=20, feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
            random_state=SEED, n_jobs=-1, verbose=-1, **kwargs,
        )
        # Fair protocol: same early-stopping as Phase 4a so the L1 baseline matches the head-to-head numbers
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(30, verbose=False)])
        p = model.predict_proba(X_te)[:, 1]
        m = bootstrap_ci({"y_true": y_te, "y_score": p}, lambda y_true, y_score: pr_auc(y_true, y_score), B=500, seed=SEED)
        res[label] = m
        print(f"[6/L1] {label}: PR-AUC = {m['point']:.4f}")
    return res


def lever_2_ensemble():
    """L2: stack reps via logistic meta-learner."""
    a = np.load(RESULTS / "phase4a_scores.npz", allow_pickle=True)
    y = a["y_test"]
    # Try with all classical reps; if D2 LLM scores exist on a subset, also include them
    classical_scores = np.column_stack([
        a["A_logistic"], a["A_lgbm"], a["B_lgbm"], a["C_gru4rec"], a["C_sasrec"], a["C_bert4rec"],
    ])
    # Split test into 2 halves: half to fit meta-learner, half to evaluate
    n = len(y)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    fit_idx = perm[: n // 2]
    eval_idx = perm[n // 2:]
    meta = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    meta.fit(classical_scores[fit_idx], y[fit_idx])
    p_ens = meta.predict_proba(classical_scores[eval_idx])[:, 1]
    m = bootstrap_ci({"y_true": y[eval_idx], "y_score": p_ens}, lambda y_true, y_score: pr_auc(y_true, y_score), B=300, seed=SEED)
    print(f"[6/L2] Stacked ensemble: PR-AUC = {m['point']:.4f}")
    # Compare against best individual rep on same eval_idx
    best = max(["A_logistic", "A_lgbm", "B_lgbm", "C_gru4rec", "C_sasrec", "C_bert4rec"], key=lambda n: pr_auc(y[eval_idx], a[n][eval_idx]))
    best_score = a[best][eval_idx]
    diff = paired_bootstrap_diff(y[eval_idx], p_ens, best_score, pr_auc, B=300, seed=SEED)
    print(f"[6/L2] Δ vs best ({best}): {diff['point']:+.4f} [{diff['lo']:.4f}, {diff['hi']:.4f}], p={diff['p']:.4f}")
    return {"ensemble": m, "best_individual": best, "delta_vs_best": diff}


def pareto():
    """Build cost-accuracy Pareto frontier."""
    a = np.load(RESULTS / "phase4a_scores.npz", allow_pickle=True)
    y = a["y_test"]
    rows = []
    for name in ["A_logistic", "A_lgbm", "B_lgbm", "C_gru4rec", "C_sasrec", "C_bert4rec"]:
        rows.append({"name": name, "pr_auc": pr_auc(y, a[name]), "cost_per_pred": COST_PER_PRED[name]})
    # Add LLM
    for variant in ["D1", "D2", "D3"]:
        p = RESULTS / f"phase4b_{variant}_scores.npz"
        if p.exists():
            d = np.load(p, allow_pickle=True)
            rows.append({"name": f"D_{variant}", "pr_auc": pr_auc(d["y_test"], d["scores"]), "cost_per_pred": COST_PER_PRED.get(f"D_{variant}", 2e-4)})

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in rows:
        ax.scatter(r["cost_per_pred"], r["pr_auc"], s=80)
        ax.annotate(r["name"], (r["cost_per_pred"], r["pr_auc"]), fontsize=8, alpha=0.8, xytext=(5, 5), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("$ per prediction (log)")
    ax.set_ylabel("PR-AUC")
    ax.set_title("Cost–accuracy Pareto frontier")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "phase6_pareto.png", dpi=130)
    plt.close()
    return rows


def main():
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")

    print("\n=== Lever 1: focal/balanced weighting ===")
    l1 = lever_1_focal(train, val, test)
    print("\n=== Lever 2: stacked ensemble ===")
    l2 = lever_2_ensemble()
    print("\n=== Pareto ===")
    par = pareto()

    out = {"lever1_focal": l1, "lever2_ensemble": l2, "pareto": par}
    (RESULTS / "phase6_levers.json").write_text(json.dumps(out, indent=2, default=str))
    print("[6] Done.")


if __name__ == "__main__":
    main()
