"""Phase 4a: Train and evaluate reps A (RFM), B (bag-of-categories), C (sequence models).

Saves per-customer scores so Phase 4c can do regime analysis across all reps.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import polars as pl
import torch

from src import SEED, T_TRAIN_CUTOFF, T_TEST_CUTOFF
from src.splits import load_split
from src.features import rfm_features, bag_of_categories, event_sequences
from src.models.tabular import train_logistic, train_lgbm, predict_proba
from src.models.sequence import fit_sequence_model, predict_sequence
from src.eval import all_metrics, pr_auc

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def to_xy(split_df: pl.DataFrame, feat_df: pl.DataFrame):
    joined = split_df.join(feat_df, on="customer_id", how="left").fill_null(0)
    feature_cols = [c for c in feat_df.columns if c != "customer_id"]
    X = joined.select(feature_cols).to_numpy().astype(np.float32)
    y = joined["label"].to_numpy()
    cids = joined["customer_id"].to_list()
    return X, y, cids, feature_cols


def main():
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")

    out: dict = {}

    # ----- Rep A: RFM -----
    print("[4a] Rep A: RFM features...", flush=True)
    rfm_train = rfm_features(train["customer_id"].to_list(), cutoff=T_TRAIN_CUTOFF)
    rfm_val = rfm_features(val["customer_id"].to_list(), cutoff=T_TRAIN_CUTOFF)
    rfm_test = rfm_features(test["customer_id"].to_list(), cutoff=T_TEST_CUTOFF)

    X_tr, y_tr, _, _ = to_xy(train, rfm_train)
    X_va, y_va, _, _ = to_xy(val, rfm_val)
    X_te, y_te, cids_te, _ = to_xy(test, rfm_test)

    print("[4a] A-logistic...", flush=True)
    lr = train_logistic(X_tr, y_tr, seed=SEED)
    p_a_lr = predict_proba(lr, X_te)
    print("[4a] A-lgbm...", flush=True)
    gb = train_lgbm(X_tr, y_tr, X_va, y_va, seed=SEED, n_estimators=400)
    p_a_gb = predict_proba(gb, X_te)

    out["A_logistic"] = all_metrics(y_te, p_a_lr, B=500, seed=SEED)
    out["A_lgbm"] = all_metrics(y_te, p_a_gb, B=500, seed=SEED)
    print(f"[4a] A_logistic PR-AUC = {out['A_logistic']['pr_auc']['point']:.4f}")
    print(f"[4a] A_lgbm     PR-AUC = {out['A_lgbm']['pr_auc']['point']:.4f}")

    # ----- Rep B: Bag-of-categories + RFM (richer features) -----
    print("[4a] Rep B: bag-of-categories...", flush=True)
    bag_train = bag_of_categories(train["customer_id"].to_list(), cutoff=T_TRAIN_CUTOFF)
    bag_val = bag_of_categories(val["customer_id"].to_list(), cutoff=T_TRAIN_CUTOFF)
    bag_test = bag_of_categories(test["customer_id"].to_list(), cutoff=T_TEST_CUTOFF)

    # combine bag + RFM
    comb_train = rfm_train.join(bag_train, on="customer_id", how="left")
    comb_val = rfm_val.join(bag_val, on="customer_id", how="left")
    comb_test = rfm_test.join(bag_test, on="customer_id", how="left")

    X_tr, y_tr, _, _ = to_xy(train, comb_train)
    X_va, y_va, _, _ = to_xy(val, comb_val)
    X_te, y_te, _, _ = to_xy(test, comb_test)
    print(f"[4a] B feature matrix: {X_tr.shape}", flush=True)

    print("[4a] B-lgbm...", flush=True)
    gb = train_lgbm(X_tr, y_tr, X_va, y_va, seed=SEED, n_estimators=600)
    p_b_gb = predict_proba(gb, X_te)
    out["B_lgbm"] = all_metrics(y_te, p_b_gb, B=500, seed=SEED)
    print(f"[4a] B_lgbm PR-AUC = {out['B_lgbm']['pr_auc']['point']:.4f}")

    # Checkpoint after tabular reps in case sequence models fail
    np.savez(RESULTS / "phase4a_tabular_scores.npz",
             customer_id=np.array(cids_te),
             y_test=y_te,
             A_logistic=p_a_lr,
             A_lgbm=p_a_gb,
             B_lgbm=p_b_gb,
             activity_bucket=np.array(test["activity_bucket"].to_list()),
             n_tx_pre_cutoff=test["n_tx_pre_cutoff"].to_numpy())
    (RESULTS / "phase4a_tabular_metrics.json").write_text(json.dumps({k: v for k, v in out.items() if k.startswith(("A_", "B_"))}, indent=2, default=str))
    print("[4a] Tabular checkpoint saved.")

    # ----- Rep C: Sequence models -----
    print("[4a] Rep C: building sequences...", flush=True)
    seq_train = event_sequences(train["customer_id"].to_list(), cutoff=T_TRAIN_CUTOFF, max_len=64)
    seq_val = event_sequences(val["customer_id"].to_list(), cutoff=T_TRAIN_CUTOFF, max_len=64)
    seq_test = event_sequences(test["customer_id"].to_list(), cutoff=T_TEST_CUTOFF, max_len=64)

    # Build item vocab from train+val
    all_items = set()
    for d in (seq_train, seq_val):
        for cust_id, seq in d.items():
            for it, _, _ in seq:
                all_items.add(int(it))
    item2id = {iid: i + 1 for i, iid in enumerate(sorted(all_items))}  # 0 = pad
    n_items = len(item2id)
    print(f"[4a] item vocab = {n_items}", flush=True)

    def to_id_seq(d, ids):
        seqs = []
        for cid in ids:
            seq = d.get(cid, [])
            seqs.append([item2id.get(int(it), 0) for it, _, _ in seq])
        return seqs

    train_ids = train["customer_id"].to_list()
    val_ids = val["customer_id"].to_list()
    test_ids_l = test["customer_id"].to_list()

    train_seqs = to_id_seq(seq_train, train_ids)
    val_seqs = to_id_seq(seq_val, val_ids)
    test_seqs = to_id_seq(seq_test, test_ids_l)
    y_train_seq = train["label"].to_numpy().tolist()
    y_val_seq = val["label"].to_numpy().tolist()
    y_test_seq = test["label"].to_numpy()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[4a] device = {device}", flush=True)

    seq_scores = {}
    for model_name in ["gru4rec", "sasrec", "bert4rec"]:
        print(f"[4a] training {model_name}...", flush=True)
        model = fit_sequence_model(
            model_name, train_seqs, y_train_seq, val_seqs, y_val_seq,
            n_items=n_items, seed=SEED, epochs=5, max_len=64, device=device,
        )
        p = predict_sequence(model, test_seqs, max_len=64, device=device)
        seq_scores[model_name] = p
        out[f"C_{model_name}"] = all_metrics(y_test_seq, p, B=500, seed=SEED)
        print(f"[4a] C_{model_name} PR-AUC = {out[f'C_{model_name}']['pr_auc']['point']:.4f}")

    # Save all scores for downstream phases
    np.savez(RESULTS / "phase4a_scores.npz",
             customer_id=np.array(cids_te),
             y_test=y_te,
             A_logistic=p_a_lr,
             A_lgbm=p_a_gb,
             B_lgbm=p_b_gb,
             C_gru4rec=seq_scores["gru4rec"],
             C_sasrec=seq_scores["sasrec"],
             C_bert4rec=seq_scores["bert4rec"],
             activity_bucket=np.array(test["activity_bucket"].to_list()),
             n_tx_pre_cutoff=test["n_tx_pre_cutoff"].to_numpy())

    (RESULTS / "phase4a_metrics.json").write_text(json.dumps(out, indent=2, default=str))
    print("[4a] Done.")


if __name__ == "__main__":
    main()
