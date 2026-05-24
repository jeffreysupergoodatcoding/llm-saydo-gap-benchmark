"""Phase 4a-extension: 5-seed variance bounds for sequence models.

Trains GRU4Rec, SASRec, BERT4Rec under 5 seeds each, reports PR-AUC mean ± std.
Required by pre-registration for the headline sequence-model claims.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import polars as pl
import torch

from src import T_TRAIN_CUTOFF, T_TEST_CUTOFF
from src.splits import load_split
from src.features import event_sequences
from src.models.sequence import fit_sequence_model, predict_sequence
from src.eval import pr_auc

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

SEEDS = [42, 123, 7, 2024, 999]
EPOCHS = 5
MAX_LEN = 64


def main():
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")

    train_ids = train["customer_id"].to_list()
    val_ids = val["customer_id"].to_list()
    test_ids = test["customer_id"].to_list()

    print("[4a-var] building sequences...")
    seq_train = event_sequences(train_ids, cutoff=T_TRAIN_CUTOFF, max_len=MAX_LEN)
    seq_val = event_sequences(val_ids, cutoff=T_TRAIN_CUTOFF, max_len=MAX_LEN)
    seq_test = event_sequences(test_ids, cutoff=T_TEST_CUTOFF, max_len=MAX_LEN)

    all_items = set()
    for d in (seq_train, seq_val):
        for cust_id, seq in d.items():
            for it, _, _ in seq:
                all_items.add(int(it))
    item2id = {iid: i + 1 for i, iid in enumerate(sorted(all_items))}
    n_items = len(item2id)
    print(f"[4a-var] item vocab = {n_items}")

    def to_id_seq(d, ids):
        seqs = []
        for cid in ids:
            seq = d.get(cid, [])
            seqs.append([item2id.get(int(it), 0) for it, _, _ in seq])
        return seqs

    train_seqs = to_id_seq(seq_train, train_ids)
    val_seqs = to_id_seq(seq_val, val_ids)
    test_seqs = to_id_seq(seq_test, test_ids)
    y_train_seq = train["label"].to_numpy().tolist()
    y_val_seq = val["label"].to_numpy().tolist()
    y_test = test["label"].to_numpy()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[4a-var] device = {device}")

    results = {}
    for model_name in ["gru4rec", "sasrec", "bert4rec"]:
        per_seed_prauc = []
        for seed in SEEDS:
            print(f"[4a-var] {model_name} seed={seed}", flush=True)
            model = fit_sequence_model(
                model_name, train_seqs, y_train_seq, val_seqs, y_val_seq,
                n_items=n_items, seed=seed, epochs=EPOCHS, max_len=MAX_LEN, device=device,
            )
            p = predict_sequence(model, test_seqs, max_len=MAX_LEN, device=device)
            pa = pr_auc(y_test, p)
            per_seed_prauc.append(pa)
            print(f"  PR-AUC = {pa:.4f}")
        arr = np.array(per_seed_prauc)
        results[model_name] = {
            "seeds": SEEDS,
            "pr_auc": per_seed_prauc,
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
        print(f"  {model_name}: mean = {arr.mean():.4f} ± {arr.std():.4f}")
        # Persist after each model so a crash doesn't lose work
        (RESULTS / "phase4a_seed_variance.json").write_text(json.dumps(results, indent=2, default=str))

    (RESULTS / "phase4a_seed_variance.json").write_text(json.dumps(results, indent=2, default=str))
    print("[4a-var] Done.")


if __name__ == "__main__":
    main()
