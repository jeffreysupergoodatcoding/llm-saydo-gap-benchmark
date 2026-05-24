"""Phase 21: H9 sensitivity with a disjoint embedder (BAAI/bge-large-en-v1.5).

Addresses Iteration-4 blind-reviewer Blocker #2: the Gemini-vendor embedder
used in Phase 11b is co-trained with the Gemini LLM that produced the verbatims,
which is a real validity threat for H9. Pre-registration v2 called for a
disjoint third-party embedder. We re-run the H9a within-bucket permutation test
on the SAME verbatim/article texts using bge-large.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from src import SEED

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if (na > 0 and nb > 0) else 0.0


def main():
    verb = json.loads((RESULTS / "phase11_verbatim.json").read_text())
    verbatims = verb["all_verbatims"]
    # We need the actual-next-article descriptions and the distractor pool.
    # These were embedded earlier but not stored as text; re-derive.
    from src.splits import load_split
    from src import T_TEST_CUTOFF
    from scripts.phase11_verbatim import _next_purchases_after_cutoff, _distractor_pool_by_bucket

    # Reconstruct text inputs for the same eligible customer set.
    p = RESULTS / "phase10_F-nobase_scores.npz"
    d = np.load(p, allow_pickle=True)
    cids = list(d["customer_id"])
    actual = d["actual"].astype(int)
    cust_verbatim_full = list(d["verbatim"])
    buckets_full = d["activity_bucket"].astype(str)

    positive_cids = [cids[i] for i in range(len(cids)) if actual[i] == 1]
    positive_verbatim = [cust_verbatim_full[i] for i in range(len(cids)) if actual[i] == 1]
    positive_buckets = [buckets_full[i] for i in range(len(cids)) if actual[i] == 1]

    next_purchases = _next_purchases_after_cutoff(positive_cids, T_TEST_CUTOFF)
    distractor_pool = _distractor_pool_by_bucket(T_TEST_CUTOFF, k_per_bucket=100)

    eligible = []
    for cid, ver, bk in zip(positive_cids, positive_verbatim, positive_buckets):
        if cid in next_purchases and ver and len(ver) > 10:
            eligible.append({"cid": cid, "verbatim": ver, "actual_text": next_purchases[cid], "bucket": bk})
    print(f"[21] eligible customers = {len(eligible)}")

    print("[21] loading bge-large-en-v1.5 (downloads ~1.3 GB on first use)...", flush=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    verb_texts = [e["verbatim"] for e in eligible]
    act_texts = [e["actual_text"] for e in eligible]
    art_texts = [a["text"] for a in distractor_pool]

    print(f"[21] embedding {len(verb_texts)} verbatims + {len(act_texts)} actual + {len(art_texts)} distractors...", flush=True)
    v_emb = model.encode(verb_texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    a_emb = model.encode(act_texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    p_emb = model.encode(art_texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)

    # cos_actual per customer (since normalized, dot = cosine)
    cos_actuals = np.einsum("ij,ij->i", v_emb, a_emb)
    # mean cos to distractors per customer
    cos_dist = v_emb @ p_emb.T
    mean_cos_dist = cos_dist.mean(axis=1)
    # MRR vs 101-item ranking
    mrrs = []
    for i in range(len(eligible)):
        scores = np.concatenate([[cos_actuals[i]], cos_dist[i]])
        order = np.argsort(-scores)
        rank = int(np.where(order == 0)[0][0]) + 1
        mrrs.append(1.0 / rank)
    mrrs = np.array(mrrs)

    # Within-bucket permutation null on cos_actual
    rng = np.random.default_rng(SEED)
    bucket_to_indices = {}
    for i, e in enumerate(eligible):
        bucket_to_indices.setdefault(e["bucket"], []).append(i)
    n_perm = 5000
    null_means = []
    for _ in range(n_perm):
        cos_perm = []
        for b, idxs in bucket_to_indices.items():
            if len(idxs) < 2:
                for k in idxs:
                    cos_perm.append(float(cos_actuals[k]))
                continue
            perm = rng.permutation(len(idxs))
            for j, k in enumerate(idxs):
                cos_perm.append(float(np.dot(v_emb[k], a_emb[idxs[perm[j]]])))
        null_means.append(float(np.mean(cos_perm)))
    null_means = np.array(null_means)
    observed = float(cos_actuals.mean())
    p_one = float((null_means >= observed).mean())

    paired_diffs = cos_actuals - mean_cos_dist
    n_pair = len(paired_diffs)
    se = float(np.std(paired_diffs, ddof=1)) / np.sqrt(n_pair) if n_pair >= 2 else 0.01
    ci_half = 1.96 * se
    diff_mean = float(paired_diffs.mean())
    n_items = len(distractor_pool) + 1
    chance_mrr = float(sum(1 / r for r in range(1, n_items + 1)) / n_items)

    out = {
        "embedder": "BAAI/bge-large-en-v1.5",
        "n_eligible": len(eligible),
        "H9a_mean_cos_actual_bge": observed,
        "H9a_mean_cos_perm_null_bge": float(null_means.mean()),
        "H9a_diff_bge": observed - float(null_means.mean()),
        "H9a_perm_p_bge": p_one,
        "paired_diff_cos_actual_minus_dist_bge_mean": diff_mean,
        "paired_diff_95CI_bge": [diff_mean - ci_half, diff_mean + ci_half],
        "H9b_MRR_bge": float(mrrs.mean()),
        "H9b_chance_MRR": chance_mrr,
        "H9b_margin_bge": float(mrrs.mean() - chance_mrr),
        "n_perm": n_perm,
        "interpretation": (
            "If the BGE-large (disjoint embedder) result agrees with the Gemini-embedder Phase 11b "
            "result (H9a practically null, H9b at or below chance), the negative H9 finding is "
            "robust to embedder-vendor co-training."
        ),
    }
    (RESULTS / "phase21_h9_bge_sensitivity.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[21] BGE H9a: cos_actual={observed:.4f}  perm null={null_means.mean():.4f}  diff={observed-null_means.mean():+.4f}  p={p_one:.4f}")
    print(f"[21] BGE paired diff (cos_actual - mean_cos_dist): {diff_mean:+.4f}  95% CI [{diff_mean-ci_half:+.4f}, {diff_mean+ci_half:+.4f}]")
    print(f"[21] BGE MRR={mrrs.mean():.4f}  chance={chance_mrr:.4f}  margin={mrrs.mean()-chance_mrr:+.4f}")


if __name__ == "__main__":
    main()
