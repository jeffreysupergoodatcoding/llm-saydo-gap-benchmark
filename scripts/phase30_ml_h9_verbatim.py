"""Phase 30: MovieLens H9 verbatim coherence.

For users with actual=1 in F-nobase ML arm, test cosine(verbatim, actual-next-movie)
vs within-bucket permutation null. Uses BGE-large (disjoint embedder).
"""

from __future__ import annotations
import json
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import polars as pl

from src import SEED
from src.movielens_data import T_TEST_CUTOFF, load_ratings, load_movies

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if (na > 0 and nb > 0) else 0.0


def main():
    d = np.load(RESULTS / "phase22_ml_F-nobase_scores.npz", allow_pickle=True)
    uids = list(d["user_id"])
    actual = d["actual"].astype(int)
    verbatim_all = list(d["verbatim"])
    buckets_all = d["activity_bucket"].astype(str)
    positive_idx = [i for i in range(len(uids)) if actual[i] == 1]
    pos_uids = [int(uids[i]) for i in positive_idx]
    pos_verbatim = [verbatim_all[i] for i in positive_idx]
    pos_buckets = [buckets_all[i] for i in positive_idx]
    print(f"[30] ML eligible (actual=1): {len(pos_uids)}")
    if len(pos_uids) < 10:
        print("[30] too few positives for stable test"); return

    # Find each user's first post-cutoff rating
    cutoff = date.fromisoformat(T_TEST_CUTOFF)
    end = cutoff + timedelta(days=30)
    rt = (load_ratings()
          .filter((pl.col("userId").is_in(pos_uids))
                   & (pl.col("t_dat") >= cutoff) & (pl.col("t_dat") < end))
          .sort(["userId", "t_dat"]).group_by("userId").head(1).collect())
    m = load_movies().collect()
    rj = rt.join(m, on="movieId", how="left")
    next_text = {r["userId"]: f"{r['title']} ({r['genres']})" for r in rj.iter_rows(named=True)}

    # Distractor pool from same window
    pool_df = (load_ratings()
               .filter((pl.col("t_dat") >= cutoff) & (pl.col("t_dat") < end))
               .group_by("movieId").len().sort("len", descending=True).head(100)
               .collect())
    pool = pool_df.join(m, on="movieId", how="left")
    pool_texts = [f"{r['title']} ({r['genres']})" for r in pool.iter_rows(named=True)]

    # Embed via BGE-large
    print("[30] loading bge-large...", flush=True)
    from sentence_transformers import SentenceTransformer
    bge = SentenceTransformer("BAAI/bge-large-en-v1.5")
    cust_pairs = []
    for j, (u, v) in enumerate(zip(pos_uids, pos_verbatim)):
        if u in next_text and v and len(v) > 10:
            cust_pairs.append((u, v, next_text[u], pos_buckets[j]))
    verbs = [c[1] for c in cust_pairs]
    actuals = [c[2] for c in cust_pairs]
    buckets = [c[3] for c in cust_pairs]
    print(f"[30] embedding {len(verbs)+len(actuals)+len(pool_texts)}...")
    v_emb = bge.encode(verbs, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    a_emb = bge.encode(actuals, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    p_emb = bge.encode(pool_texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    cos_actuals = np.einsum("ij,ij->i", v_emb, a_emb)
    cos_dist = v_emb @ p_emb.T
    mean_cos_dist = cos_dist.mean(axis=1)
    mrrs = []
    for i in range(len(cust_pairs)):
        scores = np.concatenate([[cos_actuals[i]], cos_dist[i]])
        order = np.argsort(-scores); rank = int(np.where(order == 0)[0][0]) + 1
        mrrs.append(1.0 / rank)
    mrrs = np.array(mrrs)

    # Within-bucket permutation null
    rng = np.random.default_rng(SEED + 400)
    bucket_to_idx = {}
    for i, b in enumerate(buckets):
        bucket_to_idx.setdefault(b, []).append(i)
    n_perm = 5000
    null_means = []
    for _ in range(n_perm):
        cs = []
        for b, idxs in bucket_to_idx.items():
            if len(idxs) < 2:
                for k in idxs:
                    cs.append(float(cos_actuals[k]))
                continue
            perm = rng.permutation(len(idxs))
            for j, k in enumerate(idxs):
                cs.append(float(np.dot(v_emb[k], a_emb[idxs[perm[j]]])))
        null_means.append(float(np.mean(cs)))
    null_means = np.array(null_means)
    observed = float(cos_actuals.mean())
    p_one = float((null_means >= observed).mean())

    paired_diffs = cos_actuals - mean_cos_dist
    diff_mean = float(paired_diffs.mean())
    se = float(np.std(paired_diffs, ddof=1)) / np.sqrt(len(paired_diffs))
    ci_lo, ci_hi = diff_mean - 1.96 * se, diff_mean + 1.96 * se
    chance_mrr = float(sum(1 / r for r in range(1, 102)) / 101)

    out = {
        "domain": "MovieLens", "embedder": "BAAI/bge-large-en-v1.5",
        "n_eligible": len(cust_pairs),
        "H9a_mean_cos_actual": observed,
        "H9a_mean_cos_perm_null": float(null_means.mean()),
        "H9a_diff": observed - float(null_means.mean()),
        "H9a_perm_p_one_sided": p_one,
        "paired_diff_mean_actual_minus_dist": diff_mean,
        "paired_diff_95CI": [ci_lo, ci_hi],
        "H9b_MRR": float(mrrs.mean()),
        "H9b_chance_MRR_E_uniform": chance_mrr,
        "H9b_margin": float(mrrs.mean() - chance_mrr),
        "n_perm": n_perm,
    }
    (RESULTS / "phase30_ml_h9_verbatim.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"[30] H9a diff={out['H9a_diff']:+.4f}, perm p={p_one:.4f}")
    print(f"[30] H9b MRR={out['H9b_MRR']:.4f}, chance={chance_mrr:.4f}, margin={mrrs.mean()-chance_mrr:+.4f}")


if __name__ == "__main__":
    main()
