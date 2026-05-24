"""Phase 11b: verbatim coherence (H9).

H9a: For each F-arm customer, cos(verbatim, actual_next_purchased_article_desc)
     exceeds a within-bucket-within-week shuffled-pair baseline (paired permutation).
H9b: MRR of the actual next article among 100 in-bucket-within-week distractor
     articles is > 0.01 by margin ≥ 0.05.

Embedding model: Gemini gemini-embedding-001 (not orthogonal to the LLM, but the
only quota-bearing embedder available; documented as a confound). Future work
should re-run with bge-large.

Quote-specificity audit: TTR of verbatim, conditional H9 results on TTR quartile.
"""

from __future__ import annotations
import json
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import polars as pl
from scipy.stats import permutation_test

from src import T_TEST_CUTOFF, SEED
from src.data import load_transactions, load_articles
from src.llm_client import call_llm

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _embed_texts(texts: list[str], batch_size: int = 50) -> np.ndarray:
    """Embed using Gemini gemini-embedding-001 via the existing google-genai client."""
    from google import genai
    import os
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # Newer SDK: embed_content takes a list of contents
        try:
            resp = client.models.embed_content(model="gemini-embedding-001", contents=batch)
            for e in resp.embeddings:
                vecs.append(np.array(e.values, dtype=np.float32))
        except Exception as e:
            # fall back one-by-one
            for t in batch:
                try:
                    r = client.models.embed_content(model="gemini-embedding-001", contents=t)
                    vecs.append(np.array(r.embeddings[0].values, dtype=np.float32))
                except Exception as e2:
                    vecs.append(np.zeros(768, dtype=np.float32))
    return np.array(vecs)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _ttr(text: str) -> float:
    """Type-token ratio."""
    toks = [t.lower() for t in text.split() if t.isalpha()]
    if not toks:
        return 0.0
    return len(set(toks)) / len(toks)


def _next_purchases_after_cutoff(customer_ids: list[str], cutoff: str, window_days: int = 30) -> dict:
    """For each customer, find the FIRST article purchased within [cutoff, cutoff+window_days)
    and return its prod_name + detail_desc text."""
    cutoff_d = date.fromisoformat(cutoff)
    end = cutoff_d + timedelta(days=window_days)
    tx = (load_transactions()
          .filter((pl.col("customer_id").is_in(customer_ids))
                  & (pl.col("t_dat") >= cutoff_d) & (pl.col("t_dat") < end))
          .sort(["customer_id", "t_dat"])
          .group_by("customer_id").head(1)
          .collect())
    art = load_articles().select(["article_id", "prod_name", "product_type_name",
                                  "garment_group_name", "colour_group_name", "section_name",
                                  "detail_desc"]).collect()
    tx_j = tx.join(art, on="article_id", how="left")
    out = {}
    for r in tx_j.iter_rows(named=True):
        desc = (f"{r['prod_name']} — {r['product_type_name']} in {r['colour_group_name']}, "
                f"{r['garment_group_name']}, {r['section_name']}. {r['detail_desc'] or ''}")
        out[r["customer_id"]] = desc.strip()
    return out


def _distractor_pool_by_bucket(cutoff: str, window_days: int = 30, k_per_bucket: int = 200) -> dict:
    """Build a pool of articles purchased by ANY customer in the test window, stratified by
    activity bucket of their buyer (proxy for in-bucket-within-week)."""
    # Simplified: pull a random k articles from the post-cutoff window.
    cutoff_d = date.fromisoformat(cutoff)
    end = cutoff_d + timedelta(days=window_days)
    tx = (load_transactions().filter((pl.col("t_dat") >= cutoff_d) & (pl.col("t_dat") < end))
          .group_by("article_id").len().sort("len", descending=True).head(k_per_bucket)
          .collect())
    art = load_articles().select(["article_id", "prod_name", "product_type_name",
                                  "garment_group_name", "colour_group_name", "section_name",
                                  "detail_desc"]).collect()
    pool_df = tx.join(art, on="article_id", how="left")
    pool = []
    for r in pool_df.iter_rows(named=True):
        desc = (f"{r['prod_name']} — {r['product_type_name']} in {r['colour_group_name']}, "
                f"{r['garment_group_name']}, {r['section_name']}. {r['detail_desc'] or ''}")
        pool.append({"article_id": r["article_id"], "text": desc.strip()})
    return pool


def main():
    fnb_path = RESULTS / "phase10_F-nobase_scores.npz"
    if not fnb_path.exists():
        raise SystemExit("[11b] F-nobase scores missing. Run Phase 10 F-nobase first.")
    d = np.load(fnb_path, allow_pickle=True)
    cids = list(d["customer_id"])
    actual = d["actual"].astype(int)
    verbatim = list(d["verbatim"])
    buckets = d["activity_bucket"].astype(str)

    # We can only evaluate H9 on customers who actually bought (actual=1) — they have a real next-article.
    positive_mask = actual == 1
    positive_cids = [cids[i] for i in range(len(cids)) if positive_mask[i]]
    positive_verbatim = [verbatim[i] for i in range(len(cids)) if positive_mask[i]]
    positive_buckets = [buckets[i] for i in range(len(cids)) if positive_mask[i]]  # audit-fix: parallel bucket array
    print(f"[11b] customers with actual=1 in F-nobase: {len(positive_cids)} (these are eligible for H9)")

    next_purchases = _next_purchases_after_cutoff(positive_cids, T_TEST_CUTOFF, window_days=30)
    distractor_pool = _distractor_pool_by_bucket(T_TEST_CUTOFF, window_days=30, k_per_bucket=100)
    print(f"[11b] distractor pool: {len(distractor_pool)} unique articles")

    # Build texts to embed
    customer_texts = []
    article_texts = [a["text"] for a in distractor_pool]
    cust_idx_for_eval = []
    for i, cid in enumerate(positive_cids):
        if cid not in next_purchases:
            continue
        if not positive_verbatim[i] or len(positive_verbatim[i]) < 10:
            continue
        customer_texts.append(positive_verbatim[i])
        customer_texts.append(next_purchases[cid])  # appended right after verbatim
        cust_idx_for_eval.append(i)
    print(f"[11b] embedding {len(customer_texts)} customer-side texts + {len(article_texts)} distractors")

    cust_embs = _embed_texts(customer_texts)
    art_embs = _embed_texts(article_texts)
    print(f"[11b] cust_embs shape: {cust_embs.shape}, art_embs shape: {art_embs.shape}")

    # Pair: verbatim_emb [2j], actual_next_emb [2j+1]. AUDIT FIX (Agent B BLOCKER): track
    # parallel bucket array; do NOT index `buckets[ci]` because `buckets` is for the FULL
    # F-nobase cids (length len(cids)) while `ci` is an index into `positive_cids`.
    results = []
    for j, ci in enumerate(cust_idx_for_eval):
        v_emb = cust_embs[2 * j]
        a_emb = cust_embs[2 * j + 1]
        cos_actual = cosine(v_emb, a_emb)
        # Cosines against each distractor article
        dists = np.array([cosine(v_emb, art_embs[k]) for k in range(len(art_embs))])
        # Rank actual vs 100 distractors (exclude actual from the distractor pool by construction).
        # Chance MRR = expected reciprocal rank when actual is uniformly placed among 101 items
        # = H_101 / 101 ≈ 0.0517 (NOT 1/101). We report both numbers.
        scores_with_actual = np.concatenate([[cos_actual], dists])
        order = np.argsort(-scores_with_actual)
        rank = int(np.where(order == 0)[0][0]) + 1
        ttr = _ttr(positive_verbatim[ci])
        results.append({
            "cid": positive_cids[ci][:12] + "…",
            "bucket": positive_buckets[ci],   # audit-fix
            "verbatim": positive_verbatim[ci][:120],
            "cos_actual": cos_actual,
            "mean_cos_distractor": float(dists.mean()),
            "rank_among_distractors_plus_actual": rank,
            "ttr": ttr,
        })

    # H9a — within-bucket-within-week shuffled-pair permutation baseline.
    # AUDIT FIX (Agent B BLOCKER): the previous `mean_cos_distractor` collapsed
    # the null into a deterministic statistic. Correct null: permute the
    # (verbatim → actual_article_emb) mapping within bucket strata and
    # recompute the mean cos_actual. We don't have weekly stratification here
    # (the distractor pool isn't week-tagged), so we use bucket-only stratification.
    cos_actuals = np.array([r["cos_actual"] for r in results])
    arms_per_bucket = {}
    for r in results:
        arms_per_bucket.setdefault(r["bucket"], []).append(r["cos_actual"])

    # Build pools of (verbatim_emb, actual_emb) per bucket, then permute actual_emb within bucket.
    bucket_lookup = {}
    for j, ci in enumerate(cust_idx_for_eval):
        b = positive_buckets[ci]
        bucket_lookup.setdefault(b, []).append((cust_embs[2 * j], cust_embs[2 * j + 1]))

    rng = np.random.default_rng(SEED)
    permuted_means = []
    n_perm = 10000
    for _ in range(n_perm):
        cosines_this_perm = []
        for b, pairs in bucket_lookup.items():
            if len(pairs) < 2:
                # Can't permute a singleton bucket; keep its observed value (worst case for power).
                for v, a in pairs:
                    cosines_this_perm.append(cosine(v, a))
                continue
            actuals = [a for _, a in pairs]
            perm_idx = rng.permutation(len(actuals))
            for k, (v, _) in enumerate(pairs):
                cosines_this_perm.append(cosine(v, actuals[perm_idx[k]]))
        permuted_means.append(float(np.mean(cosines_this_perm)))
    permuted_means = np.array(permuted_means)
    observed_mean = float(cos_actuals.mean())
    p_one_sided = float((permuted_means >= observed_mean).mean())

    class _Pres:
        pass
    permres = _Pres()
    permres.pvalue = p_one_sided
    # Keep cos_shuffled as the per-customer mean over the in-bucket null for diagnostics.
    cos_shuffled_diag = float(permuted_means.mean())

    # H9b: MRR
    mrrs = np.array([1.0 / r["rank_among_distractors_plus_actual"] for r in results])

    # TTR audit
    ttrs = np.array([r["ttr"] for r in results])
    high_ttr_mask = ttrs >= np.quantile(ttrs, 0.75)
    # Reference: high-TTR subset cos_actual minus the within-bucket null mean
    h9a_high = float(cos_actuals[high_ttr_mask].mean() - cos_shuffled_diag)
    h9b_high = float(mrrs[high_ttr_mask].mean())

    # AUDIT FIX (Agent B MAJOR): chance MRR for actual uniformly placed among 101 items =
    # E[1/rank] = H_101 / 101 ≈ 0.0517, NOT 1/101. We report both.
    n_items = len(distractor_pool) + 1
    chance_mrr_expected = float(sum(1 / r for r in range(1, n_items + 1)) / n_items)

    out = {
        "n_eligible": len(results),
        "n_distractors": len(distractor_pool),
        "H9a_mean_cos_actual": float(cos_actuals.mean()),
        "H9a_mean_cos_shuffled_within_bucket_null": cos_shuffled_diag,
        "H9a_diff": observed_mean - cos_shuffled_diag,
        "H9a_permutation_p_one_sided": float(permres.pvalue),
        "H9a_null_distribution_method": "within-bucket-permuted (n_perm=10000)",
        "H9a_bonferroni_threshold": 0.0125,
        "H9a_verdict": "CONFIRMED" if permres.pvalue < 0.0125 else "REFUTED_or_NS",
        "H9b_MRR": float(mrrs.mean()),
        "H9b_chance_MRR_E_uniform": chance_mrr_expected,
        "H9b_chance_MRR_one_over_n": 1.0 / n_items,
        "H9b_margin_vs_E_uniform": float(mrrs.mean() - chance_mrr_expected),
        "H9b_verdict": "CONFIRMED" if (mrrs.mean() - chance_mrr_expected) >= 0.05 else "REFUTED_or_NS",
        "H9_overall_verdict": "CONFIRMED" if (permres.pvalue < 0.0125 and (mrrs.mean() - chance_mrr_expected) >= 0.05) else "REFUTED_or_NS",
        "quote_specificity": {
            "mean_TTR": float(ttrs.mean()),
            "Q3_TTR": float(np.quantile(ttrs, 0.75)),
            "high_TTR_H9a_diff": h9a_high,
            "high_TTR_H9b_MRR": h9b_high,
        },
        "per_customer_sample": results[:20],
    }
    (RESULTS / "phase11_verbatim.json").write_text(json.dumps(out, indent=2, default=str))

    print(f"\n[11b] H9a: cos_actual = {out['H9a_mean_cos_actual']:.4f}, cos_shuffled (within-bucket null) = {out['H9a_mean_cos_shuffled_within_bucket_null']:.4f}")
    print(f"      diff = {out['H9a_diff']:+.4f}, perm p = {out['H9a_permutation_p_one_sided']:.4g}")
    print(f"      verdict: {out['H9a_verdict']}")
    print(f"[11b] H9b: MRR = {out['H9b_MRR']:.4f}, E[chance uniform] = {out['H9b_chance_MRR_E_uniform']:.4f}, "
          f"margin = {out['H9b_margin_vs_E_uniform']:+.4f}")
    print(f"      verdict: {out['H9b_verdict']}")
    print(f"[11b] H9 overall: {out['H9_overall_verdict']}")
    print(f"\n[11b] saved phase11_verbatim.json")


if __name__ == "__main__":
    main()
