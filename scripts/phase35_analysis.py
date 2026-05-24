"""Phase 35: v3 sandbox analysis. Computes:

- per-method: funnel_realized_purchase_rate, sandbox_signed_gap,
  scalar_signed_gap, commitment_shrinkage, within_bucket_ρ, DP2 MRR (where
  applicable).
- H10 adjudication: which methods cross |sandbox_gap| ≤ 0.05 with paired
  stratified bootstrap CI excluding ±0.05.
- H11 adjudication: paired bootstrap of within_bucket_ρ(S*) − within_bucket_ρ(M1).
- R4 commitment_shrinkage permutation null.
- per-bucket breakdown.
- writes results/phase35_v3_analysis.json + figures.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import polars as pl
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
SBX_DIR = ROOT / "results" / "phase34_sandbox"
OUT = ROOT / "results" / "phase35_v3_analysis.json"
METHODS = ["M1", "M3", "M8", "M9", "S1", "S2", "S3", "S4"]
BUCKETS = ["1", "2-5", "6-20", "21-100", "101+"]


def load_method(mname: str) -> list[dict]:
    fn = SBX_DIR / f"{mname}.jsonl"
    if not fn.exists():
        return []
    recs = []
    for line in fn.read_text().splitlines():
        try:
            recs.append(json.loads(line))
        except Exception:
            pass
    return [r for r in recs if "error" not in r]


def reweight_to_test(records: list[dict], test_bucket_weights: dict[str, float]) -> tuple[float, np.ndarray]:
    """Reweight per-bucket means to match test-distribution weights.
    Returns (reweighted_mean, per_record_weights).
    """
    if not records:
        return 0.0, np.array([])
    bs = [r["bucket"] for r in records]
    # Stratum sizes in sample
    n_per = {b: sum(1 for x in bs if x == b) for b in set(bs)}
    weights = np.array([test_bucket_weights.get(b, 0) / max(n_per[b], 1) for b in bs])
    weights = weights / weights.sum()
    return float(np.array([r["purchased"] for r in records]) @ weights), weights


def signed_gap(records, key, label_key="actual", weights=None):
    """E[key] − E[label_key], optionally weighted."""
    if not records:
        return None
    a = np.array([r.get(key, 0) for r in records], dtype=float)
    b = np.array([r.get(label_key, 0) for r in records], dtype=float)
    if weights is None:
        return float(a.mean() - b.mean())
    return float(a @ weights - b @ weights)


def stratified_bootstrap_gap(records, label_key="actual", key="purchased",
                             test_weights: dict[str, float] | None = None,
                             B: int = 1000, seed: int = 2026) -> dict:
    """Paired stratified bootstrap CI on signed gap (mean(key) - mean(label))."""
    rng = np.random.default_rng(seed)
    by_bucket: dict[str, list[dict]] = {}
    for r in records:
        by_bucket.setdefault(r["bucket"], []).append(r)
    buckets = list(by_bucket.keys())
    point = signed_gap(records, key, label_key)
    samples = []
    for _ in range(B):
        boot_records = []
        for b in buckets:
            grp = by_bucket[b]
            idx = rng.integers(0, len(grp), size=len(grp))
            boot_records.extend([grp[i] for i in idx])
        # Compute reweighted gap
        if test_weights:
            _, w = reweight_to_test(boot_records, test_weights)
            samples.append(signed_gap(boot_records, key, label_key, weights=w))
        else:
            samples.append(signed_gap(boot_records, key, label_key))
    samples = np.array(samples)
    return {
        "point": point, "se": float(samples.std()),
        "lo": float(np.quantile(samples, 0.025)),
        "hi": float(np.quantile(samples, 0.975)),
        "B": B,
    }


def within_bucket_spearman(records, key="purchased", label_key="actual") -> dict:
    """Bucket-weighted Spearman ρ — within each bucket, then weight by bucket size."""
    by_bucket: dict[str, list[dict]] = {}
    for r in records:
        by_bucket.setdefault(r["bucket"], []).append(r)
    rhos = {}
    n_b = {}
    for b, grp in by_bucket.items():
        if len(grp) < 5:
            continue
        a = np.array([r.get(key, 0) for r in grp], dtype=float)
        l = np.array([r.get(label_key, 0) for r in grp], dtype=float)
        if a.std() == 0 or l.std() == 0:
            rhos[b] = 0.0
        else:
            rho, _ = stats.spearmanr(a, l)
            rhos[b] = float(rho) if np.isfinite(rho) else 0.0
        n_b[b] = len(grp)
    total_n = sum(n_b.values())
    weighted = sum(rhos[b] * n_b[b] for b in rhos) / max(total_n, 1)
    return {"per_bucket_rho": rhos, "weighted_rho": float(weighted), "n_per_bucket": n_b}


def bootstrap_rho_paired(rec_a, rec_b, key="purchased", label_key="actual",
                         B: int = 1000, seed: int = 2026) -> dict:
    """Paired bootstrap of within_bucket_rho difference (A - B)."""
    # Index records by customer_id for pairing
    ca = {r["customer_id"]: r for r in rec_a}
    cb = {r["customer_id"]: r for r in rec_b}
    common = sorted(set(ca.keys()) & set(cb.keys()))
    paired_a = [ca[c] for c in common]
    paired_b = [cb[c] for c in common]
    point = within_bucket_spearman(paired_a, key, label_key)["weighted_rho"] \
        - within_bucket_spearman(paired_b, key, label_key)["weighted_rho"]
    rng = np.random.default_rng(seed)
    # Stratified by bucket
    by_bucket = {}
    for i, r in enumerate(paired_a):
        by_bucket.setdefault(r["bucket"], []).append(i)
    samples = []
    for _ in range(B):
        idxs = []
        for b, items in by_bucket.items():
            idxs.extend(rng.choice(items, size=len(items), replace=True).tolist())
        ba = [paired_a[i] for i in idxs]
        bb = [paired_b[i] for i in idxs]
        diff = within_bucket_spearman(ba, key, label_key)["weighted_rho"] \
            - within_bucket_spearman(bb, key, label_key)["weighted_rho"]
        samples.append(diff)
    samples = np.array(samples)
    return {"point": point, "lo": float(np.quantile(samples, 0.025)),
            "hi": float(np.quantile(samples, 0.975)),
            "se": float(samples.std()), "B": B}


def commitment_shrinkage_with_null(records, B: int = 1000, seed: int = 2026):
    """commitment_shrinkage(M) = scalar_gap - sandbox_gap (within method).
    Null: shuffle per-customer (scalar, purchased) pairs randomly within bucket.
    """
    with_scalar = [r for r in records if "scalar_prob" in r]
    if len(with_scalar) < 50:
        return None
    a = np.array([r["scalar_prob"] for r in with_scalar])
    p = np.array([r["purchased"] for r in with_scalar], dtype=float)
    actual = np.array([r["actual"] for r in with_scalar], dtype=float)
    scalar_gap = float(a.mean() - actual.mean())
    sandbox_gap = float(p.mean() - actual.mean())
    point = scalar_gap - sandbox_gap

    # Null: within bucket, shuffle the (scalar, purchased) labels relative to customers
    rng = np.random.default_rng(seed)
    by_bucket = {}
    for i, r in enumerate(with_scalar):
        by_bucket.setdefault(r["bucket"], []).append(i)
    nulls = []
    for _ in range(B):
        sh = np.arange(len(with_scalar))
        for b, idx in by_bucket.items():
            rng.shuffle(idx)
            sh[idx] = idx  # rotation
        # Permute purchased only within buckets
        p_perm = np.empty_like(p)
        for b, idx in by_bucket.items():
            perm = rng.permutation(idx)
            p_perm[idx] = p[perm]
        null = float(a.mean() - actual.mean()) - float(p_perm.mean() - actual.mean())
        nulls.append(null)
    nulls = np.array(nulls)
    p_val = float((np.abs(nulls) >= abs(point)).mean())
    return {"scalar_gap": scalar_gap, "sandbox_gap": sandbox_gap,
            "commitment_shrinkage": point,
            "null_se": float(nulls.std()),
            "null_lo": float(np.quantile(nulls, 0.025)),
            "null_hi": float(np.quantile(nulls, 0.975)),
            "permutation_p": p_val, "B": B}


def chosen_item_summary(records):
    """Distribution of DP2 candidate choices (A/B/C) given CONSIDER."""
    chosen = {"in-section": 0, "cross-section": 0, "OOD": 0, "none": 0}
    n_consider = 0
    for r in records:
        for wa in r.get("weekly_actions", []):
            dp2 = wa.get("dp2")
            if dp2 and dp2.get("action") == "CONSIDER":
                n_consider += 1
                cand = wa.get("chosen_candidate")
                if cand:
                    chosen[cand.get("label", "none")] = chosen.get(cand.get("label", "none"), 0) + 1
                else:
                    chosen["none"] += 1
    return {"n_consider": n_consider, "by_label": chosen}


def main():
    # Test-distribution weights for reweighting
    test = pl.read_parquet(ROOT / "data" / "splits" / "test.parquet")
    test_counts = test.group_by("activity_bucket").len().to_dicts()
    total = sum(d["len"] for d in test_counts)
    test_weights = {d["activity_bucket"]: d["len"] / total for d in test_counts}

    result = {
        "methods": {},
        "pairwise_h11": {},
        "test_weights": test_weights,
    }
    base_records = load_method("M1")
    for mname in METHODS:
        recs = load_method(mname)
        if not recs:
            print(f"WARN: no records for {mname}")
            continue
        # Reweighted gap
        _, w = reweight_to_test(recs, test_weights)
        gap = signed_gap(recs, "purchased", "actual")
        gap_re = signed_gap(recs, "purchased", "actual", weights=w)
        boot = stratified_bootstrap_gap(recs, B=1000, test_weights=test_weights)
        wbs = within_bucket_spearman(recs)
        cs = commitment_shrinkage_with_null(recs)
        cis = chosen_item_summary(recs)
        per_bucket = {}
        for b in BUCKETS:
            sub = [r for r in recs if r["bucket"] == b]
            if sub:
                per_bucket[b] = {
                    "n": len(sub),
                    "funnel_purchase_rate": float(np.mean([r["purchased"] for r in sub])),
                    "actual_rate": float(np.mean([r["actual"] for r in sub])),
                    "gap": float(np.mean([r["purchased"] for r in sub]) - np.mean([r["actual"] for r in sub])),
                }
        result["methods"][mname] = {
            "n": len(recs),
            "funnel_realized_purchase_rate": float(np.mean([r["purchased"] for r in recs])),
            "actual_rate": float(np.mean([r["actual"] for r in recs])),
            "sandbox_signed_gap": gap,
            "sandbox_signed_gap_reweighted": gap_re,
            "sandbox_gap_bootstrap": boot,
            "within_bucket_rho": wbs,
            "commitment_shrinkage": cs,
            "chosen_item_summary": cis,
            "per_bucket": per_bucket,
        }
        # H10 adjudication
        h10_pass = (abs(boot["point"]) <= 0.05) and (boot["lo"] >= -0.05) and (boot["hi"] <= 0.05)
        result["methods"][mname]["H10_pass"] = bool(h10_pass)

    # H11: each S vs M1
    for s in ["S1", "S2", "S3", "S4"]:
        r_s = load_method(s)
        if not r_s or not base_records:
            continue
        diff = bootstrap_rho_paired(r_s, base_records)
        result["pairwise_h11"][f"{s}_vs_M1"] = {
            **diff,
            "H11_pass": bool(diff["point"] >= 0.03 and diff["lo"] > 0)
        }

    # H10 aggregate
    result["H10_any_pass"] = any(v.get("H10_pass") for v in result["methods"].values())
    result["H11_any_pass"] = any(v.get("H11_pass") for v in result["pairwise_h11"].values())

    OUT.write_text(json.dumps(result, indent=2))
    print(f"Wrote {OUT}")
    print("\nSUMMARY:")
    print(f"  Test label rate (reweighted target): {sum(v * test_weights[b] for b, v in test_weights.items()):.3f}")
    for m, v in result["methods"].items():
        if "sandbox_signed_gap_reweighted" in v:
            ci = v["sandbox_gap_bootstrap"]
            print(f"  {m}: gap={v['sandbox_signed_gap_reweighted']:+.4f} "
                  f"[{ci['lo']:+.4f}, {ci['hi']:+.4f}]  ρ_wb={v['within_bucket_rho']['weighted_rho']:+.3f}  "
                  f"H10={'YES' if v['H10_pass'] else 'no'}")
    print(f"\nH10 (any method closes |gap| ≤ 0.05): {'PASS' if result['H10_any_pass'] else 'FAIL'}")
    print(f"H11 (any sandbox-native beats M1 on ρ_wb): {'PASS' if result['H11_any_pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
