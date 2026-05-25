"""Phase 41: full analysis on the Claude cross-provider 8-method data.

Reuses phase35 analysis logic but reads from results/phase40_claude_*.jsonl files.
Outputs results/phase41_claude_analysis.json with H10/H11 verdicts, per-bucket
gaps, within-bucket Spearman, commitment shrinkage with permutation null, and
Holm-Bonferroni FWER correction over the 8 methods.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import polars as pl
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase41_claude_analysis.json"
METHODS = ["M1", "M3", "M8", "M9", "S1", "S2", "S3", "S4"]
BUCKETS = ["1", "2-5", "6-20", "21-100", "101+"]
H10_ENVELOPE = 0.05
H11_DELTA_THRESH = 0.03


def load_claude_method(mname: str) -> list[dict]:
    """Load Claude method records, de-duplicating by customer_id."""
    fn = ROOT / "results" / ("phase40_claude_predictions.jsonl" if mname == "M1"
                              else f"phase40_claude_{mname}.jsonl")
    if not fn.exists():
        return []
    seen = {}
    for line in fn.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if "error" in r:
            continue
        cid = r.get("customer_id")
        if cid and cid not in seen:
            seen[cid] = r
    return list(seen.values())


def signed_gap(records, key="purchased", label_key="actual", weights=None):
    a = np.array([r.get(key, 0) for r in records], dtype=float)
    b = np.array([r.get(label_key, 0) for r in records], dtype=float)
    if weights is None:
        return float(a.mean() - b.mean())
    return float(a @ weights - b @ weights)


def reweight_to_test(records, test_bucket_weights):
    bs = [r["bucket"] for r in records]
    n_per = {b: sum(1 for x in bs if x == b) for b in set(bs)}
    weights = np.array([test_bucket_weights.get(b, 0) / max(n_per[b], 1) for b in bs])
    weights = weights / max(weights.sum(), 1e-9)
    return weights


def stratified_bootstrap_gap(records, B=1000, seed=2026, test_weights=None):
    rng = np.random.default_rng(seed)
    by_bucket: dict[str, list[dict]] = {}
    for r in records:
        by_bucket.setdefault(r["bucket"], []).append(r)
    buckets = list(by_bucket.keys())
    point = signed_gap(records)
    samples = []
    for _ in range(B):
        boot = []
        for b in buckets:
            grp = by_bucket[b]
            idx = rng.integers(0, len(grp), size=len(grp))
            boot.extend([grp[i] for i in idx])
        if test_weights:
            w = reweight_to_test(boot, test_weights)
            samples.append(signed_gap(boot, weights=w))
        else:
            samples.append(signed_gap(boot))
    samples = np.array(samples)
    return {
        "point": point, "se": float(samples.std()),
        "lo": float(np.quantile(samples, 0.025)),
        "hi": float(np.quantile(samples, 0.975)),
        "B": B,
    }


def within_bucket_spearman(records, key="purchased", label_key="actual"):
    by_b = {}
    for r in records:
        by_b.setdefault(r["bucket"], []).append(r)
    rhos = {}
    n_b = {}
    for b, grp in by_b.items():
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
    total = sum(n_b.values())
    weighted = sum(rhos[b] * n_b[b] for b in rhos) / max(total, 1)
    return {"per_bucket_rho": rhos, "weighted_rho": float(weighted), "n_per_bucket": n_b}


def bootstrap_rho_paired(rec_a, rec_b, B=1000, seed=2026):
    ca = {r["customer_id"]: r for r in rec_a}
    cb = {r["customer_id"]: r for r in rec_b}
    common = sorted(set(ca.keys()) & set(cb.keys()))
    paired_a = [ca[c] for c in common]
    paired_b = [cb[c] for c in common]
    pa = within_bucket_spearman(paired_a)["weighted_rho"]
    pb = within_bucket_spearman(paired_b)["weighted_rho"]
    point = pa - pb
    rng = np.random.default_rng(seed)
    by_b = {}
    for i, r in enumerate(paired_a):
        by_b.setdefault(r["bucket"], []).append(i)
    samples = []
    for _ in range(B):
        idxs = []
        for b, items in by_b.items():
            idxs.extend(rng.choice(items, size=len(items), replace=True).tolist())
        ba = [paired_a[i] for i in idxs]
        bb = [paired_b[i] for i in idxs]
        d = within_bucket_spearman(ba)["weighted_rho"] - within_bucket_spearman(bb)["weighted_rho"]
        samples.append(d)
    samples = np.array(samples)
    return {"point": point, "lo": float(np.quantile(samples, 0.025)),
            "hi": float(np.quantile(samples, 0.975)), "se": float(samples.std()), "B": B}


def commitment_shrinkage_with_null(records, B=1000, seed=2026):
    with_s = [r for r in records if "scalar_prob" in r]
    if len(with_s) < 50:
        return None
    sc = np.array([r["scalar_prob"] for r in with_s])
    p = np.array([r["purchased"] for r in with_s], dtype=float)
    actual = np.array([r["actual"] for r in with_s], dtype=float)
    scalar_gap = float(sc.mean() - actual.mean())
    sandbox_gap = float(p.mean() - actual.mean())
    point = scalar_gap - sandbox_gap
    rng = np.random.default_rng(seed)
    by_b = {}
    for i, r in enumerate(with_s):
        by_b.setdefault(r["bucket"], []).append(i)
    nulls = []
    for _ in range(B):
        p_perm = np.empty_like(p)
        for b, idx in by_b.items():
            perm = rng.permutation(idx)
            p_perm[idx] = p[perm]
        null = float(sc.mean() - actual.mean()) - float(p_perm.mean() - actual.mean())
        nulls.append(null)
    nulls = np.array(nulls)
    return {"scalar_gap": scalar_gap, "sandbox_gap": sandbox_gap,
            "commitment_shrinkage": point,
            "null_se": float(nulls.std()),
            "null_lo": float(np.quantile(nulls, 0.025)),
            "null_hi": float(np.quantile(nulls, 0.975)),
            "permutation_p": float((np.abs(nulls) >= abs(point)).mean()),
            "B": B}


def main():
    test = pl.read_parquet(ROOT / "data" / "splits" / "test.parquet")
    test_counts = test.group_by("activity_bucket").len().to_dicts()
    tot = sum(d["len"] for d in test_counts)
    test_weights = {d["activity_bucket"]: d["len"] / tot for d in test_counts}

    result = {"methods": {}, "pairwise_h11": {}, "test_weights": test_weights, "provider": "claude"}
    base = load_claude_method("M1")

    for m in METHODS:
        recs = load_claude_method(m)
        if not recs:
            print(f"WARN: no Claude records for {m}")
            continue
        w = reweight_to_test(recs, test_weights)
        gap = signed_gap(recs)
        gap_rw = signed_gap(recs, weights=w)
        boot = stratified_bootstrap_gap(recs, B=1000, test_weights=test_weights)
        wbs = within_bucket_spearman(recs)
        cs = commitment_shrinkage_with_null(recs)
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
        result["methods"][m] = {
            "n": len(recs),
            "funnel_realized_purchase_rate": float(np.mean([r["purchased"] for r in recs])),
            "actual_rate": float(np.mean([r["actual"] for r in recs])),
            "sandbox_signed_gap": gap,
            "sandbox_signed_gap_reweighted": gap_rw,
            "sandbox_gap_bootstrap": boot,
            "within_bucket_rho": wbs,
            "commitment_shrinkage": cs,
            "per_bucket": per_bucket,
            "H10_pass": bool(abs(boot["point"]) <= 0.05 and boot["lo"] >= -0.05 and boot["hi"] <= 0.05),
        }

    # H11: each S vs M1
    for s in ["S1", "S2", "S3", "S4"]:
        rs = load_claude_method(s)
        if rs and base:
            d = bootstrap_rho_paired(rs, base)
            result["pairwise_h11"][f"{s}_vs_M1"] = {
                **d,
                "H11_pass": bool(d["point"] >= 0.03 and d["lo"] > 0)
            }

    # Holm-Bonferroni
    K = len(METHODS)
    alpha = 0.025
    p_vals = []
    for m in METHODS:
        v = result["methods"].get(m, {})
        boot = v.get("sandbox_gap_bootstrap")
        if not boot:
            continue
        point = boot["point"]; se = boot["se"] or 1e-9
        from scipy.stats import norm
        z_pos = (H10_ENVELOPE - point) / se
        z_neg = (-H10_ENVELOPE - point) / se
        p_within = max(0.0, min(1.0, norm.cdf(z_pos) - norm.cdf(z_neg)))
        p_vals.append((m, 1 - p_within))
    p_vals.sort(key=lambda x: x[1])
    holm = {}
    for i, (m, p) in enumerate(p_vals):
        crit = alpha / (K - i)
        holm[m] = {"p": p, "holm_crit": crit, "reject_h0": bool(p <= crit)}
    result["holm_bonferroni_h10"] = holm

    # MDE for H11
    ses = [v.get("se", 0) for v in result["pairwise_h11"].values() if v.get("se")]
    if ses:
        median_se = sorted(ses)[len(ses) // 2]
        from scipy.stats import norm
        z_a = norm.ppf(1 - alpha / 2)
        z_b = norm.ppf(0.80)
        result["h11_mde"] = {
            "median_paired_se": float(median_se),
            "mde_80_power_alpha_0.025": float((z_a + z_b) * median_se),
            "preregistered_threshold": H11_DELTA_THRESH,
        }

    result["H10_any_pass"] = any(v.get("H10_pass") for v in result["methods"].values())
    result["H11_any_pass"] = any(v.get("H11_pass") for v in result["pairwise_h11"].values())
    result["H10_holm_any_pass"] = any(h["reject_h0"] for h in holm.values())

    OUT.write_text(json.dumps(result, indent=2))
    print(f"Wrote {OUT}")

    print("\n=== CLAUDE n=1000 RESULTS ===")
    actual_rate = result["methods"]["M1"]["actual_rate"]
    print(f"Test rate: {actual_rate:.3f}")
    print(f"\n{'method':<6} {'n':>5} {'gap (rew)':>10} {'CI':>20} {'wb_rho':>8} {'H10':>6} {'Holm_p':>9}")
    for m in METHODS:
        v = result["methods"].get(m, {})
        if not v:
            continue
        b = v["sandbox_gap_bootstrap"]
        rho = v["within_bucket_rho"]["weighted_rho"]
        h10 = "PASS" if v["H10_pass"] else "fail"
        holm_p = holm.get(m, {}).get("p", float("nan"))
        print(f"{m:<6} {v['n']:>5} {v['sandbox_signed_gap_reweighted']:>+10.3f} "
              f"[{b['lo']:+.3f}, {b['hi']:+.3f}] {rho:>+8.3f}   {h10:>5}   {holm_p:>.4f}")
    print(f"\nH10 any pass (Bonferroni): {result['H10_any_pass']}")
    print(f"H10 any pass (Holm-Bonf):  {result['H10_holm_any_pass']}")
    print(f"H11 any pass:              {result['H11_any_pass']}")
    if 'h11_mde' in result:
        print(f"H11 MDE @ 80% power: {result['h11_mde']['mde_80_power_alpha_0.025']:.3f} "
              f"(prereg threshold: {result['h11_mde']['preregistered_threshold']})")
    if any(v.get("commitment_shrinkage") for v in result["methods"].values()):
        print(f"\nCommitment shrinkage per method:")
        for m in METHODS:
            cs = result["methods"].get(m, {}).get("commitment_shrinkage")
            if cs:
                print(f"  {m}: scalar={cs['scalar_gap']:+.3f}  sandbox={cs['sandbox_gap']:+.3f}  "
                      f"shrink={cs['commitment_shrinkage']:+.3f}  perm_p={cs['permutation_p']:.3f}")


if __name__ == "__main__":
    main()
