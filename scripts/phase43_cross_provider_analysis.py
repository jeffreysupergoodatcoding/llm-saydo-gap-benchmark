"""Phase 43: cross-provider analysis across three arms.

Compares — on the same customer ids where possible — three operationalizations
of LLM-as-twin:

  A. Gemini per-DP                 (results/phase34_sandbox/{M}.jsonl)
  B. Claude meta-policy            (results/phase40_claude_{M}.jsonl)
  C. Claude proper per-DP          (results/phase42_claude_proper_M1.jsonl
                                     + phase42_claude_proper_M1_gapfill_*.jsonl)

Outputs results/phase43_cross_provider_analysis.json with:
- Per-arm per-method gap, bootstrap CI, within-bucket Spearman
- Per-bucket gap matrix
- Pairwise paired-customer comparison (where common cids exist)
- Provider-invariance test (signed gap of arm-A vs arm-C on common cids)
- Meta-policy-vs-per-DP test (arm-B vs arm-C on common cids, same provider)
- Methodological notes on data integrity and race-condition recovery
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import polars as pl
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
SBX_DIR = ROOT / "results" / "phase34_sandbox"
META_DIR = ROOT / "results"
PROPER_FILE = ROOT / "results" / "phase42_claude_proper_M1.jsonl"
PROPER_GAPFILL_PREFIX = "phase42_claude_proper_M1_gapfill_"
LABELS_FILE = ROOT / "results" / "phase42_actual_labels.json"
OUT = ROOT / "results" / "phase43_cross_provider_analysis.json"

GEMINI_METHODS = ["M1", "M2", "M3", "M7", "M8", "M8a", "M9", "S1", "S2", "S3", "S4"]
META_METHODS = ["M1", "M3", "M8", "M9", "S1", "S2", "S3", "S4"]
BUCKETS = ["1", "2-5", "6-20", "21-100", "101+"]


def _dedupe(records: list[dict]) -> list[dict]:
    seen = {}
    for r in records:
        if "error" in r:
            continue
        cid = r.get("customer_id")
        if cid and cid not in seen:
            seen[cid] = r
    return list(seen.values())


def load_gemini(method: str) -> list[dict]:
    fn = SBX_DIR / f"{method}.jsonl"
    if not fn.exists():
        return []
    recs = []
    for line in fn.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        has_dp_err = False
        for wa in r.get("weekly_actions", []):
            for k in ("dp1", "dp2", "dp3"):
                v = wa.get(k)
                if isinstance(v, dict) and "error" in v:
                    has_dp_err = True
                    break
            if has_dp_err:
                break
        if has_dp_err:
            continue
        recs.append(r)
    return _dedupe(recs)


def load_claude_meta(method: str) -> list[dict]:
    fn = META_DIR / ("phase40_claude_predictions.jsonl" if method == "M1"
                     else f"phase40_claude_{method}.jsonl")
    if not fn.exists():
        return []
    recs = []
    for line in fn.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        recs.append(r)
    return _dedupe(recs)


def load_claude_proper() -> list[dict]:
    """Load proper per-DP records — main file + any gap-fill per-batch files.
    Drop records whose customer_id is not in the original input batches."""
    valid_cids = set()
    for i in range(8):
        try:
            b = json.loads((META_DIR / f"phase42_claude_proper_batch_{i}.json").read_text())
            for c in b:
                valid_cids.add(c["customer_id"])
        except Exception:
            pass
    if not LABELS_FILE.exists():
        return []
    labels = json.loads(LABELS_FILE.read_text())

    raw = []
    if PROPER_FILE.exists():
        for line in PROPER_FILE.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            raw.append(r)
    for fn in META_DIR.glob(f"{PROPER_GAPFILL_PREFIX}*.jsonl"):
        for line in fn.read_text().splitlines():
            try:
                r = json.loads(line)
            except Exception:
                continue
            raw.append(r)
    deduped = _dedupe(raw)
    # Filter to valid cids only AND inject `actual` from sidecar labels.
    out = []
    for r in deduped:
        cid = r.get("customer_id")
        if cid not in valid_cids:
            continue
        if "actual" not in r:
            r["actual"] = labels.get(cid, 0)
        out.append(r)
    return out


def signed_gap(records, key="purchased", label_key="actual", weights=None):
    if not records:
        return None
    a = np.array([r.get(key, 0) for r in records], dtype=float)
    b = np.array([r.get(label_key, 0) for r in records], dtype=float)
    if weights is None:
        return float(a.mean() - b.mean())
    return float(a @ weights - b @ weights)


def reweight_to_test(records, test_w):
    bs = [r["bucket"] for r in records]
    n_per = {b: sum(1 for x in bs if x == b) for b in set(bs)}
    w = np.array([test_w.get(b, 0) / max(n_per[b], 1) for b in bs])
    return w / max(w.sum(), 1e-9)


def stratified_bootstrap_gap(records, B=1000, seed=2026, test_weights=None):
    if not records:
        return None
    rng = np.random.default_rng(seed)
    by_b = {}
    for r in records:
        by_b.setdefault(r["bucket"], []).append(r)
    point = signed_gap(records)
    samples = []
    for _ in range(B):
        boot = []
        for b, grp in by_b.items():
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
    rhos, n_b = {}, {}
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
    return {
        "per_bucket_rho": rhos,
        "weighted_rho": sum(rhos[b] * n_b[b] for b in rhos) / max(total, 1),
        "n_per_bucket": n_b,
    }


def per_bucket(records):
    out = {}
    for b in BUCKETS:
        sub = [r for r in records if r.get("bucket") == b]
        if not sub:
            continue
        out[b] = {
            "n": len(sub),
            "funnel_rate": float(np.mean([r["purchased"] for r in sub])),
            "actual_rate": float(np.mean([r["actual"] for r in sub])),
            "gap": float(np.mean([r["purchased"] - r["actual"] for r in sub])),
        }
    return out


def paired_gap_diff(arm_a, arm_b, B=1000, seed=2026):
    """Paired bootstrap of (gap_A - gap_B) on common customer ids."""
    ca = {r["customer_id"]: r for r in arm_a}
    cb = {r["customer_id"]: r for r in arm_b}
    common = sorted(set(ca.keys()) & set(cb.keys()))
    if len(common) < 30:
        return None
    pa = [ca[c] for c in common]
    pb = [cb[c] for c in common]
    point = signed_gap(pa) - signed_gap(pb)
    rng = np.random.default_rng(seed)
    by_b = {}
    for i, r in enumerate(pa):
        by_b.setdefault(r["bucket"], []).append(i)
    samples = []
    for _ in range(B):
        idxs = []
        for b, items in by_b.items():
            idxs.extend(rng.choice(items, size=len(items), replace=True).tolist())
        ba = [pa[i] for i in idxs]
        bb = [pb[i] for i in idxs]
        samples.append(signed_gap(ba) - signed_gap(bb))
    samples = np.array(samples)
    return {
        "n_common": len(common),
        "point": point, "se": float(samples.std()),
        "lo": float(np.quantile(samples, 0.025)),
        "hi": float(np.quantile(samples, 0.975)),
        "B": B,
    }


def main():
    test = pl.read_parquet(ROOT / "data" / "splits" / "test.parquet")
    counts = test.group_by("activity_bucket").len().to_dicts()
    tot = sum(d["len"] for d in counts)
    test_w = {d["activity_bucket"]: d["len"] / tot for d in counts}

    result = {"test_weights": test_w, "arms": {}}

    # Arm A: Gemini per-DP, all methods
    print("=== Arm A: Gemini per-DP ===")
    arm_a = {}
    for m in GEMINI_METHODS:
        recs = load_gemini(m)
        if not recs:
            continue
        gap_rw = signed_gap(recs, weights=reweight_to_test(recs, test_w))
        boot = stratified_bootstrap_gap(recs, test_weights=test_w)
        wbs = within_bucket_spearman(recs)
        arm_a[m] = {
            "n": len(recs),
            "actual_rate": float(np.mean([r["actual"] for r in recs])),
            "funnel_rate": float(np.mean([r["purchased"] for r in recs])),
            "gap_reweighted": gap_rw,
            "bootstrap": boot,
            "within_bucket_rho": wbs,
            "per_bucket": per_bucket(recs),
        }
        print(f"  {m}: n={len(recs)} gap={gap_rw:+.3f} CI=[{boot['lo']:+.3f},{boot['hi']:+.3f}] wb_rho={wbs['weighted_rho']:+.3f}")
    result["arms"]["gemini_per_dp"] = arm_a

    # Arm B: Claude meta-policy
    print("\n=== Arm B: Claude meta-policy ===")
    arm_b = {}
    for m in META_METHODS:
        recs = load_claude_meta(m)
        if not recs:
            continue
        gap_rw = signed_gap(recs, weights=reweight_to_test(recs, test_w))
        boot = stratified_bootstrap_gap(recs, test_weights=test_w)
        wbs = within_bucket_spearman(recs)
        arm_b[m] = {
            "n": len(recs),
            "actual_rate": float(np.mean([r["actual"] for r in recs])),
            "funnel_rate": float(np.mean([r["purchased"] for r in recs])),
            "gap_reweighted": gap_rw,
            "bootstrap": boot,
            "within_bucket_rho": wbs,
            "per_bucket": per_bucket(recs),
        }
        print(f"  {m}: n={len(recs)} gap={gap_rw:+.3f} CI=[{boot['lo']:+.3f},{boot['hi']:+.3f}] wb_rho={wbs['weighted_rho']:+.3f}")
    result["arms"]["claude_meta_policy"] = arm_b

    # Arm C: Claude proper per-DP (M1 and S4)
    print("\n=== Arm C: Claude proper per-DP (M1 + S4) ===")
    arm_c = {}
    recs_m1 = load_claude_proper()
    if recs_m1:
        gap_rw = signed_gap(recs_m1, weights=reweight_to_test(recs_m1, test_w))
        boot = stratified_bootstrap_gap(recs_m1, test_weights=test_w)
        wbs = within_bucket_spearman(recs_m1)
        arm_c["M1"] = {
            "n": len(recs_m1),
            "actual_rate": float(np.mean([r["actual"] for r in recs_m1])),
            "funnel_rate": float(np.mean([r["purchased"] for r in recs_m1])),
            "gap_reweighted": gap_rw,
            "bootstrap": boot,
            "within_bucket_rho": wbs,
            "per_bucket": per_bucket(recs_m1),
        }
        print(f"  M1: n={len(recs_m1)} gap={gap_rw:+.3f} CI=[{boot['lo']:+.3f},{boot['hi']:+.3f}] wb_rho={wbs['weighted_rho']:+.3f}")
    # S4 consolidated file
    s4_fn = ROOT / "results" / "phase42_claude_proper_S4.jsonl"
    if s4_fn.exists():
        s4_recs = []
        for line in s4_fn.read_text().splitlines():
            try:
                s4_recs.append(json.loads(line))
            except Exception:
                pass
        if s4_recs:
            s4_recs = _dedupe(s4_recs)
            gap_rw = signed_gap(s4_recs, weights=reweight_to_test(s4_recs, test_w))
            boot = stratified_bootstrap_gap(s4_recs, test_weights=test_w)
            wbs = within_bucket_spearman(s4_recs)
            arm_c["S4"] = {
                "n": len(s4_recs),
                "actual_rate": float(np.mean([r["actual"] for r in s4_recs])),
                "funnel_rate": float(np.mean([r["purchased"] for r in s4_recs])),
                "gap_reweighted": gap_rw,
                "bootstrap": boot,
                "within_bucket_rho": wbs,
                "per_bucket": per_bucket(s4_recs),
            }
            print(f"  S4: n={len(s4_recs)} gap={gap_rw:+.3f} CI=[{boot['lo']:+.3f},{boot['hi']:+.3f}] wb_rho={wbs['weighted_rho']:+.3f}")
    result["arms"]["claude_per_dp"] = arm_c

    # Arm D: Claude proper per-DP under SANDBOX V2 (real world dynamics)
    print("\n=== Arm D: Claude proper per-DP under SANDBOX V2 (world model) ===")
    arm_d = {}
    v2_fn = ROOT / "results" / "phase46_sandbox_v2_M1.jsonl"
    if v2_fn.exists():
        v2_recs = []
        for line in v2_fn.read_text().splitlines():
            try:
                v2_recs.append(json.loads(line))
            except Exception:
                pass
        if v2_recs:
            v2_recs = _dedupe(v2_recs)
            gap_rw = signed_gap(v2_recs, weights=reweight_to_test(v2_recs, test_w))
            boot = stratified_bootstrap_gap(v2_recs, test_weights=test_w)
            wbs = within_bucket_spearman(v2_recs)
            arm_d["M1"] = {
                "n": len(v2_recs),
                "actual_rate": float(np.mean([r["actual"] for r in v2_recs])),
                "funnel_rate": float(np.mean([r["purchased"] for r in v2_recs])),
                "gap_reweighted": gap_rw,
                "bootstrap": boot,
                "within_bucket_rho": wbs,
                "per_bucket": per_bucket(v2_recs),
            }
            print(f"  M1 (world v2): n={len(v2_recs)} gap={gap_rw:+.3f} CI=[{boot['lo']:+.3f},{boot['hi']:+.3f}] wb_rho={wbs['weighted_rho']:+.3f}")
    result["arms"]["claude_per_dp_world_v2"] = arm_d

    # Paired comparisons on M1
    print("\n=== Paired M1 comparisons (common customers only) ===")
    pair_a_b = paired_gap_diff(load_gemini("M1"), load_claude_meta("M1"))
    pair_a_c = paired_gap_diff(load_gemini("M1"), load_claude_proper())
    pair_b_c = paired_gap_diff(load_claude_meta("M1"), load_claude_proper())
    result["paired_m1"] = {
        "gemini_per_dp__vs__claude_meta": pair_a_b,
        "gemini_per_dp__vs__claude_per_dp": pair_a_c,
        "claude_meta__vs__claude_per_dp": pair_b_c,
    }
    for name, p in result["paired_m1"].items():
        if p:
            print(f"  {name}: n={p['n_common']} diff={p['point']:+.3f} CI=[{p['lo']:+.3f},{p['hi']:+.3f}]")
        else:
            print(f"  {name}: insufficient common customers")

    OUT.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
