"""Phase 19: Spearman ρ(stated_intent_prob, actual_label) per arm.

Pre-registered secondary metric (preregistration_v2.md §86); not previously
computed. Reports pooled, within-bucket-pooled, and per-bucket Spearman with
bootstrap 95% CIs. Sheeran 2002 meta-analytic r ≈ 0.53 is the
across-individuals comparator from social psychology.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

from src import SEED

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BUCKETS_ORDER = ["1", "2-5", "6-20", "21-100", "101+"]


def _load(name: str):
    d = np.load(RESULTS / f"phase10_{name}_scores.npz", allow_pickle=True)
    return {
        "scores": d["stated_intent_raw"].astype(float) if "stated_intent_raw" in d.files else d["stated_intent"].astype(float),
        "actual": d["actual"].astype(int),
        "buckets": d["activity_bucket"].astype(str),
    }


def _bootstrap_spearman(scores, actual, B=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(scores)
    boots = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        rho, _ = spearmanr(scores[idx], actual[idx])
        if np.isfinite(rho):
            boots.append(rho)
    boots = np.array(boots)
    rho_point, p = spearmanr(scores, actual)
    return {
        "rho": float(rho_point) if np.isfinite(rho_point) else None,
        "p": float(p) if np.isfinite(p) else None,
        "lo": float(np.quantile(boots, 0.025)),
        "hi": float(np.quantile(boots, 0.975)),
        "B": B,
    }


def main():
    arms = {}
    for name in ["F-base", "F-nobase", "D2-core"]:
        p = RESULTS / f"phase10_{name}_scores.npz"
        if not p.exists():
            continue
        arms[name] = _load(name)

    out = {}
    SHEERAN_R = 0.53
    for name, a in arms.items():
        pooled = _bootstrap_spearman(a["scores"], a["actual"])
        # Within-bucket: standardize within strata then pool
        residuals_s = np.zeros_like(a["scores"], dtype=float)
        residuals_a = np.zeros_like(a["actual"], dtype=float)
        for b in BUCKETS_ORDER:
            mask = a["buckets"] == b
            if mask.sum() == 0:
                continue
            residuals_s[mask] = a["scores"][mask] - a["scores"][mask].mean()
            residuals_a[mask] = a["actual"][mask] - a["actual"][mask].mean()
        within = _bootstrap_spearman(residuals_s, residuals_a)

        # Per-bucket
        per_bucket = {}
        for b in BUCKETS_ORDER:
            mask = a["buckets"] == b
            if mask.sum() < 20:
                continue
            rho, p = spearmanr(a["scores"][mask], a["actual"][mask])
            per_bucket[b] = {"rho": float(rho) if np.isfinite(rho) else None,
                              "p": float(p) if np.isfinite(p) else None,
                              "n": int(mask.sum())}

        out[name] = {
            "pooled_spearman": pooled,
            "within_bucket_pooled_spearman": within,
            "per_bucket_spearman": per_bucket,
            "sheeran_r_comparator": SHEERAN_R,
            "pooled_minus_sheeran": (pooled["rho"] - SHEERAN_R) if pooled["rho"] is not None else None,
            "within_minus_sheeran": (within["rho"] - SHEERAN_R) if within["rho"] is not None else None,
        }
        print(f"[19] {name}: pooled ρ = {pooled['rho']:.3f} [{pooled['lo']:+.3f}, {pooled['hi']:+.3f}]  "
              f"within-bucket ρ = {within['rho']:+.3f} [{within['lo']:+.3f}, {within['hi']:+.3f}]")
    (RESULTS / "phase19_spearman.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"[19] Sheeran 2002 reference r ≈ {SHEERAN_R} (across-individual intent-behavior, social-psych domain).")


if __name__ == "__main__":
    main()
