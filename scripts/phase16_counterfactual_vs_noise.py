"""Phase 16: proper statistical test for counterfactual vs noise floor.

Recomputes noise-floor as 2-run |Δ| distribution (apples-to-apples with the
counterfactual perturbation), then runs:
  - Mann-Whitney U (one-sided: perturbation > noise)
  - Bootstrap 95% CI on the difference of means
  - Cliff's δ (effect size)
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from itertools import combinations
from scipy.stats import mannwhitneyu

from src import SEED

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    # # pairs (a_i > b_j) − # pairs (a_i < b_j) / (n_a * n_b)
    ax, bx = np.asarray(a)[:, None], np.asarray(b)[None, :]
    gt = (ax > bx).sum()
    lt = (ax < bx).sum()
    return float((gt - lt) / (len(a) * len(b)))


def main():
    counter = json.loads((RESULTS / "phase11_counterfactual.json").read_text())
    noise = json.loads((RESULTS / "phase13_noise_floor.json").read_text())

    cf_deltas = np.array([abs(s["delta_intent"]) for s in counter["samples"]
                          if isinstance(s, dict) and "delta_intent" in s])
    # If we only have first-20 samples in json, that's enough.
    # Better: re-derive from per-sample list if larger
    cf_n = counter.get("n_perturbed", len(cf_deltas))

    # Reconstruct 2-run |Δ| pairs from noise floor samples
    noise_pairs = []
    for s in noise["samples"]:
        if not isinstance(s, dict) or "runs" not in s:
            continue
        runs = s["runs"]
        for i, j in combinations(range(len(runs)), 2):
            noise_pairs.append(abs(runs[i] - runs[j]))
    noise_pairs = np.array(noise_pairs)

    # Mann-Whitney U one-sided: cf > noise
    u, p_one = mannwhitneyu(cf_deltas, noise_pairs, alternative="greater")
    # Difference of means with bootstrap CI
    rng = np.random.default_rng(SEED)
    B = 1000
    diffs_boot = []
    for _ in range(B):
        a = rng.choice(cf_deltas, len(cf_deltas), replace=True)
        b = rng.choice(noise_pairs, len(noise_pairs), replace=True)
        diffs_boot.append(a.mean() - b.mean())
    diffs_boot = np.array(diffs_boot)
    delta_cliff = cliffs_delta(cf_deltas, noise_pairs)

    out = {
        "n_cf_pairs": len(cf_deltas),
        "n_noise_pairs": len(noise_pairs),
        "mean_cf": float(cf_deltas.mean()),
        "mean_noise_pairs": float(noise_pairs.mean()),
        "diff_of_means": float(cf_deltas.mean() - noise_pairs.mean()),
        "diff_of_means_95CI": [float(np.quantile(diffs_boot, 0.025)),
                                 float(np.quantile(diffs_boot, 0.975))],
        "mann_whitney_U": float(u),
        "p_one_sided_cf_gt_noise": float(p_one),
        "cliffs_delta": delta_cliff,
        # If perturbation is NOT significantly greater than noise, the LLM is anchoring.
        "anchoring_to_priors_inferential": (p_one > 0.05),
    }
    (RESULTS / "phase16_cf_vs_noise.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"[16] mean cf={out['mean_cf']:.4f}  vs noise-pairs={out['mean_noise_pairs']:.4f}")
    print(f"[16] diff = {out['diff_of_means']:+.4f}, 95% CI {out['diff_of_means_95CI']}")
    print(f"[16] Mann-Whitney U one-sided (cf > noise): p = {out['p_one_sided_cf_gt_noise']:.4f}")
    print(f"[16] Cliff's δ = {out['cliffs_delta']:+.3f}")
    print(f"[16] anchoring_to_priors (inferential, p>0.05 fail to reject equality): {out['anchoring_to_priors_inferential']}")


if __name__ == "__main__":
    main()
