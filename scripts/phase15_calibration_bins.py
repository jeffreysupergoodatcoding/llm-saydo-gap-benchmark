"""Phase 15: per-decile calibration with bootstrap CIs (analysis only, no LLM calls).

For each arm, bin customers into 10 deciles of stated_intent_prob, and for each
decile report (mean stated, actual rate, signed gap, n) with bootstrap CIs.
This produces the headline calibration curve and supports the discussion of
under-dispersion at the bin grain.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from src import SEED

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _bootstrap_bin_rate(actual: np.ndarray, B: int = 1000, seed: int = SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if len(actual) == 0:
        return 0.0, 0.0
    boots = np.array([rng.choice(actual, size=len(actual), replace=True).mean() for _ in range(B)])
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def analyse_arm(path: Path, name: str, score_key: str = "stated_intent_raw"):
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    if score_key in d.files:
        scores = d[score_key].astype(float)
    else:
        scores = d["scores" if "scores" in d.files else "stated_intent"].astype(float)
    actual = d["y_test" if "y_test" in d.files else "actual"].astype(int)
    n = len(scores)
    # Use 10 deciles via quantile binning
    edges = np.quantile(scores, np.linspace(0, 1, 11))
    edges[0] = -0.001
    edges[-1] = 1.001
    bin_idx = np.digitize(scores, edges) - 1
    bin_idx = np.clip(bin_idx, 0, 9)
    rows = []
    for b in range(10):
        m = bin_idx == b
        if m.sum() == 0:
            continue
        bs_lo, bs_hi = _bootstrap_bin_rate(actual[m])
        rows.append({
            "bin": int(b),
            "n": int(m.sum()),
            "mean_pred": float(scores[m].mean()),
            "actual_rate": float(actual[m].mean()),
            "actual_rate_CI_lo": bs_lo,
            "actual_rate_CI_hi": bs_hi,
            "signed_gap": float(scores[m].mean() - actual[m].mean()),
        })
    return {"name": name, "n": n, "deciles": rows}


def main():
    arms = []
    for name, score_key in [("F-base", "stated_intent_raw"),
                            ("F-nobase", "stated_intent_raw")]:
        p = RESULTS / f"phase10_{name}_scores.npz"
        a = analyse_arm(p, name, score_key=score_key)
        if a:
            arms.append(a)
    # Reuse D2 if present
    p_d2 = RESULTS / "phase4b_D2_scores.npz"
    a_d2 = analyse_arm(p_d2, "D2", score_key="scores")
    if a_d2:
        arms.append(a_d2)

    out = {"arms": {a["name"]: a for a in arms}}
    (RESULTS / "phase15_calibration_bins.json").write_text(json.dumps(out, indent=2, default=str))

    # Figure: reliability with CI ribbons
    fig, ax = plt.subplots(figsize=(7, 6))
    for a in arms:
        xs = [r["mean_pred"] for r in a["deciles"]]
        ys = [r["actual_rate"] for r in a["deciles"]]
        lo = [r["actual_rate_CI_lo"] for r in a["deciles"]]
        hi = [r["actual_rate_CI_hi"] for r in a["deciles"]]
        ax.plot(xs, ys, "o-", label=a["name"], alpha=0.85)
        ax.fill_between(xs, lo, hi, alpha=0.15)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect calibration")
    ax.set_xlabel("Mean stated intent in decile")
    ax.set_ylabel("Actual rate (with 95% bootstrap CI)")
    ax.set_title("Per-decile calibration with bootstrap CIs")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "phase15_calibration_decile.png", dpi=130)
    plt.close()
    print(f"[15] wrote {RESULTS / 'phase15_calibration_bins.json'} + figure")
    for a in arms:
        print(f"\n{a['name']}:  (n={a['n']})")
        for r in a["deciles"]:
            print(f"  decile {r['bin']}: n={r['n']:4d} pred={r['mean_pred']:.3f} actual={r['actual_rate']:.3f} "
                  f"[CI {r['actual_rate_CI_lo']:.3f}, {r['actual_rate_CI_hi']:.3f}] gap={r['signed_gap']:+.3f}")


if __name__ == "__main__":
    main()
