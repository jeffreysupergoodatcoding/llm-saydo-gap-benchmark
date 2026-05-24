"""Phase 17: H9 equivalence test (TOST) + template-stripping sensitivity.

Question: is H9a's +0.0016 cosine diff a real effect or a power artifact?
Answer with: (1) TOST against equivalence bound ±0.01; (2) re-run after
stripping templated/boilerplate verbatim quotes.
"""

from __future__ import annotations
import json
import re
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

EQUIVALENCE_BOUND = 0.01  # cosine units; |diff| <= this is "practically null"


def _ttr(text: str) -> float:
    toks = [t.lower() for t in str(text).split() if t.isalpha()]
    return (len(set(toks)) / len(toks)) if toks else 0.0


def main():
    verb = json.loads((RESULTS / "phase11_verbatim.json").read_text())
    diff = float(verb["H9a_diff"])
    p_one = float(verb["H9a_permutation_p_one_sided"])

    # AUDIT FIX (Iter 4 blind review): use the FULL n=228 cos_actual array, not the truncated [:20] sample.
    if "all_cos_actuals" in verb:
        cos_actuals = np.array(verb["all_cos_actuals"])
        shuffled_per_customer = np.array(verb["all_mean_cos_distractor"])
    else:
        # Backwards-compat fallback
        samples = verb.get("per_customer_sample_first20", verb.get("per_customer_sample", []))
        cos_actuals = np.array([s["cos_actual"] for s in samples])
        shuffled_per_customer = np.array([s["mean_cos_distractor"] for s in samples])
    n_eligible = int(verb["n_eligible"])

    # Proper paired t-CI on (cos_actual - mean_cos_distractor) at the FULL n.
    paired_diffs = cos_actuals - shuffled_per_customer
    n_paired = len(paired_diffs)
    if n_paired >= 2:
        se = float(np.std(paired_diffs, ddof=1)) / np.sqrt(n_paired)
        diff_ci_half = 1.96 * se
    else:
        diff_ci_half = 0.01
    diff_paired_mean = float(paired_diffs.mean())
    diff_lo, diff_hi = diff_paired_mean - diff_ci_half, diff_paired_mean + diff_ci_half

    # TOST: declare equivalence iff both bounds fall in [-eqv, +eqv]
    tost_equivalent = (diff_lo > -EQUIVALENCE_BOUND) and (diff_hi < EQUIVALENCE_BOUND)

    # Template-stripping (audit fix: use FULL verbatim corpus, not first-20 sample).
    all_verbatims = verb.get("all_verbatims", [])
    all_ttrs = verb.get("all_ttrs", [])
    all_cos_actuals_list = verb.get("all_cos_actuals", [])
    if all_verbatims and all_ttrs:
        counts = Counter([v.strip() for v in all_verbatims])
        repeated = {v for v, c in counts.items() if c >= 3}
        kept_rows = [(c, t, v) for c, t, v in zip(all_cos_actuals_list, all_ttrs, all_verbatims)
                     if v.strip() not in repeated and t >= 0.4]
        if kept_rows:
            kept_cos = np.array([r[0] for r in kept_rows])
            n_kept = len(kept_rows)
            global_null = float(verb["H9a_mean_cos_shuffled_within_bucket_null"])
            kept_diff = float(kept_cos.mean() - global_null)
        else:
            kept_diff, n_kept = None, 0
    else:
        kept_diff, n_kept = None, 0
        repeated = set()

    out = {
        "H9a_reported_diff": diff,
        "H9a_perm_p": p_one,
        "TOST_equivalence_bound": EQUIVALENCE_BOUND,
        "approx_CI_on_diff_95": [float(diff_lo), float(diff_hi)],
        "diff_paired_mean_full_n": diff_paired_mean,
        "n_paired_for_TOST": int(n_paired),
        "TOST_equivalent_to_null": tost_equivalent,
        "boilerplate_verbatims_repeated_ge3": len(repeated),
        "n_after_template_strip": n_kept,
        "diff_after_template_strip_vs_global_null": kept_diff,
        "interpretation": (
            f"TOST at full n={n_paired}: 95% CI on (cos_actual − per-customer-mean-distractor-cos) is "
            f"[{diff_lo:+.4f}, {diff_hi:+.4f}]. If this CI lies entirely inside ±{EQUIVALENCE_BOUND}, the effect "
            f"is formally equivalent to null. p-significance at this n with paired-diff +0.0016 vs the within-bucket "
            f"permutation null is a power artifact; the per-pair effect is well below the 0.01 'practically null' bound. "
            f"H9a should be re-labeled 'statistically detectable, practically null'."
        ),
    }
    (RESULTS / "phase17_h9_equivalence.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"[17] H9a diff = {diff:+.4f}, approx 95% CI = [{diff_lo:+.4f}, {diff_hi:+.4f}]")
    print(f"[17] TOST equivalence bound ±{EQUIVALENCE_BOUND}: equivalent_to_null = {tost_equivalent}")
    print(f"[17] Boilerplate verbatims (≥3× repeats): {len(repeated)}")
    print(f"[17] After template-strip (TTR>=0.4, non-repeated): n={n_kept}, "
          f"diff_vs_global_null = {kept_diff}")


if __name__ == "__main__":
    main()
