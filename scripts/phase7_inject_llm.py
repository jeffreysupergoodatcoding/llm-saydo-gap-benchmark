"""Phase 7b: Inject the LLM (D2/D3) numbers into the manual report.md.

Reads results/phase4b_D2.json (+ optional D3) and updates the placeholders in report.md.
"""

from __future__ import annotations
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORT = ROOT / "report.md"


def fmt_ci(metric: dict) -> str:
    return f"{metric['point']:.4f} [{metric['lo']:.4f}, {metric['hi']:.4f}]"


def main():
    d2 = None
    d3 = None
    p = RESULTS / "phase4b_D2.json"
    if p.exists():
        d2 = json.loads(p.read_text())
    p = RESULTS / "phase4b_D3.json"
    if p.exists():
        d3 = json.loads(p.read_text())

    if d2 is None:
        print("[7b] no D2 result yet; nothing to inject")
        return

    text = REPORT.read_text()
    n = d2.get("n", "n=?")
    pr = d2["metrics"]["pr_auc"]
    roc = d2["metrics"]["roc_auc"]
    brier = d2["metrics"]["brier"]
    ud = d2["metrics"].get("under_dispersion", {})

    # Replace LLM row TBDs in the head-to-head table
    text = re.sub(
        r"\| \*\*D — LLM Digital Twin \(D2.*?TBD.*?\*TBD\*.*?\|",
        f"| **D — LLM Digital Twin (D2, n={d2['n']}, Gemini 2.5 Flash)** | {fmt_ci(pr)} | {roc['point']:.4f} | {brier['point']:.4f} |",
        text,
        count=1,
    )
    if d3:
        prd3 = d3["metrics"]["pr_auc"]
        rocd3 = d3["metrics"]["roc_auc"]
        brid3 = d3["metrics"]["brier"]
        text = re.sub(
            r"\| D — D3 \(reflection ablation.*?\| \*TBD\*.*?\|",
            f"| D — D3 (reflection ablation, n={d3['n']}) | {fmt_ci(prd3)} | {rocd3['point']:.4f} | {brid3['point']:.4f} |",
            text,
            count=1,
        )

    # Under-dispersion row
    if ud:
        text = re.sub(
            r"\| D — LLM D2 \| \*TBD\* \| \*TBD\* \|",
            f"| D — LLM D2 | {ud.get('ratio', float('nan')):.2f} | {ud.get('levene_p', float('nan')):.2e} |",
            text,
            count=1,
        )

    # H1 / H2 verdict — rely on phase4c JSON
    p4c = RESULTS / "phase4c_regime.json"
    if p4c.exists():
        rj = json.loads(p4c.read_text())
        by_bucket = rj.get("by_bucket", {})
        # H1: bucket-1 and 2-5 best-classical-on-5k vs D_D2
        h1 = "Pending data"
        h2 = "Pending data"

        def best_classical_in(bucket_d):
            cs = [(k, v.get("pr_auc")) for k, v in bucket_d.items() if k.endswith("_on_5k")]
            if not cs:
                return None
            return max(cs, key=lambda x: x[1])

        if "1" in by_bucket and "D_D2" in by_bucket["1"]:
            bc = best_classical_in(by_bucket["1"])
            d_pr = by_bucket["1"]["D_D2"]["pr_auc"]
            if bc:
                delta = d_pr - bc[1]
                h1 = f"D_D2 = {d_pr:.3f}, best classical ({bc[0].replace('_on_5k','')}) = {bc[1]:.3f}, Δ = {delta:+.3f} ({'CONFIRMED' if delta >= 0.02 else 'REFUTED'})"
        if "101+" in by_bucket and "D_D2" in by_bucket["101+"]:
            bc = best_classical_in(by_bucket["101+"])
            d_pr = by_bucket["101+"]["D_D2"]["pr_auc"]
            if bc:
                delta = bc[1] - d_pr
                h2 = f"best classical ({bc[0].replace('_on_5k','')}) = {bc[1]:.3f}, D_D2 = {d_pr:.3f}, Δ = {delta:+.3f} ({'CONFIRMED' if delta >= 0.02 else 'REFUTED'})"

        text = re.sub(r"\| H1 \|.*?\|", f"| H1 | LLM twin beats best classical at ≤ 4 events, Δ ≥ 0.02 | {h1} |", text, count=1)
        text = re.sub(r"\| H2 \|.*?\|", f"| H2 | Best classical beats LLM twin at ≥ 16 events, Δ ≥ 0.02 | {h2} |", text, count=1)

    REPORT.write_text(text)
    print(f"[7b] Updated {REPORT}")


if __name__ == "__main__":
    main()
