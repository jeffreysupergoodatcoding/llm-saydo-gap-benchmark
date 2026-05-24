"""Phase 36: figures for v3 paper.

Produces:
  fig_v3_1_method_gap_bar.png — per-method sandbox signed gap + bootstrap CI
  fig_v3_2_within_bucket_rho.png — within-bucket Spearman by method
  fig_v3_3_commitment_shrinkage.png — commitment shrinkage with permutation null band
  fig_v3_4_per_bucket_heatmap.png — per-method × per-bucket signed gap heatmap
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.size": 11,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "figure.dpi": 110,
    "savefig.dpi": 200,
})

METHODS = ["M1", "M3", "M8", "M9", "S1", "S2", "S3", "S4"]
COLOURS = {"M1": "#444444", "M3": "#666666", "M8": "#888888", "M9": "#aaaaaa",
           "S1": "#1f77b4", "S2": "#2ca02c", "S3": "#d62728", "S4": "#9467bd"}


def load():
    return json.loads((ROOT / "results" / "phase35_v3_analysis.json").read_text())


def fig1_method_gap_bar():
    r = load()
    methods, gaps, los, his = [], [], [], []
    for m in METHODS:
        v = r["methods"].get(m)
        if not v:
            continue
        ci = v["sandbox_gap_bootstrap"]
        methods.append(m)
        gaps.append(v["sandbox_signed_gap_reweighted"])
        los.append(ci["lo"])
        his.append(ci["hi"])
    methods, gaps, los, his = np.array(methods), np.array(gaps), np.array(los), np.array(his)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.axhspan(-0.05, 0.05, color="#dde", alpha=0.35, label="H10 envelope ±0.05")
    ax.axhline(0, color="black", lw=0.8)
    for i, m in enumerate(methods):
        ax.errorbar(i, gaps[i], yerr=[[gaps[i] - los[i]], [his[i] - gaps[i]]],
                    fmt="o", color=COLOURS.get(m, "k"), capsize=4, markersize=6)
        ax.text(i, his[i] + 0.005, m, ha="center", fontsize=9)
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods)
    ax.set_ylabel("Sandbox signed gap (reweighted)")
    ax.set_title("Per-method funnel-realized purchase rate − actual rate, paired bootstrap 95% CI")
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG / "fig_v3_1_method_gap_bar.png")
    plt.close(fig)


def fig2_within_bucket_rho():
    r = load()
    methods, rhos = [], []
    for m in METHODS:
        v = r["methods"].get(m)
        if not v:
            continue
        methods.append(m)
        rhos.append(v["within_bucket_rho"]["weighted_rho"])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(methods, rhos, color=[COLOURS.get(m, "k") for m in methods], alpha=0.8)
    ax.axhline(0, color="black", lw=0.8)
    # human-self anchor reference line
    ax.axhline(0.39, color="green", ls="--", lw=1, label="human-self anchor r=0.39 (Phase 24)")
    ax.axhline(0.23, color="orange", ls=":", lw=1, label="prior LLM within-ρ ≈0.23 (v2)")
    ax.set_ylabel("Bucket-weighted within-bucket Spearman ρ")
    ax.set_title("Within-bucket Spearman correlation: agent vs actual purchase by method")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG / "fig_v3_2_within_bucket_rho.png")
    plt.close(fig)


def fig3_commitment_shrinkage():
    r = load()
    methods, cs, p_vals, lo, hi = [], [], [], [], []
    for m in METHODS:
        v = r["methods"].get(m, {})
        cs_obj = v.get("commitment_shrinkage")
        if not cs_obj:
            continue
        methods.append(m)
        cs.append(cs_obj["commitment_shrinkage"])
        p_vals.append(cs_obj.get("permutation_p", 1.0))
        lo.append(cs_obj.get("null_lo", 0))
        hi.append(cs_obj.get("null_hi", 0))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(methods))
    ax.bar(x, cs, color=[COLOURS.get(m, "k") for m in methods], alpha=0.8, label="observed")
    # Plot null bands (mean ± 95% null)
    null_lo, null_hi = np.array(lo), np.array(hi)
    ax.fill_between([-0.5, len(methods) - 0.5],
                    [null_lo.min()] * 2, [null_hi.max()] * 2,
                    color="#bbb", alpha=0.3, label="permutation null 95% range")
    ax.axhline(0, color="black", lw=0.8)
    for i, p in enumerate(p_vals):
        ax.text(i, cs[i] + 0.005, f"p={p:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Commitment shrinkage (scalar_gap − sandbox_gap)")
    ax.set_title("Commitment shrinkage vs per-bucket permutation null")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig_v3_3_commitment_shrinkage.png")
    plt.close(fig)


def fig4_per_bucket_heatmap():
    r = load()
    BUCKETS = ["1", "2-5", "6-20", "21-100", "101+"]
    M = []
    for m in METHODS:
        v = r["methods"].get(m, {})
        per_b = v.get("per_bucket", {})
        row = [per_b.get(b, {}).get("gap", np.nan) for b in BUCKETS]
        if any(not np.isnan(x) for x in row):
            M.append((m, row))
    methods, rows = zip(*M)
    arr = np.array(rows)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    vmax = max(0.3, np.nanmax(np.abs(arr)))
    im = ax.imshow(arr, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(BUCKETS)))
    ax.set_xticklabels(BUCKETS)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            ax.text(j, i, f"{v:+.02f}" if not np.isnan(v) else "—",
                    ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > vmax * 0.5 else "black")
    fig.colorbar(im, ax=ax, label="signed gap")
    ax.set_xlabel("Activity bucket")
    ax.set_title("Per-method × per-bucket signed gap (red = LLM over-predicts)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_v3_4_per_bucket_heatmap.png")
    plt.close(fig)


def main():
    fig1_method_gap_bar()
    fig2_within_bucket_rho()
    fig3_commitment_shrinkage()
    fig4_per_bucket_heatmap()
    print(f"Figures saved to {FIG}")


if __name__ == "__main__":
    main()
