"""Phase 45: 5 figures for the v3 ICLR paper.

Reads results/phase43_cross_provider_analysis.json and results/phase41_claude_analysis.json
and renders:

  fig_v3_1_three_arm_m1.png        — 3-arm M1 comparison (gap + wb-ρ)
  fig_v3_2_method_catalog_gaps.png — per-method gaps across 8 Claude methods + CIs
  fig_v3_3_per_bucket_heatmap.png  — per-method × per-bucket gap heatmap
  fig_v3_4_commitment_shrinkage.png— scalar vs sandbox gap by method
  fig_v3_5_within_bucket_rho.png   — within-bucket Spearman by method + cross-arm
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.size": 11, "axes.spines.right": False, "axes.spines.top": False,
    "figure.dpi": 110, "savefig.dpi": 200, "axes.labelweight": "bold",
})

METHODS = ["M1", "M3", "M8", "M9", "S1", "S2", "S3", "S4"]
BUCKETS = ["1", "2-5", "6-20", "21-100", "101+"]
M_COLOURS = {"M1": "#444", "M3": "#777", "M8": "#aa6", "M9": "#bbb",
             "S1": "#1f77b4", "S2": "#2ca02c", "S3": "#d62728", "S4": "#9467bd"}
ARM_COLOURS = {"gemini_per_dp": "#1f77b4", "claude_meta_policy": "#ff7f0e",
               "claude_per_dp": "#2ca02c"}


def load_cross():
    return json.loads((ROOT / "results" / "phase43_cross_provider_analysis.json").read_text())


def load_meta():
    return json.loads((ROOT / "results" / "phase41_claude_analysis.json").read_text())


def fig1_three_arm_m1():
    cross = load_cross()
    arms = ["gemini_per_dp", "claude_meta_policy", "claude_per_dp"]
    labels = ["Gemini per-DP\n(n=1000)", "Claude meta-policy\n(n=1000)",
              "Claude per-DP\n(n=200)"]
    gaps, gap_lo, gap_hi, rhos = [], [], [], []
    for a in arms:
        m = cross["arms"][a]["M1"]
        b = m["bootstrap"]
        gaps.append(m["gap_reweighted"])
        gap_lo.append(b["lo"])
        gap_hi.append(b["hi"])
        rhos.append(m["within_bucket_rho"]["weighted_rho"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    # Left panel: signed gap
    xs = np.arange(len(arms))
    ax1.axhspan(-0.05, 0.05, color="#dde", alpha=0.4, label="H10 envelope ±0.05")
    ax1.axhline(0, color="black", lw=0.8)
    for i, a in enumerate(arms):
        c = ARM_COLOURS[a]
        ax1.errorbar(i, gaps[i], yerr=[[gaps[i] - gap_lo[i]], [gap_hi[i] - gaps[i]]],
                     fmt="o", color=c, capsize=5, markersize=10)
        ax1.text(i, gap_hi[i] + 0.02, f"{gaps[i]:+.3f}", ha="center", fontsize=10)
    ax1.set_xticks(xs); ax1.set_xticklabels(labels)
    ax1.set_ylabel("Sandbox signed gap (reweighted)")
    ax1.set_title("(a) Population-mean gap")
    ax1.legend(loc="upper right", fontsize=9)

    # Right panel: within-bucket Spearman
    bars = ax2.bar(xs, rhos, color=[ARM_COLOURS[a] for a in arms], alpha=0.85)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.axhline(0.39, color="green", ls="--", lw=1.2, alpha=0.7,
                label="human-self r=0.39 (v2)")
    ax2.axhline(0.23, color="orange", ls=":", lw=1.2, alpha=0.7,
                label="LLM v2 ceiling ρ≈0.23")
    for i in range(len(arms)):
        ax2.text(i, rhos[i] + 0.012, f"{rhos[i]:+.3f}", ha="center", fontsize=10)
    ax2.set_xticks(xs); ax2.set_xticklabels(labels)
    ax2.set_ylabel("Within-bucket Spearman ρ")
    ax2.set_title("(b) Individual-conditioning signal")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.set_ylim(-0.1, 0.45)

    fig.suptitle("Three operationalizations of LLM-as-twin on M1 zero-shot — "
                 "the within-bucket gap is dominated by methodology, not provider",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "fig_v3_1_three_arm_m1.png", bbox_inches="tight")
    plt.close(fig)


def fig2_method_catalog():
    meta = load_meta()
    actual = meta["methods"]["M1"]["actual_rate"]
    methods = []
    gaps, gap_lo, gap_hi = [], [], []
    h10 = []
    for m in METHODS:
        v = meta["methods"].get(m)
        if not v:
            continue
        b = v["sandbox_gap_bootstrap"]
        methods.append(m)
        gaps.append(v["sandbox_signed_gap_reweighted"])
        gap_lo.append(b["lo"])
        gap_hi.append(b["hi"])
        h10.append(v["H10_pass"])
    xs = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axhspan(-0.05, 0.05, color="#dde", alpha=0.4, label="H10 envelope ±0.05")
    ax.axhline(0, color="black", lw=0.8)
    for i, m in enumerate(methods):
        c = M_COLOURS.get(m, "k")
        ax.errorbar(i, gaps[i], yerr=[[gaps[i] - gap_lo[i]], [gap_hi[i] - gaps[i]]],
                    fmt="o", color=c, capsize=5, markersize=8)
        ax.text(i, gap_hi[i] + 0.02, f"{gaps[i]:+.3f}", ha="center", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels(methods)
    ax.set_ylabel("Sandbox signed gap (reweighted)")
    ax.set_title("Per-method sandbox gap (Claude meta-policy arm, n=1000 each)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig_v3_2_method_catalog_gaps.png")
    plt.close(fig)


def fig3_per_bucket_heatmap():
    meta = load_meta()
    arr = np.full((len(METHODS), len(BUCKETS)), np.nan)
    for i, m in enumerate(METHODS):
        v = meta["methods"].get(m)
        if not v:
            continue
        for j, b in enumerate(BUCKETS):
            cell = v.get("per_bucket", {}).get(b)
            if cell:
                arr[i, j] = cell["gap"]
    fig, ax = plt.subplots(figsize=(9, 5))
    vmax = max(0.3, np.nanmax(np.abs(arr)))
    im = ax.imshow(arr, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(BUCKETS))); ax.set_xticklabels(BUCKETS)
    ax.set_yticks(np.arange(len(METHODS))); ax.set_yticklabels(METHODS)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.02f}", ha="center", va="center", fontsize=9,
                        color="white" if abs(v) > vmax * 0.55 else "black")
    cb = fig.colorbar(im, ax=ax, label="signed gap")
    ax.set_xlabel("Activity bucket")
    ax.set_title("Per-method × per-bucket signed gap (Claude meta-policy, n=1000)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_v3_3_per_bucket_heatmap.png")
    plt.close(fig)


def fig4_commitment_shrinkage():
    meta = load_meta()
    methods, scalar_gap, sandbox_gap, perm_p = [], [], [], []
    for m in METHODS:
        cs = meta["methods"].get(m, {}).get("commitment_shrinkage")
        if not cs:
            continue
        methods.append(m)
        scalar_gap.append(cs["scalar_gap"])
        sandbox_gap.append(cs["sandbox_gap"])
        perm_p.append(cs.get("permutation_p", 1.0))
    xs = np.arange(len(methods))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 4.6))
    b1 = ax.bar(xs - w/2, scalar_gap, w, label="Scalar gap", color="#888", alpha=0.85)
    b2 = ax.bar(xs + w/2, sandbox_gap, w, label="Sandbox gap", color="#1f77b4", alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    for i, m in enumerate(methods):
        ax.annotate("", xy=(i + w/2, sandbox_gap[i]), xytext=(i - w/2, scalar_gap[i]),
                    arrowprops=dict(arrowstyle="-|>", color="red", lw=1.2, alpha=0.6))
    ax.set_xticks(xs); ax.set_xticklabels(methods)
    ax.set_ylabel("Signed gap")
    ax.set_title("Commitment shrinkage: sandbox gap exceeds scalar gap for every method\n"
                 "(arrow = within-method increase; permutation p=1.000 in all cases — "
                 "structural commitment AMPLIFIES the gap, opposite of the naive prediction)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig_v3_4_commitment_shrinkage.png", bbox_inches="tight")
    plt.close(fig)


def fig5_within_bucket_rho():
    meta = load_meta()
    methods, rhos = [], []
    for m in METHODS:
        v = meta["methods"].get(m, {})
        rho = v.get("within_bucket_rho", {}).get("weighted_rho")
        if rho is None:
            continue
        methods.append(m)
        rhos.append(rho)
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.bar(methods, rhos, color=[M_COLOURS.get(m, "k") for m in methods], alpha=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(0.39, color="green", ls="--", lw=1.2, alpha=0.7, label="human-self r=0.39")
    ax.axhline(0.23, color="orange", ls=":", lw=1.2, alpha=0.7, label="proper-Claude per-DP ceiling ρ=0.23")
    for i in range(len(methods)):
        ax.text(i, rhos[i] + 0.005, f"{rhos[i]:+.3f}", ha="center", fontsize=9)
    ax.set_ylabel("Bucket-weighted within-bucket Spearman ρ")
    ax.set_title("Within-bucket ρ across 8 methods (Claude meta-policy, n=1000)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(-0.15, 0.45)
    fig.tight_layout()
    fig.savefig(FIG / "fig_v3_5_within_bucket_rho.png")
    plt.close(fig)


def main():
    fig1_three_arm_m1()
    print("wrote fig_v3_1_three_arm_m1.png")
    fig2_method_catalog()
    print("wrote fig_v3_2_method_catalog_gaps.png")
    fig3_per_bucket_heatmap()
    print("wrote fig_v3_3_per_bucket_heatmap.png")
    fig4_commitment_shrinkage()
    print("wrote fig_v3_4_commitment_shrinkage.png")
    fig5_within_bucket_rho()
    print("wrote fig_v3_5_within_bucket_rho.png")
    print(f"\nFigures in {FIG}")


if __name__ == "__main__":
    main()
