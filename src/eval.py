"""Evaluation metrics with bootstrap CIs."""

from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    average_precision_score, roc_auc_score, brier_score_loss
)
from scipy.stats import wasserstein_distance, levene, spearmanr


def pr_auc(y_true, y_score) -> float:
    return float(average_precision_score(y_true, y_score))


def roc_auc(y_true, y_score) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def brier(y_true, y_score) -> float:
    return float(brier_score_loss(y_true, y_score))


def ece(y_true, y_score, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_score, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    err = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        conf = y_score[mask].mean()
        acc = y_true[mask].mean()
        err += (mask.sum() / n) * abs(conf - acc)
    return float(err)


def calibration_curve(y_true, y_score, n_bins: int = 10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_score, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        rows.append({"bin": b, "pred_mean": float(y_score[mask].mean()), "actual_rate": float(y_true[mask].mean()), "n": int(mask.sum())})
    return rows


def decile_rates(y_true, y_score):
    """Return decile predicted-mean and actual-rate arrays."""
    order = np.argsort(y_score)
    n = len(y_score)
    deciles_pred = []
    deciles_act = []
    for d in range(10):
        lo = int(d * n / 10)
        hi = int((d + 1) * n / 10)
        if hi <= lo:
            continue
        idx = order[lo:hi]
        deciles_pred.append(float(y_score[idx].mean()))
        deciles_act.append(float(y_true[idx].mean()))
    return np.array(deciles_pred), np.array(deciles_act)


def wasserstein_decile(y_true, y_score) -> float:
    pred, act = decile_rates(y_true, y_score)
    return float(wasserstein_distance(pred, act))


def under_dispersion(y_true, y_score):
    """Ratio Var(pred) / Var(observed) and Levene test."""
    var_pred = float(np.var(y_score))
    var_obs = float(np.var(y_true.astype(float)))
    stat, p = levene(y_score, y_true.astype(float))
    return {"var_pred": var_pred, "var_obs": var_obs, "ratio": var_pred / var_obs if var_obs > 0 else float("nan"), "levene_stat": float(stat), "levene_p": float(p)}


def park_normalized_accuracy(label_curr, label_prev, y_score, threshold: float = None):
    """Park 2024 normalized accuracy adapted to retail.

    label_curr: customer's label in current 30-day window
    label_prev: customer's label in the previous 30-day window (test-retest analog)
    y_score: model's predicted probability for current window

    normalized_acc = (agent-vs-human agreement) / (human-vs-self agreement)
    """
    if threshold is None:
        # use the rate as a reasonable threshold-free option
        threshold = float(np.mean(label_curr))
    y_pred_bin = (y_score >= threshold).astype(int)
    agent_vs_human = float((y_pred_bin == label_curr).mean())
    human_vs_self = float((label_prev == label_curr).mean())
    if human_vs_self == 0:
        return float("nan")
    return agent_vs_human / human_vs_self


def bootstrap_ci(values, metric_fn, B: int = 1000, seed: int = 42, alpha: float = 0.05, paired_with: dict | None = None):
    """Generic bootstrap CI.

    values: dict with arrays needed by metric_fn (e.g. {"y_true": ..., "y_score": ...})
    metric_fn: callable taking **values -> scalar
    """
    rng = np.random.default_rng(seed)
    n = len(next(iter(values.values())))
    boots = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        sub = {k: v[idx] for k, v in values.items()}
        boots.append(metric_fn(**sub))
    boots = np.array(boots)
    point = metric_fn(**values)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return {"point": float(point), "lo": lo, "hi": hi, "se": float(boots.std())}


def paired_bootstrap_diff(y_true, score_a, score_b, metric_fn, B: int = 1000, seed: int = 42, alpha: float = 0.05):
    """Paired bootstrap CI on (metric(A) - metric(B))."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        ma = metric_fn(y_true[idx], score_a[idx])
        mb = metric_fn(y_true[idx], score_b[idx])
        diffs.append(ma - mb)
    diffs = np.array(diffs)
    point = metric_fn(y_true, score_a) - metric_fn(y_true, score_b)
    lo = float(np.quantile(diffs, alpha / 2))
    hi = float(np.quantile(diffs, 1 - alpha / 2))
    p_two = float(2 * min((diffs >= 0).mean(), (diffs <= 0).mean()))
    return {"point": float(point), "lo": lo, "hi": hi, "p": p_two, "se": float(diffs.std())}


def all_metrics(y_true: np.ndarray, y_score: np.ndarray, B: int = 1000, seed: int = 42) -> dict:
    """Compute all reported metrics with bootstrap CIs."""
    out = {}
    out["pr_auc"] = bootstrap_ci({"y_true": y_true, "y_score": y_score}, lambda y_true, y_score: pr_auc(y_true, y_score), B=B, seed=seed)
    out["roc_auc"] = bootstrap_ci({"y_true": y_true, "y_score": y_score}, lambda y_true, y_score: roc_auc(y_true, y_score) if len(np.unique(y_true)) >= 2 else float("nan"), B=B, seed=seed)
    out["brier"] = bootstrap_ci({"y_true": y_true, "y_score": y_score}, lambda y_true, y_score: brier(y_true, y_score), B=B, seed=seed)
    out["ece"] = ece(y_true, y_score)
    out["wasserstein_decile"] = wasserstein_decile(y_true, y_score)
    out["calibration_curve"] = calibration_curve(y_true, y_score)
    out["under_dispersion"] = under_dispersion(y_true, y_score)
    out["label_rate"] = float(y_true.mean())
    out["n"] = int(len(y_true))
    return out


def holm_bonferroni(p_values: list[float], alpha: float = 0.05):
    """Return list of (idx, p, p_adj, reject) sorted by p, with Holm-Bonferroni adjustment."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    out = []
    max_p = 0.0
    for k, (i, p) in enumerate(indexed):
        p_adj = min(1.0, p * (m - k))
        p_adj = max(p_adj, max_p)
        max_p = p_adj
        out.append({"idx": i, "p": p, "p_adj": p_adj, "reject": p_adj < alpha})
    return out
