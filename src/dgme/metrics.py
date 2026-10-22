"""Evaluation metrics, chosen to match the decision each study serves.

Pure functions over numpy arrays. ROC-AUC comes from scikit-learn; the
decision-grade metrics (precision@k, incremental conversions under a ranking,
Qini) are implemented here so their definitions are explicit and testable.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(y_true, scores))


def top_k_index(scores: np.ndarray, k_frac: float) -> np.ndarray:
    """Indices of the top ``k_frac`` fraction of items by score (desc)."""
    n = len(scores)
    k = max(1, int(round(k_frac * n)))
    # Stable: break ties by original order so results are deterministic.
    order = np.argsort(-scores, kind="stable")
    return order[:k]


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k_frac: float) -> float:
    """Fraction of positives among the top ``k_frac`` scored items."""
    idx = top_k_index(scores, k_frac)
    return float(np.mean(y_true[idx]))


def mean_effect_in_top_k(tau: np.ndarray, scores: np.ndarray, k_frac: float) -> float:
    """Mean true treatment effect among the top ``k_frac`` by a ranking.

    This is the number a propensity ranker never sees: it tells you whether the
    people you targeted were actually *moved* by the treatment.
    """
    idx = top_k_index(scores, k_frac)
    return float(np.mean(tau[idx]))


def cumulative_incremental(tau: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Cumulative true incremental conversions as you target by ``scores``.

    Walk down the ranking; at each depth, the value is the summed true effect
    of everyone targeted so far. A ranking that puts sleeping dogs first goes
    *negative* early.
    """
    order = np.argsort(-scores, kind="stable")
    return np.cumsum(tau[order])


def qini_points(tau: np.ndarray, scores: np.ndarray, n_points: int = 101):
    """(fraction_targeted, cumulative_incremental) sampled for a curve."""
    curve = cumulative_incremental(tau, scores)
    n = len(curve)
    fracs = np.linspace(0.0, 1.0, n_points)
    idx = np.clip((fracs * n).astype(int) - 1, 0, n - 1)
    values = curve[idx]
    values[0] = 0.0
    return fracs, values


def roc_points(y_true: np.ndarray, scores: np.ndarray):
    """(fpr, tpr) arrays for an ROC curve, via scikit-learn."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, scores)
    return fpr, tpr
