"""Study 1 — target leakage via a post-outcome feature, and the audit that
recovers honest performance.

Scenario: a marketing team wants to rank customers by their propensity to
subscribe. A `call_duration` feature is enormously predictive — because a long
call only *happens* when someone is already subscribing. It exists only after
the outcome, so it must fail a feature-contract audit. We show the inflated vs
honest performance, and that a one-line contract check catches it.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from . import config as C
from .metrics import auc, precision_at_k, roc_points


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def generate(seed: int = C.SEED):
    """Return (X_honest, call_duration, y) for the synthetic campaign."""
    rng = np.random.default_rng(seed)
    beta = np.array(C.LEAK_BETA)
    n, k = C.N_LEAK, len(beta)

    X = rng.standard_normal((n, k))
    logit = C.LEAK_INTERCEPT + C.LEAK_SIGNAL_SCALE * (X @ beta)
    p = _sigmoid(logit)
    y = (rng.random(n) < p).astype(int)

    # The leak: duration is generated FROM the label (plus mild coupling and
    # noise). In production it is unknown at decision time.
    duration = (
        C.DUR_BASE
        + C.DUR_PER_LABEL * y
        + C.DUR_X0_COUPLING * X[:, 0]
        + C.DUR_NOISE * rng.standard_normal(n)
    )
    return X, duration, y


def _fit_eval(X_train, y_train, X_test, y_test):
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(scaler.transform(X_train), y_train)
    scores = model.predict_proba(scaler.transform(X_test))[:, 1]
    return scores


# A feature contract: each field tagged with whether it is known at decision
# time. The audit simply refuses any feature that is not.
FEATURE_CONTRACT = {
    "customer_features": {"available_at_decision_time": True},
    "call_duration": {"available_at_decision_time": False,
                      "reason": "only known after the call that realises the outcome"},
}


def audit_features(contract: Dict[str, dict]) -> list:
    """Return the features that FAIL the contract (not known at decision time)."""
    return [name for name, meta in contract.items()
            if not meta.get("available_at_decision_time", False)]


@dataclass
class LeakageResult:
    n: int
    base_rate: float
    auc_honest: float
    auc_leaky: float
    precision_at_10_honest: float
    precision_at_10_leaky: float
    auc_inflation: float
    leaking_features: list

    def as_dict(self) -> dict:
        return asdict(self)


def run(seed: int = C.SEED) -> LeakageResult:
    X, duration, y = generate(seed)
    X_tr, X_te, d_tr, d_te, y_tr, y_te = train_test_split(
        X, duration, y, test_size=C.LEAK_TEST_FRAC, random_state=seed, stratify=y
    )

    # Honest model: customer features only (all pass the contract).
    honest_scores = _fit_eval(X_tr, y_tr, X_te, y_te)

    # Leaky model: the same, plus the post-outcome duration feature.
    Xd_tr = np.column_stack([X_tr, d_tr])
    Xd_te = np.column_stack([X_te, d_te])
    leaky_scores = _fit_eval(Xd_tr, y_tr, Xd_te, y_te)

    auc_h = auc(y_te, honest_scores)
    auc_l = auc(y_te, leaky_scores)
    return LeakageResult(
        n=int(C.N_LEAK),
        base_rate=round(float(np.mean(y)), 4),
        auc_honest=round(auc_h, 4),
        auc_leaky=round(auc_l, 4),
        precision_at_10_honest=round(precision_at_k(y_te, honest_scores, 0.10), 4),
        precision_at_10_leaky=round(precision_at_k(y_te, leaky_scores, 0.10), 4),
        auc_inflation=round(auc_l - auc_h, 4),
        leaking_features=audit_features(FEATURE_CONTRACT),
    )


def roc_curves(seed: int = C.SEED):
    """ROC points for both models, for the report figure."""
    X, duration, y = generate(seed)
    X_tr, X_te, d_tr, d_te, y_tr, y_te = train_test_split(
        X, duration, y, test_size=C.LEAK_TEST_FRAC, random_state=seed, stratify=y
    )
    honest_scores = _fit_eval(X_tr, y_tr, X_te, y_te)
    leaky_scores = _fit_eval(
        np.column_stack([X_tr, d_tr]), y_tr, np.column_stack([X_te, d_te]), y_te
    )
    return {
        "honest": roc_points(y_te, honest_scores),
        "leaky": roc_points(y_te, leaky_scores),
    }
