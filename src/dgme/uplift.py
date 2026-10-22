"""Study 2 — the propensity-vs-uplift trap on a synthetic randomised trial.

Scenario: a campaign has a fixed budget to contact the top 10% of customers.
The obvious ranking is "propensity to convert". But some high-propensity
customers are *sleeping dogs*: they convert anyway, and contacting them lowers
conversion. Because the data is a true RCT and the effect is generated, we know
each customer's real treatment effect and can show:

  * propensity-ranked targeting selects the sleeping-dog segment (negative
    true effect), so the "winning" conversion@10% is an illusion; while
  * a two-model (T-learner) uplift estimator recovers the average effect and
    ranks the persuadables first.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import config as C
from .metrics import mean_effect_in_top_k, precision_at_k, qini_points, top_k_index


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def generate(seed: int = C.SEED):
    """Return a dict describing the synthetic RCT, including ground-truth tau."""
    rng = np.random.default_rng(seed)
    beta = np.array(C.UP_BETA)
    n, k = C.N_UPLIFT, len(beta)

    X = rng.standard_normal((n, k))
    # The segment is identifiable from the covariates (high X0 = already-loyal,
    # high-value), which is exactly why a propensity ranker selects it.
    sleeping_dog = (X[:, 0] > C.HB_X0_THRESHOLD).astype(int)

    base_logit = C.UP_INTERCEPT + X @ beta + C.HB_BASELINE_BOOST * sleeping_dog
    p0 = _sigmoid(base_logit)                       # control potential outcome

    tau = np.where(
        sleeping_dog == 1,
        C.TAU_SLEEPING_DOG,
        C.TAU_PERSUADABLE,
    ) + C.TAU_X0_COUPLING * X[:, 0] * (1 - sleeping_dog)
    p1 = np.clip(p0 + tau, 0.001, 0.999)            # treated potential outcome
    tau_true = p1 - p0

    t = (rng.random(n) < C.TREATMENT_SHARE).astype(int)
    p_obs = np.where(t == 1, p1, p0)
    y = (rng.random(n) < p_obs).astype(int)

    return {
        "X": X, "t": t, "y": y, "p0": p0, "p1": p1,
        "tau_true": tau_true, "sleeping_dog": sleeping_dog,
    }


def propensity_scores(data: dict) -> np.ndarray:
    """P(convert) ignoring treatment — the naive 'who will convert?' ranking."""
    model = LogisticRegression(max_iter=1000)
    model.fit(data["X"], data["y"])
    return model.predict_proba(data["X"])[:, 1]


def uplift_scores(data: dict) -> np.ndarray:
    """Two-model (T-learner) uplift: P(y|treated) - P(y|control)."""
    X, t, y = data["X"], data["t"], data["y"]
    m_t = LogisticRegression(max_iter=1000).fit(X[t == 1], y[t == 1])
    m_c = LogisticRegression(max_iter=1000).fit(X[t == 0], y[t == 0])
    return m_t.predict_proba(X)[:, 1] - m_c.predict_proba(X)[:, 1]


def difference_in_means_ate(data: dict) -> float:
    """The experiment's headline: mean(y|treated) - mean(y|control)."""
    y, t = data["y"], data["t"]
    return float(np.mean(y[t == 1]) - np.mean(y[t == 0]))


@dataclass
class UpliftResult:
    n: int
    true_ate: float
    estimated_ate_tlearner: float
    diff_in_means_ate: float
    conversion_at_10_by_propensity: float
    true_effect_in_top10_propensity: float
    true_effect_in_top10_uplift: float
    incremental_top10_propensity: float
    incremental_top10_uplift: float
    sleeping_dog_share_top10_propensity: float

    def as_dict(self) -> dict:
        return asdict(self)


def run(seed: int = C.SEED) -> UpliftResult:
    data = generate(seed)
    tau = data["tau_true"]
    y = data["y"]

    prop = propensity_scores(data)
    upl = uplift_scores(data)

    # Incremental conversions from treating the top decile under each ranking
    # (sum of true effect over the targeted set).
    prop_top = top_k_index(prop, C.TOP_FRAC)
    upl_top = top_k_index(upl, C.TOP_FRAC)

    return UpliftResult(
        n=int(C.N_UPLIFT),
        true_ate=round(float(np.mean(tau)), 4),
        estimated_ate_tlearner=round(float(np.mean(upl)), 4),
        diff_in_means_ate=round(difference_in_means_ate(data), 4),
        conversion_at_10_by_propensity=round(precision_at_k(y, prop, C.TOP_FRAC), 4),
        true_effect_in_top10_propensity=round(mean_effect_in_top_k(tau, prop, C.TOP_FRAC), 4),
        true_effect_in_top10_uplift=round(mean_effect_in_top_k(tau, upl, C.TOP_FRAC), 4),
        incremental_top10_propensity=round(float(np.sum(tau[prop_top])), 2),
        incremental_top10_uplift=round(float(np.sum(tau[upl_top])), 2),
        sleeping_dog_share_top10_propensity=round(
            float(np.mean(data["sleeping_dog"][prop_top])), 4
        ),
    )


def targeting_curves(seed: int = C.SEED):
    """Qini-style cumulative-incremental curves for the report figure."""
    data = generate(seed)
    tau = data["tau_true"]
    prop = propensity_scores(data)
    upl = uplift_scores(data)
    random_scores = np.linspace(1.0, 0.0, len(tau))  # a fixed 'random' ranking
    return {
        "propensity": qini_points(tau, prop),
        "uplift": qini_points(tau, upl),
        "random": qini_points(tau, random_scores),
    }
