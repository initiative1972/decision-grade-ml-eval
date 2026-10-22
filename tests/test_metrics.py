import numpy as np

from dgme.metrics import (
    cumulative_incremental,
    mean_effect_in_top_k,
    precision_at_k,
    top_k_index,
)


def test_top_k_index_picks_highest():
    scores = np.array([0.1, 0.9, 0.5, 0.8])
    idx = top_k_index(scores, 0.5)          # top 2 of 4
    assert set(idx.tolist()) == {1, 3}


def test_precision_at_k_counts_positives_in_top():
    y = np.array([0, 1, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.8, 0.2, 0.7])
    # top 40% = top 2 -> indices 1,2 -> both positive
    assert precision_at_k(y, scores, 0.4) == 1.0


def test_mean_effect_in_top_k():
    tau = np.array([-0.2, 0.5, 0.4, -0.1])
    scores = np.array([0.9, 0.1, 0.2, 0.8])   # top 2 -> idx 0,3 -> tau -0.2,-0.1
    assert abs(mean_effect_in_top_k(tau, scores, 0.5) - (-0.15)) < 1e-9


def test_cumulative_incremental_runs_down_ranking():
    tau = np.array([0.3, -0.1, 0.2])
    scores = np.array([0.9, 0.5, 0.1])         # order 0,1,2
    cum = cumulative_incremental(tau, scores)
    assert np.allclose(cum, [0.3, 0.2, 0.4])
