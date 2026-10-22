import warnings

import numpy as np

from dgme import uplift

warnings.filterwarnings("ignore")


def test_generate_is_a_randomised_trial():
    data = uplift.generate()
    # Treatment assignment independent of covariates (RCT): ~50/50.
    assert 0.47 <= data["t"].mean() <= 0.53
    assert data["tau_true"].shape[0] == 20000


def test_propensity_targeting_hits_sleeping_dogs():
    r = uplift.run()
    # The trap: propensity top decile has a NEGATIVE true effect...
    assert r.true_effect_in_top10_propensity < 0
    # ...and is dominated by the sleeping-dog segment.
    assert r.sleeping_dog_share_top10_propensity > 0.7
    # Its incremental conversions are negative: contacting them loses sales.
    assert r.incremental_top10_propensity < 0


def test_uplift_targeting_is_positive():
    r = uplift.run()
    assert r.true_effect_in_top10_uplift > 0
    assert r.incremental_top10_uplift > 0
    assert r.true_effect_in_top10_uplift > r.true_effect_in_top10_propensity


def test_tlearner_recovers_the_ate():
    r = uplift.run()
    assert abs(r.estimated_ate_tlearner - r.true_ate) < 0.03


def test_determinism():
    assert uplift.run().as_dict() == uplift.run().as_dict()


def test_true_tau_matches_segments():
    data = uplift.generate()
    sd = data["sleeping_dog"] == 1
    assert np.mean(data["tau_true"][sd]) < 0      # sleeping dogs: negative
    assert np.mean(data["tau_true"][~sd]) > 0     # persuadables: positive
