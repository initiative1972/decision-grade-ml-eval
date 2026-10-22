import warnings

from dgme import leakage

warnings.filterwarnings("ignore")


def test_generate_shapes():
    X, duration, y = leakage.generate()
    assert X.shape[0] == duration.shape[0] == y.shape[0] == 40000
    assert set(y.tolist()) <= {0, 1}


def test_leak_inflates_auc():
    r = leakage.run()
    # Honest model ~0.81, leaky ~0.94 — tolerant bands for cross-env drift.
    assert 0.78 <= r.auc_honest <= 0.83
    assert 0.92 <= r.auc_leaky <= 0.96
    assert r.auc_leaky > r.auc_honest
    assert r.auc_inflation > 0.10


def test_base_rate_reasonable():
    r = leakage.run()
    assert 0.12 <= r.base_rate <= 0.18


def test_feature_contract_flags_duration():
    r = leakage.run()
    assert r.leaking_features == ["call_duration"]
    assert leakage.audit_features(leakage.FEATURE_CONTRACT) == ["call_duration"]


def test_determinism():
    a = leakage.run()
    b = leakage.run()
    assert a.as_dict() == b.as_dict()
