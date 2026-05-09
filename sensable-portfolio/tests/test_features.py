import numpy as np
from pidview import SignalRegistry, Snapshot

from sensable_portfolio.signals.features import build_feature_vector, FEATURE_DIM
from sensable_portfolio.signals.view import build_registry, CONTEXT_KINDS


def test_feature_vector_dimension_is_36():
    reg = build_registry()
    for i in range(10):
        for k in CONTEXT_KINDS + ("valence", "arousal"):
            reg.push(k, float(i), float(i) * 0.1)
    fv = build_feature_vector(reg.snapshot_all())
    assert fv.shape == (FEATURE_DIM,)
    assert FEATURE_DIM == 36


def test_feature_vector_handles_short_history():
    reg = build_registry()
    for k in CONTEXT_KINDS + ("valence", "arousal"):
        reg.push(k, 0.0, 0.5)
    fv = build_feature_vector(reg.snapshot_all())
    assert fv.shape == (FEATURE_DIM,)
    assert not np.isnan(fv).any()
