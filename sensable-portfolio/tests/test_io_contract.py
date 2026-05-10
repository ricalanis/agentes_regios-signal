"""Pin the IO contracts between neurable_connector, pidview, and
sensable-portfolio. These tests fail in CI when an upstream package
changes its public shape, before runtime sees the drift."""
from __future__ import annotations

import pathlib
from dataclasses import fields

import pytest


def test_affect_sample_has_all_eight_dims_and_t_and_features():
    from neurable_connector import AffectSample
    names = {f.name for f in fields(AffectSample)}
    expected = {
        "t", "features",
        "focus", "stress", "valence", "arousal",
        "joy", "calm", "excitement", "neutral",
    }
    assert expected <= names, f"AffectSample missing {expected - names}"


def test_signals_view_imports_without_drift():
    """signals/view.py asserts at module-load; this test fails fast if it raises."""
    import importlib
    import sensable_portfolio.signals.view as v
    importlib.reload(v)
    assert set(v.ALL_DIMS) == {
        "focus", "stress", "valence", "arousal",
        "joy", "calm", "excitement", "neutral",
    }
    # Order is wire-stable; do not change.
    assert v.ALL_DIMS == (
        "focus", "stress", "valence", "arousal",
        "joy", "calm", "excitement", "neutral",
    )


def test_pidview_snapshot_history_is_n_by_2():
    """features.py assumes history rows are [t, x]; pin that here too."""
    from pidview import SignalView
    v = SignalView("test", history_seconds=10.0, integral_tau=1.0)
    v.push(0.0, 1.0)
    v.push(1.0, 2.0)
    h = v.snapshot().history
    assert h.ndim == 2 and h.shape[1] == 2


def test_unix_seconds_to_ms_matches_prior_inline_behavior():
    """Truncation, not rounding — preserves byte-identical wire output."""
    from sensable_portfolio.contracts import unix_seconds_to_ms
    assert unix_seconds_to_ms(0.0) == 0
    assert unix_seconds_to_ms(1.5) == 1500
    assert unix_seconds_to_ms(1778364427.578) == 1778364427578
    # Truncation property: a value that rounds up under round() should
    # truncate down under our helper.
    assert unix_seconds_to_ms(0.0009999) == 0


def test_app_uses_helper_for_ts_conversion_no_inline_regression():
    """Catch any regression that re-introduces inline `* 1000` in app.py."""
    here = pathlib.Path(__file__).resolve().parent
    app_path = here.parent / "src" / "sensable_portfolio" / "app.py"
    src = app_path.read_text()
    assert "unix_seconds_to_ms" in src, "app.py must import + use unix_seconds_to_ms"
    assert "* 1000" not in src, (
        "app.py must not perform raw seconds→ms conversion; "
        "use unix_seconds_to_ms instead"
    )


def test_mood_frame_vector_can_carry_all_eight_dims():
    from sensable_portfolio.contracts import MoodFrame
    from sensable_portfolio.signals.view import ALL_DIMS
    f = MoodFrame(vector={k: 0.0 for k in ALL_DIMS}, ts=0)
    assert set(f.vector.keys()) == set(ALL_DIMS)


def test_agent_action_frame_signals_at_decision_keys_use_all_dims():
    from sensable_portfolio.contracts import AgentActionFrame, AgentInfo, Intervention
    from sensable_portfolio.signals.view import ALL_DIMS
    f = AgentActionFrame(
        ts=0, decision_id="d",
        agent=AgentInfo(id="x", persona="p", model="m"),
        intervention=Intervention(decision_id="d", arm_id="x", ts=0.0,
            action_type="breath", title="t", body="b",
            duration_s=10.0, intensity="low", rationale="r"),
        signals_at_decision={k: 0.0 for k in ALL_DIMS},
    )
    assert set(f.signals_at_decision.keys()) == set(ALL_DIMS)
