"""Tests for SignalRegistry."""
from __future__ import annotations

import pytest

from pidview import SignalRegistry, SignalView


def test_register_and_get():
    reg = SignalRegistry()
    v = reg.register("focus", history_seconds=120.0)
    assert isinstance(v, SignalView)
    assert reg.get("focus") is v
    assert "focus" in reg


def test_register_duplicate_raises():
    reg = SignalRegistry()
    reg.register("focus")
    with pytest.raises(ValueError):
        reg.register("focus")


def test_get_unknown_raises_keyerror():
    reg = SignalRegistry()
    with pytest.raises(KeyError):
        reg.get("nope")


def test_push_forwards_to_view():
    reg = SignalRegistry()
    reg.register("x")
    reg.push("x", 0.0, 3.0)
    snap = reg.get("x").snapshot()
    assert snap.present == 3.0


def test_snapshot_all_returns_dict():
    reg = SignalRegistry()
    reg.register("a")
    reg.register("b")
    reg.push("a", 0.0, 1.0)
    reg.push("b", 0.0, 2.0)
    snaps = reg.snapshot_all()
    assert set(snaps) == {"a", "b"}
    assert snaps["a"].present == 1.0
    assert snaps["b"].present == 2.0


def test_view_kwargs_passthrough():
    reg = SignalRegistry()
    v = reg.register("x", history_seconds=42.0, integral_tau=None,
                    differential_window_seconds=1.5)
    assert v.history_seconds == 42.0
    assert v.integral_tau is None
    assert v.differential_window_seconds == 1.5
