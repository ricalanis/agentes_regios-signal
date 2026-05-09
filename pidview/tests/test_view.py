"""Tests for SignalView and Snapshot semantics."""
from __future__ import annotations

import math

import numpy as np
import pytest

from pidview import SignalView, Snapshot


# ---- Snapshot dataclass shape ----------------------------------------------


def test_snapshot_is_frozen():
    snap = Snapshot(
        name="x",
        t=0.0,
        present=0.0,
        differential=0.0,
        integral=0.0,
        history=np.zeros((0, 2)),
        stats={},
    )
    with pytest.raises(Exception):
        snap.t = 1.0  # frozen dataclass should reject assignment


# ---- Empty view ------------------------------------------------------------


def test_empty_view_returns_zeros():
    v = SignalView("focus")
    s = v.snapshot()
    assert s.name == "focus"
    assert s.present == 0.0
    assert s.differential == 0.0
    assert s.integral == 0.0
    assert s.history.shape == (0, 2)
    for k in ("mean", "std", "p10", "p50", "p90", "slope"):
        assert s.stats[k] == 0.0


# ---- Single sample ---------------------------------------------------------


def test_single_sample_present_equals_x_others_zero():
    v = SignalView("x")
    v.push(10.0, 3.5)
    s = v.snapshot()
    assert s.present == 3.5
    assert s.t == 10.0
    assert s.differential == 0.0  # need 2+ points for slope
    assert s.integral == 0.0  # first sample initializes I=0
    assert s.history.shape == (1, 2)
    assert s.history[0, 0] == 10.0
    assert s.history[0, 1] == 3.5


# ---- Differential: ramp ----------------------------------------------------


def test_differential_constant_ramp():
    """x(t) = 2t over a 1-second window: slope must be 2.0."""
    v = SignalView("x", differential_window_seconds=2.0)
    for i in range(11):  # t = 0.0, 0.1, ..., 1.0
        t = i * 0.1
        v.push(t, 2.0 * t)
    s = v.snapshot()
    assert math.isclose(s.differential, 2.0, abs_tol=1e-9)


def test_differential_only_uses_window():
    """Old samples outside window must not influence slope."""
    v = SignalView("x", differential_window_seconds=2.0)
    # First a flat region far in the past.
    for i in range(5):
        v.push(float(i), 100.0)
    # Then a steep ramp inside the last 2 seconds.
    base = 100.0  # last t
    for i in range(1, 11):
        v.push(base + i * 0.1, i * 1.0)  # x grows by 1 per 0.1s -> slope 10
    s = v.snapshot()
    assert math.isclose(s.differential, 10.0, abs_tol=1e-6)


def test_differential_zero_with_one_in_window():
    v = SignalView("x", differential_window_seconds=0.5)
    v.push(0.0, 1.0)
    v.push(10.0, 2.0)  # only this sample inside window relative to t_latest=10
    s = v.snapshot()
    assert s.differential == 0.0


# ---- Integral: trapezoidal (tau=None) --------------------------------------


def test_integral_trapezoidal_constant():
    v = SignalView("x", integral_tau=None)
    v.push(0.0, 5.0)  # first push: integral stays at 0
    v.push(2.0, 5.0)  # trapezoid: (5+5)/2 * 2 = 10
    assert math.isclose(v.snapshot().integral, 10.0, abs_tol=1e-9)
    v.push(3.0, 7.0)  # add (5+7)/2 * 1 = 6 -> total 16
    assert math.isclose(v.snapshot().integral, 16.0, abs_tol=1e-9)


# ---- Integral: leaky -------------------------------------------------------


def test_integral_leaky_two_sample_analytic():
    """One step from I=0 with input x, dt -> I = dt * (x - 0/tau) = dt * x."""
    v = SignalView("x", integral_tau=10.0)
    v.push(0.0, 1.0)  # I=0
    v.push(0.5, 4.0)  # forward Euler: I += 0.5*(4 - 0/10) = 2.0
    assert math.isclose(v.snapshot().integral, 2.0, abs_tol=1e-12)


def test_integral_leaky_decays_to_steady_state():
    """Constant input x drives I -> x*tau as t -> infinity."""
    tau = 5.0
    x = 1.0
    v = SignalView("x", integral_tau=tau)
    t = 0.0
    v.push(t, x)
    dt = 0.01
    for _ in range(20000):  # 200 seconds, well > 5*tau
        t += dt
        v.push(t, x)
    # Steady state of dI/dt = x - I/tau is I = x*tau = 5.0
    assert math.isclose(v.snapshot().integral, x * tau, rel_tol=1e-3)


# ---- History eviction ------------------------------------------------------


def test_history_eviction_after_long_gap():
    v = SignalView("x", history_seconds=10.0)
    for i in range(100):
        v.push(float(i) * 0.1, float(i))  # 0..9.9s, well within 10s
    # Now push one sample 700s later; everything older than t_latest - 10 must drop.
    v.push(700.0, 999.0)
    s = v.snapshot()
    assert s.history.shape == (1, 2)
    assert s.history[0, 0] == 700.0
    assert s.history[0, 1] == 999.0


def test_history_returns_copy_not_view():
    v = SignalView("x")
    v.push(0.0, 1.0)
    s = v.snapshot()
    s.history[0, 1] = 999.0
    s2 = v.snapshot()
    assert s2.history[0, 1] == 1.0


# ---- Stats -----------------------------------------------------------------


def test_stats_over_history():
    v = SignalView("x", history_seconds=1000.0)
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    for i, x in enumerate(xs):
        v.push(float(i), x)
    s = v.snapshot()
    assert math.isclose(s.stats["mean"], 3.0)
    assert math.isclose(s.stats["p50"], 3.0)
    # slope over (0,1,2,3,4) vs (1..5) is 1.0
    assert math.isclose(s.stats["slope"], 1.0, abs_tol=1e-9)


# ---- Pub/sub ---------------------------------------------------------------


def test_subscriber_called_once_per_push_in_order():
    v = SignalView("x")
    seen: list[tuple[str, float]] = []
    v.subscribe(lambda s: seen.append(("a", s.present)))
    v.subscribe(lambda s: seen.append(("b", s.present)))
    v.push(0.0, 7.0)
    v.push(1.0, 8.0)
    assert seen == [("a", 7.0), ("b", 7.0), ("a", 8.0), ("b", 8.0)]


def test_unsubscribe_stops_callbacks():
    v = SignalView("x")
    seen: list[float] = []
    unsub = v.subscribe(lambda s: seen.append(s.present))
    v.push(0.0, 1.0)
    unsub()
    v.push(1.0, 2.0)
    assert seen == [1.0]


def test_subscriber_exception_does_not_crash_push(capsys):
    v = SignalView("x")
    seen: list[float] = []

    def boom(_):
        raise RuntimeError("boom")

    v.subscribe(boom)
    v.subscribe(lambda s: seen.append(s.present))
    v.push(0.0, 42.0)  # must not raise
    assert seen == [42.0]


# ---- Out-of-order timestamps ----------------------------------------------


def test_out_of_order_timestamp_raises():
    v = SignalView("x")
    v.push(5.0, 1.0)
    with pytest.raises(ValueError):
        v.push(4.0, 2.0)
