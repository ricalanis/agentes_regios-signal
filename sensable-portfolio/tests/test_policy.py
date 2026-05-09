import numpy as np
import pytest

from sensable_portfolio.policy.linucb import LinUCBPolicy


def _ctx(*xs):
    return np.asarray(xs, dtype=np.float64)


def test_linucb_predict_returns_known_arm():
    p = LinUCBPolicy(arms=["a", "b"], context_dim=2, alpha=1.0)
    arm = p.predict(_ctx(0.0, 1.0))
    assert arm in {"a", "b"}


def test_linucb_learns_arm_for_regime():
    """In contexts where x[0]>0 'a' is better; x[0]<=0 'b' is better."""
    rng = np.random.default_rng(0)
    p = LinUCBPolicy(arms=["a", "b"], context_dim=2, alpha=0.5)
    for _ in range(400):
        x0 = float(rng.normal())
        ctx = _ctx(x0, 1.0)
        arm = p.predict(ctx)
        if arm == "a":
            r = 0.5 * x0 + rng.normal(0, 0.1)
        else:
            r = -0.5 * x0 + rng.normal(0, 0.1)
        p.partial_fit(ctx, arm, float(np.clip(r, -1, 1)))

    a_wins = sum(p.predict(_ctx(1.0, 1.0)) == "a" for _ in range(50))
    b_wins = sum(p.predict(_ctx(-1.0, 1.0)) == "b" for _ in range(50))
    assert a_wins > 35
    assert b_wins > 35


def test_linucb_handles_arm_growth():
    p = LinUCBPolicy(arms=["a"], context_dim=2, alpha=1.0)
    p.partial_fit(_ctx(1.0, 1.0), "a", 0.5)
    p.add_arm("b")
    arm = p.predict(_ctx(0.0, 0.0))
    assert arm in {"a", "b"}


def test_linucb_snapshot_round_trip():
    p = LinUCBPolicy(arms=["a", "b"], context_dim=2, alpha=1.0)
    for _ in range(20):
        p.partial_fit(_ctx(0.5, 0.5), "a", 0.3)
    blob = p.snapshot()
    p2 = LinUCBPolicy(arms=["a", "b"], context_dim=2, alpha=1.0)
    p2.restore(blob)
    arm1 = p.predict(_ctx(0.5, 0.5))
    arm2 = p2.predict(_ctx(0.5, 0.5))
    assert arm1 == arm2
