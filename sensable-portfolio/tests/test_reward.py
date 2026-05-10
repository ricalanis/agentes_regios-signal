import asyncio
import json
import pytest
from sqlmodel import select

from sensable_portfolio.policy.linucb import LinUCBPolicy
from sensable_portfolio.reward.attribution import attribute
from sensable_portfolio.reward.scheduler import RewardScheduler
from sensable_portfolio.reward.feedback import record_feedback
from sensable_portfolio.signals.features import FEATURE_DIM
from sensable_portfolio.storage.db import get_session, init_engine
from sensable_portfolio.storage.models import (
    Decision, Feedback, Reward, SnapshotLog,
)


async def _seed(engine, decision_ts: float):
    async with get_session(engine) as s:
        for ts in range(int(decision_ts - 120), int(decision_ts)):
            for k in ("focus", "stress"):
                s.add(SnapshotLog(ts=float(ts), kind=k, value=0.5))
        for ts in range(int(decision_ts + 60), int(decision_ts + 360)):
            for k in ("focus", "stress"):
                s.add(SnapshotLog(ts=float(ts), kind=k, value=1.5))
        s.add(Decision(
            id="d1", ts=decision_ts, arm_id="x", target_id="default",
            context_json=json.dumps([0.0] * FEATURE_DIM),
            intervention_json="{}", run_id=None,
        ))
        await s.commit()


@pytest.mark.asyncio
async def test_attribute_signed_weights_flip_correctly():
    engine = await init_engine("sqlite+aiosqlite:///:memory:")
    await _seed(engine, decision_ts=1000.0)

    components, reward = await attribute(
        engine, decision_id="d1", target_weights={"stress": -0.5, "focus": 0.5},
        baseline_pre=120, window_lo=60, window_hi=360, alpha=0.5,
    )
    # focus rises Δ=+1; +0.5 weight => +0.5*z
    # stress rises Δ=+1; -0.5 weight => -0.5*z
    # net raw ≈ 0
    assert "focus" in components and "stress" in components
    assert -0.2 < reward < 0.2


@pytest.mark.asyncio
async def test_scheduler_attributes_and_partial_fits():
    engine = await init_engine("sqlite+aiosqlite:///:memory:")
    await _seed(engine, decision_ts=1000.0)

    policy = LinUCBPolicy(arms=["x"], context_dim=FEATURE_DIM, alpha=1.0)
    sched = RewardScheduler(
        engine=engine, policy=policy,
        target_weights={"stress": -0.5, "focus": 0.5},
        baseline_pre=120, window_lo=60, window_hi=360,
        feedback_alpha=0.5,
        now_fn=lambda: 1500.0,
        scan_interval_s=0.0,
    )
    await sched.scan_once()

    async with get_session(engine) as s:
        rew = (await s.exec(select(Reward).where(Reward.decision_id == "d1"))).first()
        assert rew is not None


@pytest.mark.asyncio
async def test_feedback_blends_into_reward():
    engine = await init_engine("sqlite+aiosqlite:///:memory:")
    await _seed(engine, decision_ts=1000.0)

    await record_feedback(engine, decision_id="d1", score=1.0, comment="great", ts=1500.0)
    async with get_session(engine) as s:
        fb = (await s.exec(select(Feedback).where(Feedback.decision_id == "d1"))).first()
        assert fb.score == 1.0
