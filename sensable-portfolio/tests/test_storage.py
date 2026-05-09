import pytest
from sensable_portfolio.storage.db import init_engine, get_session
from sensable_portfolio.storage.models import (
    Decision, Reward, Feedback, Arm, PolicySnapshot, SnapshotLog,
)


@pytest.mark.asyncio
async def test_round_trip_decision_and_reward():
    engine = await init_engine("sqlite+aiosqlite:///:memory:")
    async with get_session(engine) as s:
        d = Decision(
            id="d1", ts=1000.0, arm_id="breath_coach.v1", target_id="default",
            context_json="{}", intervention_json='{"action_type":"breath"}',
            run_id=None,
        )
        s.add(d)
        await s.commit()

    async with get_session(engine) as s:
        r = Reward(
            decision_id="d1", components_json='{"focus":0.3,"stress":0.5}',
            user_score=None, reward=0.4, computed_at=2000.0,
        )
        s.add(r)
        await s.commit()

    async with get_session(engine) as s:
        from sqlmodel import select
        got = (await s.exec(select(Decision).where(Decision.id == "d1"))).first()
        rew = (await s.exec(select(Reward).where(Reward.decision_id == "d1"))).first()
        assert got.arm_id == "breath_coach.v1"
        assert rew.reward == 0.4


@pytest.mark.asyncio
async def test_snapshot_log_indexed_by_kind_ts():
    engine = await init_engine("sqlite+aiosqlite:///:memory:")
    async with get_session(engine) as s:
        for k in ("focus", "stress"):
            for i in range(3):
                s.add(SnapshotLog(ts=float(i), kind=k, value=float(i) * 0.1))
        await s.commit()

    async with get_session(engine) as s:
        from sqlmodel import select
        rows = (await s.exec(
            select(SnapshotLog).where(SnapshotLog.kind == "focus").order_by(SnapshotLog.ts)
        )).all()
        assert [r.value for r in rows] == [0.0, 0.1, 0.2]
