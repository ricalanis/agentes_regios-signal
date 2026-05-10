import asyncio
import pytest

from langchain_core.runnables import RunnableLambda

from sensable_portfolio.arms.registry import ArmRegistry, ArmRow
from sensable_portfolio.evolve.meta import MetaEvolver
from sensable_portfolio.policy.linucb import LinUCBPolicy
from sensable_portfolio.signals.features import FEATURE_DIM
from sensable_portfolio.storage.db import init_engine, get_session
from sensable_portfolio.storage.models import Reward, Decision


def _fake_mutator_factory(_model: str):
    def _fn(inputs: dict):
        return {
            "id": f"evolved.{inputs['parent_id'].split('.')[0]}.x",
            "system": "You are an evolved variant.",
            "human": inputs.get("human", "Propose one Intervention."),
        }
    return RunnableLambda(_fn)


@pytest.mark.asyncio
async def test_evolver_picks_top_arm_and_registers_variant():
    engine = await init_engine("sqlite+aiosqlite:///:memory:")
    arm_reg = ArmRegistry.from_default_pkg()
    arms = arm_reg.active_arms()
    parent = arms[0]

    async with get_session(engine) as s:
        for i in range(5):
            d_id = f"d{i}"
            s.add(Decision(id=d_id, ts=float(i), arm_id=parent.id,
                           target_id="default", context_json="[]",
                           intervention_json="{}", run_id=None))
            s.add(Reward(decision_id=d_id, components_json="{}",
                         user_score=None, reward=0.8, computed_at=float(i)))
        await s.commit()

    policy = LinUCBPolicy(arms=[a.id for a in arms], context_dim=FEATURE_DIM, alpha=1.0)
    n0 = len(arm_reg.active_arms())

    ev = MetaEvolver(
        engine=engine, arm_registry=arm_reg, policy=policy,
        mutator_factory=_fake_mutator_factory, top_k=1, min_pulls=3,
    )
    await ev.run_once()
    assert len(arm_reg.active_arms()) == n0 + 1
    new = [a for a in arm_reg.active_arms() if a.parent_id == parent.id]
    assert len(new) == 1
    assert new[0].id in policy._arms
