import asyncio
import pytest
from langchain_core.runnables import RunnableLambda

from sensable_portfolio.contracts import Intervention
from sensable_portfolio.policy.linucb import LinUCBPolicy
from sensable_portfolio.arms.registry import ArmRegistry
from sensable_portfolio.signals.view import build_registry, ALL_DIMS, CONTEXT_KINDS
from sensable_portfolio.signals.features import FEATURE_DIM
from sensable_portfolio.graph.decision import DecisionGraph
from sensable_portfolio.storage.db import init_engine


def _fake_llm_factory(_model):
    def _fn(_):
        return Intervention(
            decision_id="will_overwrite", arm_id="will_overwrite", ts=0.0,
            action_type="breath", title="t", body="b",
            duration_s=10.0, intensity="low", rationale="r",
        )
    return RunnableLambda(_fn)


@pytest.mark.asyncio
async def test_decision_graph_emits_intervention_and_persists_decision():
    registry = build_registry()
    for k in ALL_DIMS:
        for i in range(5):
            registry.push(k, float(i), 0.1)

    arm_reg = ArmRegistry.from_default_pkg()
    arm_ids = [a.id for a in arm_reg.active_arms()]
    policy = LinUCBPolicy(arms=arm_ids, context_dim=FEATURE_DIM, alpha=1.0)
    engine = await init_engine("sqlite+aiosqlite:///:memory:")

    emitted: list = []

    async def on_emit(event):
        emitted.append(event)

    graph = DecisionGraph(
        signal_registry=registry, arm_registry=arm_reg, policy=policy,
        engine=engine, llm_factory=_fake_llm_factory, on_emit=on_emit,
    )
    await graph.run_one()
    assert len(emitted) == 1
    ev = emitted[0]
    assert ev["arm_id"] in arm_ids
    assert ev["intervention"].decision_id == ev["decision_id"]
    assert set(ev["signals_at_decision"]) == set(ALL_DIMS)
