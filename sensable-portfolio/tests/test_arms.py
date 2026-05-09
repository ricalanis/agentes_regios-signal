import numpy as np
import pytest

from sensable_portfolio.arms.registry import ArmRegistry, ArmRow
from sensable_portfolio.arms.factory import build_arm_runnable
from sensable_portfolio.contracts import Intervention


def test_registry_loads_seed_personas():
    reg = ArmRegistry.from_default_pkg()
    arm_ids = [a.id for a in reg.active_arms()]
    expected_personas = {
        "breath_coach", "micro_break", "reframe_cbt",
        "body_scan", "env_tweak", "social_nudge", "deep_focus",
    }
    personas = {a.persona for a in reg.active_arms()}
    assert expected_personas <= personas
    assert len(arm_ids) == len(set(arm_ids))


def test_registry_add_and_retire():
    reg = ArmRegistry.from_default_pkg()
    n0 = len(reg.active_arms())
    reg.add(ArmRow(id="evolved.x", persona="breath_coach", prompt_id="breath_coach.v2",
                   model="fake", parent_id="breath_coach.v1", created_at=1.0))
    assert len(reg.active_arms()) == n0 + 1
    reg.retire("evolved.x", at=2.0)
    assert "evolved.x" not in [a.id for a in reg.active_arms()]


@pytest.mark.asyncio
async def test_factory_runs_with_fake_llm():
    """Use a RunnableLambda to simulate the LLM, bypassing real API."""
    from langchain_core.runnables import RunnableLambda

    def fake_llm(_inputs):
        return Intervention(
            decision_id="d1", arm_id="breath_coach.v1", ts=1.0,
            action_type="breath", title="Box breath",
            body="Inhale 4s, hold 4s, exhale 4s, hold 4s.",
            duration_s=90.0, intensity="low", rationale="from fake llm",
        )

    runnable = build_arm_runnable(
        arm=ArmRow(id="breath_coach.v1", persona="breath_coach",
                   prompt_id="breath_coach.v1", model="fake", parent_id=None,
                   created_at=0.0),
        llm_factory=lambda _model: RunnableLambda(fake_llm),
    )
    out = await runnable.ainvoke({
        "decision_id": "d1", "ts": 1.0,
        "context_features": np.zeros(36).tolist(),
        "signals_at_decision": {k: 0.0 for k in (
            "focus","stress","valence","arousal","joy","calm","excitement","neutral")},
    })
    assert isinstance(out, Intervention)
    assert out.arm_id == "breath_coach.v1"
