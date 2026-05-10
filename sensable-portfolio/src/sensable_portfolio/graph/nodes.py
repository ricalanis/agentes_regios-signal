"""Pure functions for each step in the decision graph.

We deliberately keep these as plain async functions so the test suite
exercises them without spinning up a LangGraph runtime; the LangGraph
compile step in decision.py is a thin wrapper that orders them."""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

import numpy as np
from sqlalchemy.ext.asyncio import AsyncEngine

from sensable_portfolio.arms.factory import build_arm_runnable
from sensable_portfolio.arms.registry import ArmRegistry
from sensable_portfolio.contracts import Intervention
from sensable_portfolio.policy.linucb import LinUCBPolicy
from sensable_portfolio.signals.features import FEATURE_DIM, build_feature_vector
from sensable_portfolio.signals.view import ALL_DIMS
from sensable_portfolio.storage.db import get_session
from sensable_portfolio.storage.models import Decision


async def featurize(state: dict[str, Any]) -> dict[str, Any]:
    snaps = state["signal_registry"].snapshot_all()
    state["context"] = build_feature_vector(snaps)
    state["signals_at_decision"] = {
        k: float(snaps[k].present) if k in snaps else 0.0 for k in ALL_DIMS
    }
    return state


async def select(state: dict[str, Any]) -> dict[str, Any]:
    policy: LinUCBPolicy = state["policy"]
    state["arm_id"] = policy.predict(state["context"])
    return state


async def run_arm(state: dict[str, Any]) -> dict[str, Any]:
    reg: ArmRegistry = state["arm_registry"]
    arm = reg.get(state["arm_id"])
    runnable = build_arm_runnable(arm, state["llm_factory"])
    decision_id = state.get("decision_id") or uuid.uuid4().hex
    state["decision_id"] = decision_id
    out = await runnable.ainvoke({
        "decision_id": decision_id,
        "ts": state.get("ts") or time.time(),
        "context_features": state["context"].tolist(),
        "signals_at_decision": state["signals_at_decision"],
    })
    if not isinstance(out, Intervention):
        raise TypeError(f"Arm {arm.id} did not return an Intervention")
    state["intervention"] = out.model_copy(update={
        "decision_id": decision_id, "arm_id": arm.id,
    })
    return state


async def persist(state: dict[str, Any]) -> dict[str, Any]:
    engine: AsyncEngine = state["engine"]
    inter: Intervention = state["intervention"]
    async with get_session(engine) as s:
        s.add(Decision(
            id=inter.decision_id, ts=inter.ts, arm_id=inter.arm_id,
            target_id="default",
            context_json=json.dumps(state["context"].tolist()),
            intervention_json=inter.model_dump_json(),
            run_id=None,
        ))
        await s.commit()
    return state


async def emit(state: dict[str, Any]) -> dict[str, Any]:
    inter: Intervention = state["intervention"]
    arm = state["arm_registry"].get(inter.arm_id)
    event = {
        "decision_id": inter.decision_id,
        "arm_id": inter.arm_id,
        "agent": {"id": arm.id, "persona": arm.persona, "model": arm.model,
                  "parent_id": arm.parent_id},
        "intervention": inter,
        "signals_at_decision": state["signals_at_decision"],
        "ts": inter.ts,
    }
    on_emit = state.get("on_emit")
    if on_emit is not None:
        await on_emit(event)
    return state
