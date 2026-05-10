"""Glue all decision-graph nodes into a runnable async sequence."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from langchain_core.runnables import Runnable
from sqlalchemy.ext.asyncio import AsyncEngine

from sensable_portfolio.arms.registry import ArmRegistry
from sensable_portfolio.graph.nodes import emit, featurize, persist, run_arm, select
from sensable_portfolio.policy.linucb import LinUCBPolicy
from pidview import SignalRegistry


@dataclass
class DecisionGraph:
    signal_registry: SignalRegistry
    arm_registry: ArmRegistry
    policy: LinUCBPolicy
    engine: AsyncEngine
    llm_factory: Callable[[str], Runnable]
    on_emit: Callable[[dict[str, Any]], Awaitable[None]]

    async def run_one(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "signal_registry": self.signal_registry,
            "arm_registry": self.arm_registry,
            "policy": self.policy,
            "engine": self.engine,
            "llm_factory": self.llm_factory,
            "on_emit": self.on_emit,
            "ts": time.time(),
        }
        state = await featurize(state)
        state = await select(state)
        state = await run_arm(state)
        state = await persist(state)
        state = await emit(state)
        return state
