"""Periodic mutator: pick top-K arms, ask an LLM to mutate, register variant."""
from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from langchain_core.runnables import Runnable
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import select

from sensable_portfolio.arms.registry import ArmRegistry, ArmRow
from sensable_portfolio.policy.linucb import LinUCBPolicy
from sensable_portfolio.storage.db import get_session
from sensable_portfolio.storage.models import Decision, Reward


@dataclass
class MetaEvolver:
    engine: AsyncEngine
    arm_registry: ArmRegistry
    policy: LinUCBPolicy
    mutator_factory: Callable[[str], Runnable]
    top_k: int = 2
    min_pulls: int = 20

    async def _mean_rewards(self) -> dict[str, tuple[int, float]]:
        async with get_session(self.engine) as s:
            rows = (await s.exec(
                select(Decision.arm_id, Reward.reward).join(
                    Reward, Reward.decision_id == Decision.id,
                )
            )).all()
        bag: dict[str, list[float]] = defaultdict(list)
        for arm_id, r in rows:
            bag[arm_id].append(float(r))
        return {k: (len(v), sum(v) / len(v)) for k, v in bag.items()}

    async def run_once(self) -> int:
        means = await self._mean_rewards()
        eligible = [(arm_id, n, m) for arm_id, (n, m) in means.items() if n >= self.min_pulls]
        eligible.sort(key=lambda x: x[2], reverse=True)
        top = eligible[: self.top_k]
        added = 0
        for arm_id, n, m in top:
            try:
                parent = self.arm_registry.get(arm_id)
            except KeyError:
                continue
            mutator = self.mutator_factory(parent.model)
            out = await mutator.ainvoke({
                "parent_id": parent.id,
                "parent_persona": parent.persona,
                "parent_system": parent.system,
                "parent_human": parent.human,
                "recent_reward": round(m, 4),
            })
            new_id = out.get("id") or f"evolved.{parent.persona}.{uuid.uuid4().hex[:6]}"
            new = ArmRow(
                id=new_id, persona=parent.persona,
                prompt_id=new_id, model=parent.model,
                parent_id=parent.id, created_at=time.time(),
                system=out.get("system", parent.system),
                human=out.get("human", parent.human),
            )
            self.arm_registry.add(new)
            self.policy.add_arm(new.id)
            added += 1
        return added

    async def run(self, period_s: float) -> None:
        if period_s <= 0:
            return
        while True:
            try:
                await self.run_once()
            except Exception:
                import logging; logging.exception("evolver run_once failed")
            await asyncio.sleep(period_s)
