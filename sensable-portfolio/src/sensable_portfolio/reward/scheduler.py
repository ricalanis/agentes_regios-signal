"""Background scanner: attribute rewards for decisions whose post-window has closed."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Callable

from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import select

from sensable_portfolio.policy.linucb import LinUCBPolicy
from sensable_portfolio.reward.attribution import attribute
from sensable_portfolio.signals.features import FEATURE_DIM
from sensable_portfolio.storage.db import get_session
from sensable_portfolio.storage.models import Decision, Reward
import numpy as np


@dataclass
class RewardScheduler:
    engine: AsyncEngine
    policy: LinUCBPolicy
    target_weights: dict[str, float]
    baseline_pre: int
    window_lo: int
    window_hi: int
    feedback_alpha: float
    now_fn: Callable[[], float] = time.time
    scan_interval_s: float = 30.0

    async def scan_once(self) -> int:
        cutoff = self.now_fn() - self.window_hi
        async with get_session(self.engine) as s:
            unscored = (await s.exec(
                select(Decision).where(
                    (Decision.ts <= cutoff)
                    & (~Decision.id.in_(select(Reward.decision_id)))
                )
            )).all()
        n = 0
        for d in unscored:
            try:
                components, reward = await attribute(
                    self.engine, d.id, self.target_weights,
                    self.baseline_pre, self.window_lo, self.window_hi,
                    self.feedback_alpha,
                )
                async with get_session(self.engine) as s:
                    s.add(Reward(
                        decision_id=d.id,
                        components_json=json.dumps(components),
                        user_score=None, reward=reward,
                        computed_at=self.now_fn(),
                    ))
                    await s.commit()
                ctx = np.asarray(json.loads(d.context_json), dtype=np.float64)
                if ctx.shape == (FEATURE_DIM,):
                    self.policy.partial_fit(ctx, d.arm_id, reward)
                n += 1
            except Exception:
                import logging; logging.exception("attribute failed for %s", d.id)
        return n

    async def run(self) -> None:
        while True:
            try:
                await self.scan_once()
            except Exception:
                import logging; logging.exception("reward scheduler scan failed")
            await asyncio.sleep(max(0.1, self.scan_interval_s))
