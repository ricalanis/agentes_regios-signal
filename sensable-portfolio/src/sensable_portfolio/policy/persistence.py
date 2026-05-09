"""Persist/restore policy state through PolicySnapshot rows."""
from __future__ import annotations

import time
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import select

from sensable_portfolio.policy.linucb import LinUCBPolicy
from sensable_portfolio.storage.db import get_session
from sensable_portfolio.storage.models import PolicySnapshot


async def save(engine: AsyncEngine, policy: LinUCBPolicy, algo: str = "linucb_disjoint") -> None:
    blob = policy.snapshot()
    async with get_session(engine) as s:
        s.add(PolicySnapshot(ts=time.time(), algo=algo, blob=blob))
        await s.commit()


async def load_latest(engine: AsyncEngine, policy: LinUCBPolicy) -> bool:
    """Restore the most-recent snapshot. Returns True iff something was loaded."""
    async with get_session(engine) as s:
        row = (await s.exec(
            select(PolicySnapshot).order_by(PolicySnapshot.ts.desc()).limit(1)
        )).first()
        if row is None:
            return False
        policy.restore(row.blob)
        return True
