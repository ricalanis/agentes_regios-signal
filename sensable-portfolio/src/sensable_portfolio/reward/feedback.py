"""POST /feedback handler: record a user's outcome score for a decision."""
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine

from sensable_portfolio.storage.db import get_session
from sensable_portfolio.storage.models import Feedback


async def record_feedback(
    engine: AsyncEngine, *, decision_id: str, score: float,
    comment: str | None = None, ts: float,
) -> None:
    if not -1.0 <= score <= 1.0:
        raise ValueError("score must be in [-1, 1]")
    async with get_session(engine) as s:
        s.add(Feedback(decision_id=decision_id, score=score, comment=comment, ts=ts))
        await s.commit()
