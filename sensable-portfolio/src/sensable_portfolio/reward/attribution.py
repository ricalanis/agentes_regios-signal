"""Compute the goal-conditioned reward for a decision from SnapshotLog history."""
from __future__ import annotations

import math
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import select

from sensable_portfolio.storage.db import get_session
from sensable_portfolio.storage.models import Decision, Feedback, SnapshotLog


def _z_safe(delta: float, std: float) -> float:
    if std <= 1e-9 or math.isnan(std):
        return 0.0
    return delta / std


async def _mean_in_window(engine: AsyncEngine, kind: str, t_lo: float, t_hi: float) -> tuple[float, float]:
    async with get_session(engine) as s:
        rows = (await s.exec(
            select(SnapshotLog.value).where(
                (SnapshotLog.kind == kind)
                & (SnapshotLog.ts >= t_lo)
                & (SnapshotLog.ts < t_hi)
            )
        )).all()
    vals = [float(v) for v in rows]
    if not vals:
        return (0.0, 0.0)
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / max(1, n - 1)
    return (mean, math.sqrt(var))


async def _running_std_1h(engine: AsyncEngine, kind: str, t: float) -> float:
    _, std = await _mean_in_window(engine, kind, t - 3600.0, t)
    return std if std > 1e-9 else 1.0


async def attribute(
    engine: AsyncEngine,
    decision_id: str,
    target_weights: dict[str, float],
    baseline_pre: int,
    window_lo: int,
    window_hi: int,
    alpha: float,
) -> tuple[dict[str, float], float]:
    async with get_session(engine) as s:
        d = (await s.exec(select(Decision).where(Decision.id == decision_id))).first()
        if d is None:
            raise ValueError(decision_id)
        fb = (await s.exec(select(Feedback).where(Feedback.decision_id == decision_id))).first()

    components: dict[str, float] = {}
    raw = 0.0
    for kind, weight in target_weights.items():
        b_mean, _ = await _mean_in_window(engine, kind, d.ts - baseline_pre, d.ts)
        o_mean, _ = await _mean_in_window(engine, kind, d.ts + window_lo, d.ts + window_hi)
        std = await _running_std_1h(engine, kind, d.ts)
        z = _z_safe(o_mean - b_mean, std)
        components[kind] = float(weight * z)
        raw += components[kind]

    user_score = float(fb.score) if fb is not None else 0.0
    reward = max(-1.0, min(1.0, raw + (alpha * user_score if fb is not None else 0.0)))
    return components, reward
