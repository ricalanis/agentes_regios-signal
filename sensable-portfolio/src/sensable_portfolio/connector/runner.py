"""Run nc.stream() → push to SignalRegistry, write SnapshotLog, fire callbacks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

from neurable_connector import AffectSample, Baseline, NeurableConnector, Source
from pidview import SignalRegistry
from sqlalchemy.ext.asyncio import AsyncEngine

from sensable_portfolio.signals.view import ALL_DIMS
from sensable_portfolio.storage.db import get_session
from sensable_portfolio.storage.models import SnapshotLog
from sensable_portfolio.connector.source import build_connector


@dataclass
class ConnectorRunner:
    source: Source
    baseline: Baseline
    registry: SignalRegistry
    engine: AsyncEngine
    on_new_data: Callable[[], None]
    on_signal: Callable[[AffectSample], Awaitable[None]]
    snapshot_log_hz: float = 1.0

    async def run(self) -> None:
        nc = build_connector(self.source, self.baseline)
        last_log_t: float = 0.0
        log_period = 1.0 / self.snapshot_log_hz

        async with nc as conn:
            async for s in conn.stream():
                for name in ALL_DIMS:
                    self.registry.push(name, s.t, getattr(s, name))
                self.on_new_data()
                await self.on_signal(s)

                if s.t - last_log_t >= log_period:
                    last_log_t = s.t
                    async with get_session(self.engine) as sess:
                        for name in ALL_DIMS:
                            sess.add(SnapshotLog(ts=s.t, kind=name, value=getattr(s, name)))
                        await sess.commit()
