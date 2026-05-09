"""Decision-tick scheduler: fires when new_data + min_interval gate is open."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable


@dataclass
class TickScheduler:
    new_data: asyncio.Event
    min_interval_s: float
    on_tick: Callable[[], Awaitable[None]]

    async def run(self) -> None:
        last = 0.0
        loop = asyncio.get_running_loop()
        while True:
            await self.new_data.wait()
            self.new_data.clear()
            now = loop.time()
            wait = max(0.0, self.min_interval_s - (now - last))
            if wait > 0:
                await asyncio.sleep(wait)
            last = loop.time()
            try:
                await self.on_tick()
            except Exception:
                import logging
                logging.exception("on_tick failed")
