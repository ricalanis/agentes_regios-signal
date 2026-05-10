"""SSE debug sink: mirrors WS-out frames over /debug/stream."""
from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

from sensable_portfolio.stream.pubsub import PubSub


async def sse_debug_stream(bus: PubSub) -> AsyncIterator[dict]:
    """Yield SSE-shaped events: {event, data} for both signals and actions."""
    async with bus.subscribe("signals") as qs, bus.subscribe("actions") as qa:
        while True:
            done, _ = await asyncio.wait(
                {asyncio.create_task(qs.get()), asyncio.create_task(qa.get())},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in done:
                msg = t.result()
                ev = "signals" if msg.get("type") == "mood" else "actions"
                yield {"event": ev, "data": json.dumps(msg)}
