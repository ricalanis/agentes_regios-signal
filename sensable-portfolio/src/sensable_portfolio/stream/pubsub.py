"""In-process asyncio fan-out bus with named channels."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any


class PubSub:
    def __init__(self) -> None:
        self._channels: dict[str, set[asyncio.Queue]] = {}

    async def publish(self, channel: str, message: Any) -> None:
        for q in list(self._channels.get(channel, ())):
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                q.put_nowait(message)

    @asynccontextmanager
    async def subscribe(self, channel: str, maxsize: int = 256):
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._channels.setdefault(channel, set()).add(q)
        try:
            yield q
        finally:
            self._channels[channel].discard(q)
