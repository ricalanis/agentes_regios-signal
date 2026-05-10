"""WebSocket client to the renderer at ws://127.0.0.1:7777.

One connection carries:
  - type:"mood"          frames at signals_hz, drawn from signals channel
  - type:"agent_action"  frames per decision, drawn from actions channel

Fire-and-forget: never blocks the rest of the system. 1s→5s reconnect."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

import websockets

from sensable_portfolio.stream.pubsub import PubSub

log = logging.getLogger(__name__)


@dataclass
class RendererClient:
    url: str
    bus: PubSub
    signals_hz: float = 1.0
    backoff_initial: float = 1.0
    backoff_max: float = 5.0

    def __post_init__(self) -> None:
        self._cancel = asyncio.Event()

    def cancel(self) -> None:
        self._cancel.set()

    async def run(self) -> None:
        backoff = self.backoff_initial
        while not self._cancel.is_set():
            try:
                async with websockets.connect(self.url, open_timeout=2.0) as ws:
                    backoff = self.backoff_initial
                    await self._send_loop(ws)
            except Exception as e:
                log.info("renderer client lost (%s); reconnect in %.2fs", e, backoff)
                try:
                    await asyncio.wait_for(self._cancel.wait(), timeout=backoff)
                    return
                except asyncio.TimeoutError:
                    pass
                backoff = min(self.backoff_max, backoff * 2 if backoff > 0 else self.backoff_initial)

    async def _send_loop(self, ws) -> None:
        period = 1.0 / max(self.signals_hz, 0.001)
        last_signal_send = 0.0
        async with self.bus.subscribe("signals", maxsize=4) as qs, \
                   self.bus.subscribe("actions", maxsize=64) as qa:
            while not self._cancel.is_set():
                signal_task = asyncio.create_task(qs.get())
                action_task = asyncio.create_task(qa.get())
                # recv_task detects server-side close / unexpected disconnection
                recv_task = asyncio.create_task(ws.recv())
                done, pending = await asyncio.wait(
                    {signal_task, action_task, recv_task},
                    return_when=asyncio.FIRST_COMPLETED, timeout=period,
                )
                for p in pending:
                    p.cancel()
                # If the recv task finished, the server sent us something or closed
                if recv_task in done:
                    exc = recv_task.exception() if not recv_task.cancelled() else None
                    if exc is not None or recv_task.result() is not None:
                        # Connection closed by server → raise to trigger reconnect
                        raise websockets.exceptions.ConnectionClosed(None, None)
                now = asyncio.get_running_loop().time()
                if action_task in done:
                    msg = action_task.result()
                    await ws.send(json.dumps(msg))
                if signal_task in done:
                    msg = signal_task.result()
                    if (now - last_signal_send) >= period:
                        await ws.send(json.dumps(msg))
                        last_signal_send = now
