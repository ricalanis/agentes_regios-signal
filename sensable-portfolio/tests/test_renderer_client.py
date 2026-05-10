import asyncio
import json
import pytest
import websockets

from sensable_portfolio.contracts import (
    AgentInfo, AgentActionFrame, Intervention, MoodFrame,
)
from sensable_portfolio.renderer.client import RendererClient
from sensable_portfolio.stream.pubsub import PubSub


@pytest.mark.asyncio
async def test_client_sends_mood_and_action_on_one_socket():
    received = []

    async def handler(ws):
        async for msg in ws:
            received.append(json.loads(msg))

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        bus = PubSub()
        client = RendererClient(url=f"ws://127.0.0.1:{port}", bus=bus,
                                signals_hz=20.0)
        task = asyncio.create_task(client.run())
        await asyncio.sleep(0.05)

        mood = MoodFrame(vector={"focus": 0.1, "stress": 0.2, "valence": 0.0,
                                 "arousal": 0.0, "joy": 0.0, "calm": 0.0,
                                 "excitement": 0.0, "neutral": 0.0}, ts=1)
        action = AgentActionFrame(
            ts=2, decision_id="d",
            agent=AgentInfo(id="x", persona="p", model="m"),
            intervention=Intervention(decision_id="d", arm_id="x", ts=2.0,
                action_type="breath", title="t", body="b",
                duration_s=10.0, intensity="low", rationale="r"),
            signals_at_decision={"focus": 0.0, "stress": 0.0, "valence": 0.0,
                                 "arousal": 0.0, "joy": 0.0, "calm": 0.0,
                                 "excitement": 0.0, "neutral": 0.0},
        )
        await bus.publish("signals", json.loads(mood.model_dump_json()))
        await bus.publish("actions", json.loads(action.model_dump_json()))
        await asyncio.sleep(0.2)

        client.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        types = [m["type"] for m in received]
        assert "mood" in types
        assert "agent_action" in types


@pytest.mark.asyncio
async def test_client_reconnects_after_close():
    accepted = 0

    async def handler(ws):
        nonlocal accepted
        accepted += 1
        await ws.close()

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        bus = PubSub()
        client = RendererClient(url=f"ws://127.0.0.1:{port}", bus=bus,
                                backoff_initial=0.05, backoff_max=0.1)
        task = asyncio.create_task(client.run())
        await asyncio.sleep(0.4)
        client.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert accepted >= 2
