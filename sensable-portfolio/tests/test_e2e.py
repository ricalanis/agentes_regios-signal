"""End-to-end smoke test: FakeSource → bandit graph → renderer WebSocket."""
from __future__ import annotations

import asyncio
import json
import tempfile
import os

import pytest
import websockets
from httpx import ASGITransport, AsyncClient

from sensable_portfolio.app import build_app
from sensable_portfolio.connector.runner import ConnectorRunner
from sensable_portfolio.connector.source import calibrate_baseline
from sensable_portfolio.tick.scheduler import TickScheduler
from .conftest import FakeSource


@pytest.mark.asyncio
async def test_e2e_with_fake_source_emits_action_to_renderer():
    """Wire app + FakeSource + fake renderer; assert mood + action both emitted."""

    received: list[dict] = []

    async def handler(ws):
        async for msg in ws:
            received.append(json.loads(msg))

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        # Use a temp file DB to avoid concurrent in-memory SQLite limitations
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            db_url = f"sqlite+aiosqlite:///{db_path}"
            app = build_app(start_runtime=False, db_url=db_url)
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
                async with app.router.lifespan_context(app):
                    from sensable_portfolio.renderer.client import RendererClient
                    renderer = RendererClient(
                        url=f"ws://127.0.0.1:{port}",
                        bus=app.state.bus, signals_hz=20.0,
                    )

                    # Need at least 1 second of synthetic data for Baseline.fit (500 frames)
                    source = FakeSource(runtime_s=1.1, seed=11)
                    baseline = calibrate_baseline(FakeSource(runtime_s=1.1, seed=10))
                    runner = ConnectorRunner(
                        source=source, baseline=baseline,
                        registry=app.state.signal_registry,
                        engine=app.state.engine,
                        on_new_data=lambda: app.state.new_data.set(),
                        on_signal=app.state.on_signal_sample,
                        snapshot_log_hz=4.0,
                    )
                    tick = TickScheduler(
                        new_data=app.state.new_data, min_interval_s=0.1,
                        on_tick=app.state.graph.run_one,
                    )
                    tasks = [
                        asyncio.create_task(renderer.run()),
                        asyncio.create_task(runner.run()),
                        asyncio.create_task(tick.run()),
                    ]
                    await asyncio.sleep(2.0)  # give the synthetic stream time to play out
                    renderer.cancel()
                    for t in tasks:
                        t.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            try:
                os.unlink(db_path)
            except OSError:
                pass

        types = [m["type"] for m in received]
        assert "mood" in types
        assert "agent_action" in types
