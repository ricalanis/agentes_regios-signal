import asyncio
import pytest
from neurable_connector import Baseline, FS_HZ
from .conftest import FakeSource

from sensable_portfolio.connector.runner import ConnectorRunner
from sensable_portfolio.signals.view import build_registry, ALL_DIMS
from sensable_portfolio.storage.db import init_engine, get_session
from sensable_portfolio.storage.models import SnapshotLog
from sqlmodel import select


@pytest.mark.asyncio
async def test_runner_pushes_to_registry_and_writes_snapshot_log():
    baseline = Baseline.fit(list(FakeSource(runtime_s=1.1, seed=1)), fs=float(FS_HZ))
    src = FakeSource(runtime_s=1.1, seed=2)

    engine = await init_engine("sqlite+aiosqlite:///:memory:")
    registry = build_registry()
    new_data = asyncio.Event()
    signals_pubsub = []

    async def on_signal(sample):
        signals_pubsub.append(sample)

    runner = ConnectorRunner(
        source=src, baseline=baseline, registry=registry, engine=engine,
        on_new_data=lambda: new_data.set(), on_signal=on_signal,
        snapshot_log_hz=4.0,
    )
    await runner.run()

    assert len(signals_pubsub) >= 1
    assert new_data.is_set()
    snaps = registry.snapshot_all()
    assert set(snaps.keys()) >= set(ALL_DIMS)
    async with get_session(engine) as s:
        rows = (await s.exec(select(SnapshotLog))).all()
        assert any(r.kind == "focus" for r in rows)
        assert any(r.kind == "stress" for r in rows)
