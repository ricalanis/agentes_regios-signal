import asyncio
import pytest

from sensable_portfolio.tick.scheduler import TickScheduler


@pytest.mark.asyncio
async def test_first_tick_fires_immediately_after_new_data():
    new_data = asyncio.Event()
    fired = []

    async def on_tick():
        fired.append(asyncio.get_running_loop().time())

    sched = TickScheduler(new_data=new_data, min_interval_s=0.05, on_tick=on_tick)
    task = asyncio.create_task(sched.run())
    new_data.set()
    await asyncio.sleep(0.02)
    assert len(fired) == 1
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_min_interval_debounces_subsequent_ticks():
    new_data = asyncio.Event()
    fired = []

    async def on_tick():
        fired.append(1)

    sched = TickScheduler(new_data=new_data, min_interval_s=0.1, on_tick=on_tick)
    task = asyncio.create_task(sched.run())

    new_data.set(); await asyncio.sleep(0.02)
    new_data.set(); await asyncio.sleep(0.03)
    new_data.set(); await asyncio.sleep(0.12)
    assert len(fired) == 2

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
