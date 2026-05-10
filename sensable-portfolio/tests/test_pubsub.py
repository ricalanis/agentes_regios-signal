import asyncio
import pytest

from sensable_portfolio.stream.pubsub import PubSub


@pytest.mark.asyncio
async def test_pubsub_fans_out_to_subscribers():
    bus = PubSub()
    seen_a, seen_b = [], []

    async def consume(seen):
        async with bus.subscribe("signals") as q:
            try:
                while True:
                    seen.append(await asyncio.wait_for(q.get(), timeout=0.2))
            except asyncio.TimeoutError:
                return

    ta = asyncio.create_task(consume(seen_a))
    tb = asyncio.create_task(consume(seen_b))
    await asyncio.sleep(0.02)
    await bus.publish("signals", {"x": 1})
    await bus.publish("signals", {"x": 2})
    await asyncio.gather(ta, tb)
    assert seen_a == [{"x": 1}, {"x": 2}]
    assert seen_b == [{"x": 1}, {"x": 2}]


@pytest.mark.asyncio
async def test_pubsub_channels_are_isolated():
    bus = PubSub()
    s_seen, a_seen = [], []

    async def consume_s():
        async with bus.subscribe("signals") as q:
            try:
                s_seen.append(await asyncio.wait_for(q.get(), timeout=0.2))
            except asyncio.TimeoutError:
                pass

    async def consume_a():
        async with bus.subscribe("actions") as q:
            try:
                a_seen.append(await asyncio.wait_for(q.get(), timeout=0.2))
            except asyncio.TimeoutError:
                pass

    t1 = asyncio.create_task(consume_s())
    t2 = asyncio.create_task(consume_a())
    await asyncio.sleep(0.02)
    await bus.publish("actions", {"a": 1})
    await asyncio.gather(t1, t2)
    assert s_seen == []
    assert a_seen == [{"a": 1}]
