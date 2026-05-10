import asyncio
import pytest
from httpx import ASGITransport, AsyncClient

from sensable_portfolio.app import build_app

_TEST_DB = "sqlite+aiosqlite:///:memory:"


@pytest.mark.asyncio
async def test_healthz_returns_status_block():
    app = build_app(start_runtime=False, db_url=_TEST_DB)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        async with app.router.lifespan_context(app):
            r = await ac.get("/healthz")
            assert r.status_code == 200
            body = r.json()
            assert body["status"] in ("ok", "starting")
            assert "decisions_total" in body


@pytest.mark.asyncio
async def test_feedback_records():
    app = build_app(start_runtime=False, db_url=_TEST_DB)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        async with app.router.lifespan_context(app):
            from sensable_portfolio.storage.db import get_session
            from sensable_portfolio.storage.models import Decision
            engine = app.state.engine
            async with get_session(engine) as s:
                s.add(Decision(id="d1", ts=1.0, arm_id="x", target_id="default",
                               context_json="[]", intervention_json="{}", run_id=None))
                await s.commit()

            r = await ac.post("/feedback", json={
                "decision_id": "d1", "score": 0.7, "comment": "nice",
            })
            assert r.status_code == 200
            assert r.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_arms_leaderboard_lists_seed_arms():
    app = build_app(start_runtime=False, db_url=_TEST_DB)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        async with app.router.lifespan_context(app):
            r = await ac.get("/arms/leaderboard")
            assert r.status_code == 200
            rows = r.json()
            assert len(rows) >= 7
            assert {row["persona"] for row in rows} >= {"breath_coach", "deep_focus"}
