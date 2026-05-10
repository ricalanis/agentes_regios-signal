"""FastAPI app: operational routes + lifespan wiring of the live pipeline."""
from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import select

from sensable_portfolio.arms.registry import ArmRegistry
from sensable_portfolio.config import load_settings
from sensable_portfolio.contracts import (
    AgentActionFrame, AgentInfo, Intervention, MoodFrame,
)
from sensable_portfolio.graph.decision import DecisionGraph
from sensable_portfolio.policy.linucb import LinUCBPolicy
from sensable_portfolio.policy.persistence import load_latest, save as save_policy
from sensable_portfolio.renderer.client import RendererClient
from sensable_portfolio.reward.feedback import record_feedback
from sensable_portfolio.reward.scheduler import RewardScheduler
from sensable_portfolio.signals.features import FEATURE_DIM
from sensable_portfolio.signals.view import ALL_DIMS, build_registry
from sensable_portfolio.storage.db import get_session, init_engine
from sensable_portfolio.storage.models import Decision, Reward, Feedback
from sensable_portfolio.stream.pubsub import PubSub


class FeedbackBody(BaseModel):
    decision_id: str
    score: float
    comment: str | None = None


def _stub_llm_factory(_model: str) -> Runnable:
    """Default LLM factory used when no real LLM is configured.

    Returns a deterministic Intervention so the graph is exercisable
    without an API key. Production code should pass a real factory.

    Note: this lambda receives a ChatPromptValue (after prompt formatting),
    not the original dict, so we ignore the input entirely."""
    def _fn(_inputs: Any) -> Intervention:
        return Intervention(
            decision_id="stub",
            arm_id="will_be_overwritten",
            ts=0.0,
            action_type="breath",
            title="Box breath, 90s",
            body="Inhale 4s, hold 4s, exhale 4s, hold 4s. Repeat.",
            duration_s=90.0, intensity="low",
            rationale="stub_llm_factory placeholder",
        )
    return RunnableLambda(_fn)


def build_app(start_runtime: bool = True, db_url: str | None = None) -> FastAPI:
    settings = load_settings()
    _db_url = db_url if db_url is not None else settings.db_url

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = await init_engine(_db_url)
        signal_registry = build_registry()
        arm_registry = ArmRegistry.from_default_pkg()
        policy = LinUCBPolicy(
            arms=[a.id for a in arm_registry.active_arms()],
            context_dim=FEATURE_DIM, alpha=1.0,
        )
        await load_latest(engine, policy)

        bus = PubSub()
        new_data = asyncio.Event()
        decisions_total = {"n": 0}
        last_decision_ts = {"t": 0.0}

        async def on_emit(event):
            decisions_total["n"] += 1
            last_decision_ts["t"] = float(event["ts"])
            arm = arm_registry.get(event["arm_id"])
            frame = AgentActionFrame(
                ts=int(event["ts"] * 1000),
                decision_id=event["decision_id"],
                agent=AgentInfo(id=arm.id, persona=arm.persona, model=arm.model,
                                parent_id=arm.parent_id),
                intervention=event["intervention"],
                signals_at_decision=event["signals_at_decision"],
            )
            await bus.publish("actions", json.loads(frame.model_dump_json()))

        async def on_signal_sample(sample):
            mood = MoodFrame(
                vector={k: float(getattr(sample, k)) for k in ALL_DIMS},
                ts=int(sample.t * 1000),
            )
            await bus.publish("signals", json.loads(mood.model_dump_json()))

        if settings.llm_provider == "ollama":
            from sensable_portfolio.llm.factories import ollama_llm_factory
            llm_factory = ollama_llm_factory(
                default_model=settings.ollama_model,
                base_url=settings.ollama_base_url,
            )
        elif settings.llm_provider == "anthropic":
            from sensable_portfolio.llm.factories import anthropic_llm_factory
            llm_factory = anthropic_llm_factory(
                default_model=settings.anthropic_model,
                api_key=settings.anthropic_api_key,
            )
        else:
            llm_factory = _stub_llm_factory

        graph = DecisionGraph(
            signal_registry=signal_registry, arm_registry=arm_registry,
            policy=policy, engine=engine, llm_factory=llm_factory,
            on_emit=on_emit,
        )

        reward_sched = RewardScheduler(
            engine=engine, policy=policy,
            target_weights=settings.target_weights,
            baseline_pre=settings.baseline_pre,
            window_lo=settings.window_lo,
            window_hi=settings.window_hi,
            feedback_alpha=settings.feedback_alpha,
            scan_interval_s=30.0,
        )

        renderer = (
            RendererClient(url=settings.renderer_ws_url, bus=bus,
                           signals_hz=settings.renderer_signals_hz)
            if settings.renderer_enabled else None
        )

        app.state.engine = engine
        app.state.signal_registry = signal_registry
        app.state.arm_registry = arm_registry
        app.state.policy = policy
        app.state.bus = bus
        app.state.new_data = new_data
        app.state.graph = graph
        app.state.reward_sched = reward_sched
        app.state.renderer = renderer
        app.state.decisions_total = decisions_total
        app.state.last_decision_ts = last_decision_ts
        app.state.on_signal_sample = on_signal_sample
        app.state.start_time = time.time()
        app.state.connector_alive = False
        app.state.tasks = []

        if start_runtime:
            from sensable_portfolio.connector.runner import ConnectorRunner
            from sensable_portfolio.connector.source import calibrate_baseline, production_source
            from sensable_portfolio.tick.scheduler import TickScheduler

            source = production_source()
            baseline = calibrate_baseline(source)
            runner = ConnectorRunner(
                source=source, baseline=baseline, registry=signal_registry,
                engine=engine, on_new_data=lambda: new_data.set(),
                on_signal=on_signal_sample, snapshot_log_hz=1.0,
            )
            tick = TickScheduler(
                new_data=new_data, min_interval_s=settings.min_decision_interval_s,
                on_tick=graph.run_one,
            )
            app.state.tasks.append(asyncio.create_task(runner.run()))
            app.state.tasks.append(asyncio.create_task(tick.run()))
            app.state.tasks.append(asyncio.create_task(reward_sched.run()))
            if renderer is not None:
                app.state.tasks.append(asyncio.create_task(renderer.run()))
            app.state.connector_alive = True

        try:
            yield
        finally:
            for t in app.state.tasks:
                t.cancel()
            if app.state.tasks:
                await asyncio.gather(*app.state.tasks, return_exceptions=True)
            try:
                await save_policy(engine, policy)
            except Exception:
                import logging; logging.exception("policy snapshot on shutdown failed")

    app = FastAPI(title="sensable-portfolio", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz():
        return {
            "status": "ok",
            "uptime": time.time() - app.state.start_time if hasattr(app.state, "start_time") else 0.0,
            "connector_alive": getattr(app.state, "connector_alive", False),
            "decisions_total": app.state.decisions_total["n"] if hasattr(app.state, "decisions_total") else 0,
            "last_decision_ts": app.state.last_decision_ts["t"] if hasattr(app.state, "last_decision_ts") else 0.0,
            "ws_renderer_connected": app.state.renderer is not None if hasattr(app.state, "renderer") else False,
        }

    @app.post("/feedback")
    async def feedback(body: FeedbackBody):
        try:
            await record_feedback(app.state.engine, decision_id=body.decision_id,
                                  score=body.score, comment=body.comment, ts=time.time())
            return {"status": "ok"}
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/decide")
    async def decide():
        await app.state.graph.run_one()
        return {"status": "ok"}

    @app.get("/arms/leaderboard")
    async def leaderboard():
        out = []
        for a in app.state.arm_registry.active_arms():
            out.append({
                "id": a.id, "persona": a.persona, "model": a.model,
                "parent_id": a.parent_id,
                "pulls": 0, "mean_reward": 0.0, "last_pulled": None,
            })
        async with get_session(app.state.engine) as s:
            rows = (await s.exec(select(Decision.arm_id, Reward.reward).join(
                Reward, Reward.decision_id == Decision.id, isouter=True,
            ))).all()
        agg: dict[str, list[float]] = {}
        for arm_id, r in rows:
            if r is not None:
                agg.setdefault(arm_id, []).append(r)
        for row in out:
            r_list = agg.get(row["id"], [])
            row["pulls"] = len(r_list)
            row["mean_reward"] = sum(r_list) / len(r_list) if r_list else 0.0
        return out

    @app.get("/decisions/{decision_id}")
    async def get_decision(decision_id: str):
        async with get_session(app.state.engine) as s:
            d = (await s.exec(select(Decision).where(Decision.id == decision_id))).first()
            if d is None:
                raise HTTPException(404, "decision not found")
            r = (await s.exec(select(Reward).where(Reward.decision_id == decision_id))).first()
            f = (await s.exec(select(Feedback).where(Feedback.decision_id == decision_id))).first()
        return {
            "decision": d.model_dump(),
            "reward": r.model_dump() if r is not None else None,
            "feedback": f.model_dump() if f is not None else None,
        }

    if load_settings().debug_sse_enabled:
        from sse_starlette.sse import EventSourceResponse
        from sensable_portfolio.stream.sinks import sse_debug_stream

        @app.get("/debug/stream")
        async def debug_stream():
            return EventSourceResponse(sse_debug_stream(app.state.bus))

    return app


app = build_app()
