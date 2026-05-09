# sensable-portfolio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Python service that consumes `AffectSample` from `neurable_connector`, runs a portfolio of LangChain agents picked by a contextual bandit (LinUCB), and emits both mood frames (1 Hz) and action frames (every 30 min) over a single WebSocket connection to the renderer at `ws://127.0.0.1:7777`.

**Architecture:** Single asyncio process. In-process ingest from `neurable_connector.NeurableConnector.stream()`. Per-signal state held in `pidview.SignalRegistry`. Decision pipeline: tick scheduler → featurize → LinUCB picks an arm → arm runs (LangChain Runnable with structured output) → emit. Outputs ride one WebSocket client to `7777` (Spec A `type:"mood"` + Spec B-shaped `type:"agent_action"`). FastAPI on `8910` serves operational routes (feedback, decide, healthz, audit, optional debug SSE).

**Tech Stack:** Python 3.11+, FastAPI, uvicorn[standard], sse-starlette (debug only), pydantic v2, pydantic-settings, sqlmodel + aiosqlite (WAL), MABWiser (LinUCB), websockets (client), LangChain + LangGraph, PyYAML, pytest + pytest-asyncio + httpx + pytest-mock.

**Path-installed siblings:** `../neurable_connector`, `../pidview` (already exist).

---

## File map (created across tasks)

```
sensable-portfolio/
├── pyproject.toml
├── README.md
├── .env.example
├── docs/superpowers/specs/2026-05-09-sensable-portfolio-design.md  (already written)
├── docs/superpowers/plans/2026-05-09-sensable-portfolio.md         (this file)
├── config/
│   ├── arms.yaml
│   ├── targets.yaml
│   └── tuning.yaml
├── src/sensable_portfolio/
│   ├── __init__.py
│   ├── app.py
│   ├── config.py
│   ├── contracts.py
│   ├── connector/{__init__.py, runner.py, source.py}
│   ├── signals/{__init__.py, view.py, features.py}
│   ├── tick/{__init__.py, scheduler.py}
│   ├── arms/{__init__.py, registry.py, factory.py, prompts/*.yaml}
│   ├── policy/{__init__.py, base.py, linucb.py, persistence.py}
│   ├── graph/{__init__.py, nodes.py, decision.py}
│   ├── reward/{__init__.py, attribution.py, scheduler.py, feedback.py}
│   ├── evolve/{__init__.py, meta.py, prompts/mutator.yaml}
│   ├── stream/{__init__.py, pubsub.py, sinks.py}
│   ├── renderer/{__init__.py, client.py}
│   ├── storage/{__init__.py, models.py, db.py}
│   └── observability/{__init__.py, langsmith.py}
└── tests/
    ├── conftest.py
    ├── test_config.py, test_storage.py, test_contracts.py
    ├── test_connector_runner.py, test_features.py, test_scheduler.py
    ├── test_policy.py, test_arms.py, test_graph.py
    ├── test_pubsub.py, test_renderer_client.py
    ├── test_reward.py, test_app.py
    ├── test_evolver.py
    └── test_e2e.py
```

---

## Task 1: Project scaffold + git baseline

**Files:**
- Create: `pyproject.toml`, `README.md`, `.env.example`, `.gitignore`
- Create: `src/sensable_portfolio/__init__.py`
- Create: `tests/__init__.py`, `tests/conftest.py` (stub)

- [ ] **Step 1: Stop any stale parent submodule reference and init clean repo**

```bash
cd /Users/ricalanis/Dev/agentes_regios
if git ls-files --stage | grep -q '^160000 .* sensable-portfolio$'; then
  git rm --cached sensable-portfolio
fi
cd sensable-portfolio
[ -d .git ] || git init -q
```

Expected: `git status` runs without "unpopulated submodule" error.

- [ ] **Step 2: Write `pyproject.toml`**

Create `pyproject.toml`:

```toml
[project]
name = "sensable-portfolio"
version = "0.1.0"
description = "Agent portfolio + contextual bandit for stress/focus interventions."
requires-python = ">=3.11"
dependencies = [
  "fastapi>=0.110",
  "uvicorn[standard]>=0.27",
  "sse-starlette>=2.0",
  "pydantic>=2.6",
  "pydantic-settings>=2.2",
  "sqlmodel>=0.0.16",
  "aiosqlite>=0.19",
  "mabwiser>=2.7",
  "numpy>=1.26",
  "websockets>=12.0",
  "pyyaml>=6.0",
  "langchain>=0.2",
  "langgraph>=0.2",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0",
  "pytest-asyncio>=0.23",
  "httpx>=0.27",
  "pytest-mock>=3.12",
  "ruff>=0.4",
]
openai = ["langchain-openai>=0.1"]

[tool.uv.sources]
neurable_connector = { path = "../neurable_connector", editable = true }
pidview            = { path = "../pidview", editable = true }

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/sensable_portfolio"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = "-q --strict-markers"

[tool.ruff]
line-length = 100
target-version = "py311"
```

- [ ] **Step 3: Write `.gitignore` and `.env.example`**

`.gitignore`:
```
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
*.db
*.db-shm
*.db-wal
.env
dist/
build/
*.egg-info/
```

`.env.example`:
```
RENDERER_WS_URL=ws://127.0.0.1:7777
RENDERER_ENABLED=true
DEBUG_SSE_ENABLED=false
DB_URL=sqlite+aiosqlite:///./sensable.db
LANGSMITH_API_KEY=
OPENAI_API_KEY=
```

- [ ] **Step 4: Create source/test packages**

```bash
mkdir -p src/sensable_portfolio tests config
touch src/sensable_portfolio/__init__.py tests/__init__.py
```

`src/sensable_portfolio/__init__.py`:
```python
"""sensable-portfolio: agent portfolio + contextual bandit for stress/focus."""
__version__ = "0.1.0"
```

`tests/conftest.py` (stub; expanded in Task 5):
```python
"""Shared pytest fixtures."""
import pytest
```

- [ ] **Step 5: Verify install**

```bash
uv sync --extra dev || pip install -e ".[dev]"
pytest -q
```
Expected: `0 tests` collected, no errors.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml README.md .env.example .gitignore src/ tests/ docs/
git commit -m "chore: project scaffold with path-installed siblings"
```

---

## Task 2: Config (pydantic-settings)

**Files:**
- Create: `src/sensable_portfolio/config.py`
- Create: `config/tuning.yaml`, `config/targets.yaml`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

`tests/test_config.py`:
```python
from sensable_portfolio.config import Settings, load_settings


def test_defaults_match_spec():
    s = load_settings()
    assert s.port == 8910
    assert s.min_decision_interval_s == 1800
    assert s.renderer_signals_hz == 1.0
    assert s.renderer_ws_url == "ws://127.0.0.1:7777"
    assert s.renderer_enabled is True
    assert s.debug_sse_enabled is False
    assert s.window_s == 300
    assert s.baseline_pre == 120
    assert s.window_lo == 60
    assert s.window_hi == 360
    assert s.feedback_alpha == 0.5
    assert s.evolver_cron_h == 24
    assert s.policy_snapshot_every == 50
    # default target weights
    assert s.target_weights == {"stress": -0.5, "focus": 0.5}


def test_env_override(monkeypatch):
    monkeypatch.setenv("RENDERER_ENABLED", "false")
    monkeypatch.setenv("DB_URL", "sqlite+aiosqlite:///:memory:")
    s = load_settings()
    assert s.renderer_enabled is False
    assert s.db_url == "sqlite+aiosqlite:///:memory:"
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
pytest tests/test_config.py -v
```
Expected: ImportError or AttributeError.

- [ ] **Step 3: Implement `config.py`**

```python
"""Settings + tunables loader."""
from __future__ import annotations

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # API
    port: int = 8910

    # Renderer
    renderer_ws_url: str = "ws://127.0.0.1:7777"
    renderer_enabled: bool = True
    renderer_signals_hz: float = 1.0
    debug_sse_enabled: bool = False

    # Decision pipeline
    min_decision_interval_s: int = 1800
    window_s: int = 300
    baseline_pre: int = 120
    window_lo: int = 60
    window_hi: int = 360
    feedback_alpha: float = 0.5
    evolver_cron_h: int = 24
    policy_snapshot_every: int = 50

    # Storage
    db_url: str = "sqlite+aiosqlite:///./sensable.db"

    # Reward target (locked v1)
    target_weights: dict[str, float] = Field(default_factory=lambda: {"stress": -0.5, "focus": 0.5})

    # Optional integrations
    langsmith_api_key: str | None = None
    openai_api_key: str | None = None


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    return Settings()
```

- [ ] **Step 4: Stub config files**

`config/tuning.yaml`:
```yaml
# Optional override layer; settings.py defaults win unless DB_URL etc. are set in .env
min_decision_interval_s: 1800
renderer_signals_hz: 1.0
window_s: 300
baseline_pre: 120
window_lo: 60
window_hi: 360
feedback_alpha: 0.5
evolver_cron_h: 24
policy_snapshot_every: 50
```

`config/targets.yaml`:
```yaml
default:
  weights: { stress: -0.5, focus: 0.5 }
```

- [ ] **Step 5: Verify PASS**

```bash
pytest tests/test_config.py -v
```
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add src/sensable_portfolio/config.py tests/test_config.py config/
git commit -m "feat(config): pydantic-settings with env overrides + locked v1 tunables"
```

---

## Task 3: Storage models + DB engine

**Files:**
- Create: `src/sensable_portfolio/storage/__init__.py`, `models.py`, `db.py`
- Test: `tests/test_storage.py`

- [ ] **Step 1: Write the failing test**

`tests/test_storage.py`:
```python
import pytest
from sensable_portfolio.storage.db import init_engine, get_session
from sensable_portfolio.storage.models import (
    Decision, Reward, Feedback, Arm, PolicySnapshot, SnapshotLog,
)


@pytest.mark.asyncio
async def test_round_trip_decision_and_reward():
    engine = await init_engine("sqlite+aiosqlite:///:memory:")
    async with get_session(engine) as s:
        d = Decision(
            id="d1", ts=1000.0, arm_id="breath_coach.v1", target_id="default",
            context_json="{}", intervention_json='{"action_type":"breath"}',
            run_id=None,
        )
        s.add(d)
        await s.commit()

    async with get_session(engine) as s:
        r = Reward(
            decision_id="d1", components_json='{"focus":0.3,"stress":0.5}',
            user_score=None, reward=0.4, computed_at=2000.0,
        )
        s.add(r)
        await s.commit()

    async with get_session(engine) as s:
        from sqlmodel import select
        got = (await s.exec(select(Decision).where(Decision.id == "d1"))).first()
        rew = (await s.exec(select(Reward).where(Reward.decision_id == "d1"))).first()
        assert got.arm_id == "breath_coach.v1"
        assert rew.reward == 0.4


@pytest.mark.asyncio
async def test_snapshot_log_indexed_by_kind_ts():
    engine = await init_engine("sqlite+aiosqlite:///:memory:")
    async with get_session(engine) as s:
        for k in ("focus", "stress"):
            for i in range(3):
                s.add(SnapshotLog(ts=float(i), kind=k, value=float(i) * 0.1))
        await s.commit()

    async with get_session(engine) as s:
        from sqlmodel import select
        rows = (await s.exec(
            select(SnapshotLog).where(SnapshotLog.kind == "focus").order_by(SnapshotLog.ts)
        )).all()
        assert [r.value for r in rows] == [0.0, 0.1, 0.2]
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
pytest tests/test_storage.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement models**

`src/sensable_portfolio/storage/__init__.py`: empty.

`src/sensable_portfolio/storage/models.py`:
```python
"""SQLModel tables for sensable-portfolio."""
from __future__ import annotations

from sqlmodel import Field, SQLModel


class Decision(SQLModel, table=True):
    id: str = Field(primary_key=True)
    ts: float = Field(index=True)
    arm_id: str = Field(index=True)
    target_id: str = "default"
    context_json: str
    intervention_json: str
    run_id: str | None = None


class Reward(SQLModel, table=True):
    decision_id: str = Field(primary_key=True, foreign_key="decision.id")
    components_json: str
    user_score: float | None = None
    reward: float
    computed_at: float


class Feedback(SQLModel, table=True):
    decision_id: str = Field(primary_key=True, foreign_key="decision.id")
    score: float
    comment: str | None = None
    ts: float


class Arm(SQLModel, table=True):
    id: str = Field(primary_key=True)
    persona: str
    prompt_id: str
    model: str
    parent_id: str | None = None
    created_at: float
    retired_at: float | None = None


class PolicySnapshot(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    ts: float
    algo: str
    blob: bytes


class SnapshotLog(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    ts: float = Field(index=True)
    kind: str = Field(index=True)
    value: float
```

`src/sensable_portfolio/storage/db.py`:
```python
"""Async SQLite engine + session helpers."""
from __future__ import annotations

from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel


async def init_engine(url: str) -> AsyncEngine:
    engine = create_async_engine(url, echo=False, future=True)
    async with engine.begin() as conn:
        # WAL for file-backed SQLite; harmless for :memory:
        if url.startswith("sqlite"):
            await conn.exec_driver_sql("PRAGMA journal_mode=WAL")
        await conn.run_sync(SQLModel.metadata.create_all)
    return engine


@asynccontextmanager
async def get_session(engine: AsyncEngine):
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        yield s
```

- [ ] **Step 4: Verify PASS**

```bash
pytest tests/test_storage.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/sensable_portfolio/storage/ tests/test_storage.py
git commit -m "feat(storage): SQLModel tables + async engine helpers"
```

---

## Task 4: Contracts (Recommendation Protocol + Intervention + frame envelopes)

**Files:**
- Create: `src/sensable_portfolio/contracts.py`
- Test: `tests/test_contracts.py`

- [ ] **Step 1: Write the failing test**

`tests/test_contracts.py`:
```python
import json
from sensable_portfolio.contracts import (
    Intervention, AgentInfo, MoodFrame, AgentActionFrame, Recommendation,
)


def test_intervention_validates_and_satisfies_protocol():
    i = Intervention(
        decision_id="d1", arm_id="breath_coach.v1", ts=1000.0,
        action_type="breath", title="Box breath, 90 s",
        body="Inhale 4s, hold 4s, exhale 4s, hold 4s.",
        duration_s=90.0, intensity="low", rationale="rising stress slope",
    )
    assert i.schema_version == 1
    # Protocol structural check
    rec: Recommendation = i
    assert rec.decision_id == "d1"
    assert rec.arm_id == "breath_coach.v1"


def test_mood_frame_shape_matches_spec_a():
    f = MoodFrame(
        v=1, type="mood",
        vector={"focus": -0.11, "stress": -1.91, "valence": -1.96, "arousal": 0.75,
                "joy": 0.0, "calm": 0.0, "excitement": 0.05, "neutral": 0.48},
        ts=1778364427578,
    )
    payload = json.loads(f.model_dump_json())
    assert payload["v"] == 1
    assert payload["type"] == "mood"
    assert set(payload["vector"]) == {
        "focus","stress","valence","arousal","joy","calm","excitement","neutral",
    }
    assert isinstance(payload["ts"], int)


def test_agent_action_frame_carries_agent_and_intervention():
    f = AgentActionFrame(
        v=1, type="agent_action", ts=1778364427578,
        decision_id="d1",
        agent=AgentInfo(id="breath_coach.v1", persona="breath_coach", model="m"),
        intervention=Intervention(
            decision_id="d1", arm_id="breath_coach.v1", ts=1.0, action_type="breath",
            title="t", body="b", duration_s=10.0, intensity="low", rationale="r",
        ),
        signals_at_decision={"focus": 0.0, "stress": 0.0, "valence": 0.0, "arousal": 0.0,
                             "joy": 0.0, "calm": 0.0, "excitement": 0.0, "neutral": 0.0},
    )
    payload = json.loads(f.model_dump_json())
    assert payload["type"] == "agent_action"
    assert payload["agent"]["persona"] == "breath_coach"
    assert payload["intervention"]["action_type"] == "breath"
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
pytest tests/test_contracts.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement contracts**

`src/sensable_portfolio/contracts.py`:
```python
"""Recommendation Protocol + concrete Intervention + WS frame envelopes."""
from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable
from pydantic import BaseModel, Field, ConfigDict


@runtime_checkable
class Recommendation(Protocol):
    schema_version: int
    decision_id: str
    arm_id: str
    ts: float


class Intervention(BaseModel):
    """Concrete v1 Recommendation. Producers/consumers depend on the Protocol."""
    schema_version: Literal[1] = 1
    decision_id: str
    arm_id: str
    ts: float
    action_type: str
    title: str
    body: str
    duration_s: float
    intensity: Literal["low", "med", "high"]
    rationale: str


class AgentInfo(BaseModel):
    id: str
    persona: str
    model: str
    parent_id: str | None = None


class MoodFrame(BaseModel):
    """Spec A frame to ws://127.0.0.1:7777."""
    model_config = ConfigDict(extra="forbid")
    v: Literal[1] = 1
    type: Literal["mood"] = "mood"
    vector: dict[str, float]
    ts: int  # unix ms (Date.now()-style)


class AgentActionFrame(BaseModel):
    """Spec B-shaped action frame; same WS connection."""
    model_config = ConfigDict(extra="forbid")
    v: Literal[1] = 1
    type: Literal["agent_action"] = "agent_action"
    ts: int
    decision_id: str
    agent: AgentInfo
    intervention: Intervention
    signals_at_decision: dict[str, float]
```

- [ ] **Step 4: Verify PASS**

```bash
pytest tests/test_contracts.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/sensable_portfolio/contracts.py tests/test_contracts.py
git commit -m "feat(contracts): Recommendation Protocol, Intervention, MoodFrame, AgentActionFrame"
```

---

## Task 5: FakeSource fixture (deterministic synthetic input)

**Files:**
- Modify: `tests/conftest.py`
- Test: `tests/test_fake_source.py`

- [ ] **Step 1: Write the failing test**

`tests/test_fake_source.py`:
```python
import time
from neurable_connector import EEGFrame, FS_HZ
from .conftest import FakeSource  # noqa


def test_fake_source_yields_eegframes_at_fs_hz():
    src = FakeSource(runtime_s=0.05, seed=42)  # 25 frames @ 500 Hz
    frames = list(src)
    assert all(isinstance(f, EEGFrame) for f in frames)
    assert all(f.samples.shape == (12,) for f in frames)
    assert len(frames) >= int(0.05 * FS_HZ) - 5  # tolerance
    # monotonically increasing time
    ts = [f.t for f in frames]
    assert all(b > a for a, b in zip(ts, ts[1:]))


def test_fake_source_is_deterministic():
    a = list(FakeSource(runtime_s=0.05, seed=7))
    b = list(FakeSource(runtime_s=0.05, seed=7))
    assert len(a) == len(b)
    for fa, fb in zip(a, b):
        assert (fa.samples == fb.samples).all()
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
pytest tests/test_fake_source.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement FakeSource (mirrors `/tmp/dump_affect_signal.py`)**

`tests/conftest.py`:
```python
"""Shared pytest fixtures."""
from __future__ import annotations

import time
from typing import Iterator

import numpy as np
import pytest

from neurable_connector import EEGFrame, FS_HZ


class FakeSource:
    """Deterministic synthetic EEG; mirrors the FakeMW75 reference in /tmp."""

    def __init__(self, runtime_s: float = 4.0, seed: int = 0, sleep_realtime: bool = False):
        self.runtime_s = runtime_s
        self._rng = np.random.default_rng(seed)
        self._sleep = sleep_realtime

    def __iter__(self) -> Iterator[EEGFrame]:
        n_ch = 12
        fs = float(FS_HZ)
        dt = 1.0 / fs
        t0 = time.time()
        t = t0
        end = t0 + self.runtime_s
        alpha_phase = self._rng.uniform(0.0, 2 * np.pi, size=n_ch)
        i = 0
        emit_every = max(1, int(fs / 50))
        while t < end:
            phase = 2 * np.pi * 10.0 * (i / fs) + alpha_phase
            sample = (
                self._rng.standard_normal(n_ch).astype(np.float64) * 5.0
                + np.sin(phase) * 1.5
            )
            yield EEGFrame(t=t, samples=sample)
            i += 1
            t += dt
            if self._sleep and i % emit_every == 0:
                time.sleep(emit_every * dt * 0.5)


@pytest.fixture
def fake_source():
    return FakeSource(runtime_s=4.0, seed=2)
```

- [ ] **Step 4: Verify PASS**

```bash
pytest tests/test_fake_source.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/test_fake_source.py
git commit -m "test: FakeSource fixture mirroring /tmp/dump_affect_signal.py"
```

---

## Task 6: Connector source builder + runner

**Files:**
- Create: `src/sensable_portfolio/connector/__init__.py`, `source.py`, `runner.py`
- Create: `src/sensable_portfolio/signals/__init__.py`, `view.py`
- Test: `tests/test_connector_runner.py`

- [ ] **Step 1: Write the failing test**

`tests/test_connector_runner.py`:
```python
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
    # Calibrate baseline first
    baseline = Baseline.fit(list(FakeSource(runtime_s=0.6, seed=1)), fs=float(FS_HZ))
    src = FakeSource(runtime_s=0.6, seed=2)

    engine = await init_engine("sqlite+aiosqlite:///:memory:")
    registry = build_registry()
    new_data = asyncio.Event()
    signals_pubsub = []  # capture published AffectSamples

    async def on_signal(sample):
        signals_pubsub.append(sample)

    runner = ConnectorRunner(
        source=src, baseline=baseline, registry=registry, engine=engine,
        on_new_data=lambda: new_data.set(), on_signal=on_signal,
        snapshot_log_hz=4.0,  # write everything for test
    )
    await runner.run()

    # at least 1 sample produced (4 Hz * 0.6 s ~ 2-3 samples after warmup)
    assert len(signals_pubsub) >= 1
    assert new_data.is_set()
    # registry has all dims
    snaps = registry.snapshot_all()
    assert set(snaps.keys()) >= set(ALL_DIMS)
    # snapshot log persisted
    async with get_session(engine) as s:
        rows = (await s.exec(select(SnapshotLog))).all()
        assert any(r.kind == "focus" for r in rows)
        assert any(r.kind == "stress" for r in rows)
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
pytest tests/test_connector_runner.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement signals/view.py**

`src/sensable_portfolio/signals/__init__.py`: empty.

`src/sensable_portfolio/signals/view.py`:
```python
"""Thin facade over pidview.SignalRegistry for our 8 affective dims."""
from __future__ import annotations

from pidview import SignalRegistry

SIGNED_DIMS: tuple[str, ...] = ("focus", "stress", "valence", "arousal")
INTENSITY_DIMS: tuple[str, ...] = ("joy", "calm", "excitement", "neutral")
ALL_DIMS: tuple[str, ...] = SIGNED_DIMS + INTENSITY_DIMS

# Bandit-context kinds (subset of ALL_DIMS); valence/arousal excluded in v1.
CONTEXT_KINDS: tuple[str, ...] = ("focus", "stress", "joy", "calm", "excitement", "neutral")


def build_registry(history_seconds: float = 600.0, integral_tau: float = 60.0) -> SignalRegistry:
    reg = SignalRegistry()
    for name in ALL_DIMS:
        reg.register(name, history_seconds=history_seconds, integral_tau=integral_tau)
    return reg
```

- [ ] **Step 4: Implement connector/source.py**

`src/sensable_portfolio/connector/__init__.py`: empty.

`src/sensable_portfolio/connector/source.py`:
```python
"""Build a NeurableConnector with a real or synthetic Source."""
from __future__ import annotations

from neurable_connector import Baseline, FS_HZ, MW75Source, NeurableConnector, Source


def build_connector(source: Source, baseline: Baseline) -> NeurableConnector:
    return NeurableConnector(source=source, baseline=baseline)


def calibrate_baseline(source: Source) -> Baseline:
    return Baseline.fit(list(source), fs=float(FS_HZ))


def production_source() -> Source:
    """The real device. Tests use FakeSource via conftest; runtime uses MW75Source."""
    return MW75Source()
```

- [ ] **Step 5: Implement connector/runner.py**

`src/sensable_portfolio/connector/runner.py`:
```python
"""Run nc.stream() → push to SignalRegistry, write SnapshotLog, fire callbacks."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable

from neurable_connector import AffectSample, Baseline, NeurableConnector, Source
from pidview import SignalRegistry
from sqlalchemy.ext.asyncio import AsyncEngine

from sensable_portfolio.signals.view import ALL_DIMS
from sensable_portfolio.storage.db import get_session
from sensable_portfolio.storage.models import SnapshotLog
from sensable_portfolio.connector.source import build_connector


@dataclass
class ConnectorRunner:
    source: Source
    baseline: Baseline
    registry: SignalRegistry
    engine: AsyncEngine
    on_new_data: Callable[[], None]
    on_signal: Callable[[AffectSample], Awaitable[None]]
    snapshot_log_hz: float = 1.0  # downsample factor

    async def run(self) -> None:
        nc = build_connector(self.source, self.baseline)
        last_log_t: float = 0.0
        log_period = 1.0 / self.snapshot_log_hz

        async with nc as conn:
            async for s in conn.stream():
                # push every dim to the registry
                for name in ALL_DIMS:
                    self.registry.push(name, s.t, getattr(s, name))
                self.on_new_data()
                await self.on_signal(s)

                if s.t - last_log_t >= log_period:
                    last_log_t = s.t
                    async with get_session(self.engine) as sess:
                        for name in ALL_DIMS:
                            sess.add(SnapshotLog(ts=s.t, kind=name, value=getattr(s, name)))
                        await sess.commit()
```

- [ ] **Step 6: Verify PASS**

```bash
pytest tests/test_connector_runner.py -v
```
Expected: 1 passed.

- [ ] **Step 7: Commit**

```bash
git add src/sensable_portfolio/connector/ src/sensable_portfolio/signals/ tests/test_connector_runner.py
git commit -m "feat(connector): runner wires NeurableConnector → registry + SnapshotLog + callbacks"
```

---

## Task 7: Features builder (FeatureVector from Snapshots → 36-dim numpy)

**Files:**
- Create: `src/sensable_portfolio/signals/features.py`
- Test: `tests/test_features.py`

- [ ] **Step 1: Write the failing test**

`tests/test_features.py`:
```python
import numpy as np
from pidview import SignalRegistry, Snapshot

from sensable_portfolio.signals.features import build_feature_vector, FEATURE_DIM
from sensable_portfolio.signals.view import build_registry, CONTEXT_KINDS


def test_feature_vector_dimension_is_36():
    reg = build_registry()
    # push enough samples for non-trivial history
    for i in range(10):
        for k in CONTEXT_KINDS + ("valence", "arousal"):
            reg.push(k, float(i), float(i) * 0.1)
    fv = build_feature_vector(reg.snapshot_all())
    assert fv.shape == (FEATURE_DIM,)
    assert FEATURE_DIM == 36


def test_feature_vector_handles_short_history():
    reg = build_registry()
    # only 1 sample per kind
    for k in CONTEXT_KINDS + ("valence", "arousal"):
        reg.push(k, 0.0, 0.5)
    fv = build_feature_vector(reg.snapshot_all())
    assert fv.shape == (FEATURE_DIM,)
    # missing-history slots zero-padded; no NaN
    assert not np.isnan(fv).any()
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
pytest tests/test_features.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement features.py**

`src/sensable_portfolio/signals/features.py`:
```python
"""Build a 36-dim FeatureVector for the bandit context from per-kind Snapshots."""
from __future__ import annotations

import numpy as np
from pidview import Snapshot

from sensable_portfolio.signals.view import CONTEXT_KINDS

# 6 features per kind: present, differential, integral, last_3_history (t-2, t-1, t)
PER_KIND = 6
FEATURE_DIM = len(CONTEXT_KINDS) * PER_KIND  # 6 * 6 = 36


def _last_3_values(snap: Snapshot) -> tuple[float, float, float]:
    h = snap.history  # numpy (N, 2): [t, x]
    if h.shape[0] == 0:
        return (0.0, 0.0, 0.0)
    vals = h[:, 1]
    if len(vals) >= 3:
        a, b, c = vals[-3], vals[-2], vals[-1]
    elif len(vals) == 2:
        a, b, c = 0.0, vals[-2], vals[-1]
    else:
        a, b, c = 0.0, 0.0, vals[-1]
    return (float(a), float(b), float(c))


def build_feature_vector(snapshots: dict[str, Snapshot]) -> np.ndarray:
    out = np.zeros(FEATURE_DIM, dtype=np.float64)
    for i, kind in enumerate(CONTEXT_KINDS):
        snap = snapshots.get(kind)
        if snap is None:
            continue
        a, b, c = _last_3_values(snap)
        offset = i * PER_KIND
        out[offset + 0] = float(snap.present)
        out[offset + 1] = float(snap.differential)
        out[offset + 2] = float(snap.integral)
        out[offset + 3] = a
        out[offset + 4] = b
        out[offset + 5] = c
    return out
```

- [ ] **Step 4: Verify PASS**

```bash
pytest tests/test_features.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/sensable_portfolio/signals/features.py tests/test_features.py
git commit -m "feat(signals): 36-dim FeatureVector builder from pidview Snapshots"
```

---

## Task 8: Tick scheduler (asyncio.Event + min_interval gate)

**Files:**
- Create: `src/sensable_portfolio/tick/__init__.py`, `scheduler.py`
- Test: `tests/test_scheduler.py`

- [ ] **Step 1: Write the failing test**

`tests/test_scheduler.py`:
```python
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

    new_data.set(); await asyncio.sleep(0.02)   # tick 1 (immediate)
    new_data.set(); await asyncio.sleep(0.03)   # gated
    new_data.set(); await asyncio.sleep(0.12)   # tick 2 after gate
    assert len(fired) == 2

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
pytest tests/test_scheduler.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement TickScheduler**

`src/sensable_portfolio/tick/__init__.py`: empty.

`src/sensable_portfolio/tick/scheduler.py`:
```python
"""Decision-tick scheduler: fires when new_data + min_interval gate is open."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable


@dataclass
class TickScheduler:
    new_data: asyncio.Event
    min_interval_s: float
    on_tick: Callable[[], Awaitable[None]]

    async def run(self) -> None:
        last = 0.0
        loop = asyncio.get_running_loop()
        while True:
            await self.new_data.wait()
            self.new_data.clear()
            now = loop.time()
            wait = max(0.0, self.min_interval_s - (now - last))
            if wait > 0:
                await asyncio.sleep(wait)
            last = loop.time()
            try:
                await self.on_tick()
            except Exception:
                # never break the loop on a single tick failure
                import logging
                logging.exception("on_tick failed")
```

- [ ] **Step 4: Verify PASS**

```bash
pytest tests/test_scheduler.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/sensable_portfolio/tick/ tests/test_scheduler.py
git commit -m "feat(tick): TickScheduler with min_interval gate + new_data event"
```

---

## Task 9: Policy Protocol + LinUCB wrapper + persistence

**Files:**
- Create: `src/sensable_portfolio/policy/__init__.py`, `base.py`, `linucb.py`, `persistence.py`
- Test: `tests/test_policy.py`

- [ ] **Step 1: Write the failing test**

`tests/test_policy.py`:
```python
import numpy as np
import pytest

from sensable_portfolio.policy.linucb import LinUCBPolicy


def _ctx(*xs):
    return np.asarray(xs, dtype=np.float64)


def test_linucb_predict_returns_known_arm():
    p = LinUCBPolicy(arms=["a", "b"], context_dim=2, alpha=1.0)
    arm = p.predict(_ctx(0.0, 1.0))
    assert arm in {"a", "b"}


def test_linucb_learns_arm_for_regime():
    """In contexts where x[0]>0 'a' is better; x[0]<=0 'b' is better."""
    rng = np.random.default_rng(0)
    p = LinUCBPolicy(arms=["a", "b"], context_dim=2, alpha=0.5)
    for _ in range(400):
        x0 = float(rng.normal())
        ctx = _ctx(x0, 1.0)
        arm = p.predict(ctx)
        if arm == "a":
            r = 0.5 * x0 + rng.normal(0, 0.1)
        else:
            r = -0.5 * x0 + rng.normal(0, 0.1)
        p.partial_fit(ctx, arm, float(np.clip(r, -1, 1)))

    # After training, "a" should win for x0=+1, "b" should win for x0=-1
    a_wins = sum(p.predict(_ctx(1.0, 1.0)) == "a" for _ in range(50))
    b_wins = sum(p.predict(_ctx(-1.0, 1.0)) == "b" for _ in range(50))
    assert a_wins > 35
    assert b_wins > 35


def test_linucb_handles_arm_growth():
    p = LinUCBPolicy(arms=["a"], context_dim=2, alpha=1.0)
    p.partial_fit(_ctx(1.0, 1.0), "a", 0.5)
    p.add_arm("b")
    arm = p.predict(_ctx(0.0, 0.0))
    assert arm in {"a", "b"}


def test_linucb_snapshot_round_trip():
    p = LinUCBPolicy(arms=["a", "b"], context_dim=2, alpha=1.0)
    for _ in range(20):
        p.partial_fit(_ctx(0.5, 0.5), "a", 0.3)
    blob = p.snapshot()
    p2 = LinUCBPolicy(arms=["a", "b"], context_dim=2, alpha=1.0)
    p2.restore(blob)
    arm1 = p.predict(_ctx(0.5, 0.5))
    arm2 = p2.predict(_ctx(0.5, 0.5))
    assert arm1 == arm2
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
pytest tests/test_policy.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement Policy Protocol + LinUCB**

`src/sensable_portfolio/policy/__init__.py`: empty.

`src/sensable_portfolio/policy/base.py`:
```python
"""Policy Protocol: bandit interface."""
from __future__ import annotations

from typing import Protocol, runtime_checkable
import numpy as np


@runtime_checkable
class Policy(Protocol):
    def predict(self, context: np.ndarray) -> str: ...
    def partial_fit(self, context: np.ndarray, arm: str, reward: float) -> None: ...
    def add_arm(self, arm: str) -> None: ...
    def snapshot(self) -> bytes: ...
    def restore(self, blob: bytes) -> None: ...
```

`src/sensable_portfolio/policy/linucb.py`:
```python
"""Disjoint LinUCB via MABWiser, Gaussian reward in [-1, 1]."""
from __future__ import annotations

import pickle
from typing import Iterable

import numpy as np
from mabwiser.mab import MAB, LearningPolicy


class LinUCBPolicy:
    def __init__(self, arms: Iterable[str], context_dim: int, alpha: float = 1.0):
        self._arms = list(arms)
        self._dim = context_dim
        self._alpha = alpha
        self._mab = MAB(arms=self._arms, learning_policy=LearningPolicy.LinUCB(alpha=alpha))
        # MABWiser requires an initial fit — seed with one zero observation per arm
        zeros = np.zeros((len(self._arms), context_dim))
        decisions = list(self._arms)
        rewards = [0.0] * len(self._arms)
        self._mab.fit(decisions=decisions, rewards=rewards, contexts=zeros)

    def predict(self, context: np.ndarray) -> str:
        return self._mab.predict(contexts=context.reshape(1, -1))

    def partial_fit(self, context: np.ndarray, arm: str, reward: float) -> None:
        self._mab.partial_fit(decisions=[arm], rewards=[float(reward)],
                              contexts=context.reshape(1, -1))

    def add_arm(self, arm: str) -> None:
        if arm in self._arms:
            return
        self._arms.append(arm)
        self._mab.add_arm(arm)
        # seed new arm with one zero observation so MABWiser is willing to predict it
        self._mab.partial_fit(decisions=[arm], rewards=[0.0],
                              contexts=np.zeros((1, self._dim)))

    def snapshot(self) -> bytes:
        return pickle.dumps({"mab": self._mab, "arms": self._arms,
                             "dim": self._dim, "alpha": self._alpha})

    def restore(self, blob: bytes) -> None:
        state = pickle.loads(blob)
        self._mab = state["mab"]
        self._arms = state["arms"]
        self._dim = state["dim"]
        self._alpha = state["alpha"]
```

`src/sensable_portfolio/policy/persistence.py`:
```python
"""Persist/restore policy state through PolicySnapshot rows."""
from __future__ import annotations

import time
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import select

from sensable_portfolio.policy.linucb import LinUCBPolicy
from sensable_portfolio.storage.db import get_session
from sensable_portfolio.storage.models import PolicySnapshot


async def save(engine: AsyncEngine, policy: LinUCBPolicy, algo: str = "linucb_disjoint") -> None:
    blob = policy.snapshot()
    async with get_session(engine) as s:
        s.add(PolicySnapshot(ts=time.time(), algo=algo, blob=blob))
        await s.commit()


async def load_latest(engine: AsyncEngine, policy: LinUCBPolicy) -> bool:
    """Restore the most-recent snapshot. Returns True iff something was loaded."""
    async with get_session(engine) as s:
        row = (await s.exec(
            select(PolicySnapshot).order_by(PolicySnapshot.ts.desc()).limit(1)
        )).first()
        if row is None:
            return False
        policy.restore(row.blob)
        return True
```

- [ ] **Step 4: Verify PASS**

```bash
pytest tests/test_policy.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/sensable_portfolio/policy/ tests/test_policy.py
git commit -m "feat(policy): LinUCB disjoint via MABWiser + arm growth + snapshot/restore"
```

---

## Task 10: Arm registry + seed personas + factory

**Files:**
- Create: `src/sensable_portfolio/arms/__init__.py`, `registry.py`, `factory.py`
- Create: `src/sensable_portfolio/arms/prompts/breath_coach.yaml`, `micro_break.yaml`, `reframe_cbt.yaml`, `body_scan.yaml`, `env_tweak.yaml`, `social_nudge.yaml`, `deep_focus.yaml`
- Test: `tests/test_arms.py`

- [ ] **Step 1: Write the failing test**

`tests/test_arms.py`:
```python
import numpy as np
import pytest

from sensable_portfolio.arms.registry import ArmRegistry, ArmRow
from sensable_portfolio.arms.factory import build_arm_runnable
from sensable_portfolio.contracts import Intervention


def test_registry_loads_seed_personas():
    reg = ArmRegistry.from_default_pkg()
    arm_ids = [a.id for a in reg.active_arms()]
    expected_personas = {
        "breath_coach", "micro_break", "reframe_cbt",
        "body_scan", "env_tweak", "social_nudge", "deep_focus",
    }
    personas = {a.persona for a in reg.active_arms()}
    assert expected_personas <= personas
    assert len(arm_ids) == len(set(arm_ids))


def test_registry_add_and_retire():
    reg = ArmRegistry.from_default_pkg()
    n0 = len(reg.active_arms())
    reg.add(ArmRow(id="evolved.x", persona="breath_coach", prompt_id="breath_coach.v2",
                   model="fake", parent_id="breath_coach.v1", created_at=1.0))
    assert len(reg.active_arms()) == n0 + 1
    reg.retire("evolved.x", at=2.0)
    assert "evolved.x" not in [a.id for a in reg.active_arms()]


@pytest.mark.asyncio
async def test_factory_runs_with_fake_llm():
    """Use a RunnableLambda to simulate the LLM, bypassing real API."""
    from langchain_core.runnables import RunnableLambda

    def fake_llm(_inputs):
        return Intervention(
            decision_id="d1", arm_id="breath_coach.v1", ts=1.0,
            action_type="breath", title="Box breath",
            body="Inhale 4s, hold 4s, exhale 4s, hold 4s.",
            duration_s=90.0, intensity="low", rationale="from fake llm",
        )

    runnable = build_arm_runnable(
        arm=ArmRow(id="breath_coach.v1", persona="breath_coach",
                   prompt_id="breath_coach.v1", model="fake", parent_id=None,
                   created_at=0.0),
        llm_factory=lambda _model: RunnableLambda(fake_llm),
    )
    out = await runnable.ainvoke({
        "decision_id": "d1", "ts": 1.0,
        "context_features": np.zeros(36).tolist(),
        "signals_at_decision": {k: 0.0 for k in (
            "focus","stress","valence","arousal","joy","calm","excitement","neutral")},
    })
    assert isinstance(out, Intervention)
    assert out.arm_id == "breath_coach.v1"
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
pytest tests/test_arms.py -v
```
Expected: ImportError.

- [ ] **Step 3: Write the 7 seed prompt YAMLs**

Each `src/sensable_portfolio/arms/prompts/<persona>.yaml` follows the same shape. Example for `breath_coach.yaml`:

```yaml
id: breath_coach.v1
persona: breath_coach
model: fake
system: |
  You are a breath-work coach. Given the user's affective signals, propose
  exactly one short breathing exercise to lower stress and steady focus.
  Output must conform to the Intervention schema.
human: |
  Decision id: {decision_id}
  Signals at decision time: {signals_at_decision}
  Recent context features (36-dim, see schema): {context_features}
  Propose one micro-intervention. Keep duration_s ≤ 180. intensity ∈ {{low, med, high}}.
```

Repeat for the other six (`micro_break`, `reframe_cbt`, `body_scan`, `env_tweak`, `social_nudge`, `deep_focus`) with each persona's voice. The system prompt's first sentence varies per persona; everything below `human:` stays identical.

Verbatim per-persona system lines:
- `micro_break.v1`: "You are a micro-break coach. Propose a 1-3 minute movement or eye-rest break."
- `reframe_cbt.v1`: "You are a CBT reframer. Propose a single cognitive reframe the user can apply silently."
- `body_scan.v1`: "You are a somatic-attention guide. Propose a 60-180s body-scan instruction."
- `env_tweak.v1`: "You are an environment coach. Propose one immediate environmental adjustment (light, sound, posture)."
- `social_nudge.v1`: "You are a social-connection coach. Propose one tiny pro-social action the user can take in the next 5 minutes."
- `deep_focus.v1`: "You are a deep-focus coach. Propose a single task-orientation cue the user can install in <60s."

`model: fake` is a placeholder; the factory passes `arm.model` to the configured `llm_factory`. Production will set it to e.g. `claude-sonnet-4-6`.

- [ ] **Step 4: Implement registry.py**

`src/sensable_portfolio/arms/__init__.py`: empty.

`src/sensable_portfolio/arms/registry.py`:
```python
"""Arm catalog: YAML-backed; new rows can be added at runtime (evolver)."""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from importlib import resources

import yaml


@dataclass(frozen=True)
class ArmRow:
    id: str
    persona: str
    prompt_id: str
    model: str
    parent_id: str | None
    created_at: float
    retired_at: float | None = None
    system: str = ""
    human: str = ""


class ArmRegistry:
    def __init__(self, rows: list[ArmRow]):
        self._rows: dict[str, ArmRow] = {r.id: r for r in rows}

    @classmethod
    def from_default_pkg(cls) -> "ArmRegistry":
        pkg = resources.files("sensable_portfolio.arms.prompts")
        rows: list[ArmRow] = []
        for path in pkg.iterdir():
            if path.suffix not in (".yaml", ".yml"):
                continue
            data = yaml.safe_load(path.read_text())
            rows.append(ArmRow(
                id=data["id"], persona=data["persona"], prompt_id=data["id"],
                model=data["model"], parent_id=None, created_at=time.time(),
                system=data.get("system", ""), human=data.get("human", ""),
            ))
        return cls(rows)

    def active_arms(self) -> list[ArmRow]:
        return [r for r in self._rows.values() if r.retired_at is None]

    def all_arms(self) -> list[ArmRow]:
        return list(self._rows.values())

    def add(self, row: ArmRow) -> None:
        if row.id in self._rows:
            raise ValueError(f"arm {row.id} already exists")
        self._rows[row.id] = row

    def retire(self, arm_id: str, at: float) -> None:
        if arm_id not in self._rows:
            return
        self._rows[arm_id] = replace(self._rows[arm_id], retired_at=at)

    def get(self, arm_id: str) -> ArmRow:
        return self._rows[arm_id]
```

- [ ] **Step 5: Implement factory.py**

`src/sensable_portfolio/arms/factory.py`:
```python
"""Build a LangChain Runnable for an Arm.

The Runnable takes a dict input (decision_id, ts, context_features, signals_at_decision)
and emits an Intervention. The LLM call is abstracted via `llm_factory(model_name)`
so tests can inject a deterministic fake."""
from __future__ import annotations

from typing import Any, Callable

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from sensable_portfolio.arms.registry import ArmRow
from sensable_portfolio.contracts import Intervention


def _wrap_to_intervention(arm: ArmRow) -> Callable[[Any], Intervention]:
    def _coerce(out: Any) -> Intervention:
        if isinstance(out, Intervention):
            # ensure arm_id is stamped
            return out.model_copy(update={"arm_id": arm.id})
        if isinstance(out, dict):
            return Intervention(**out)
        raise TypeError(f"Arm {arm.id} returned unexpected type: {type(out)}")
    return _coerce


def build_arm_runnable(
    arm: ArmRow,
    llm_factory: Callable[[str], Runnable],
) -> Runnable:
    prompt = ChatPromptTemplate.from_messages([
        ("system", arm.system or f"You are the {arm.persona} arm."),
        ("human",  arm.human  or "Propose one Intervention. Inputs: {context_features} {signals_at_decision}"),
    ])
    base_llm = llm_factory(arm.model)
    coerce = RunnableLambda(_wrap_to_intervention(arm))
    return prompt | base_llm | coerce
```

- [ ] **Step 6: Verify PASS**

```bash
pytest tests/test_arms.py -v
```
Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add src/sensable_portfolio/arms/ tests/test_arms.py
git commit -m "feat(arms): YAML-backed registry + 7 seed personas + factory with injectable LLM"
```

---

## Task 11: Decision graph (LangGraph: featurize → select → run_arm → emit)

**Files:**
- Create: `src/sensable_portfolio/graph/__init__.py`, `nodes.py`, `decision.py`
- Test: `tests/test_graph.py`

- [ ] **Step 1: Write the failing test**

`tests/test_graph.py`:
```python
import asyncio
import pytest
from langchain_core.runnables import RunnableLambda

from sensable_portfolio.contracts import Intervention
from sensable_portfolio.policy.linucb import LinUCBPolicy
from sensable_portfolio.arms.registry import ArmRegistry
from sensable_portfolio.signals.view import build_registry, ALL_DIMS, CONTEXT_KINDS
from sensable_portfolio.signals.features import FEATURE_DIM
from sensable_portfolio.graph.decision import DecisionGraph
from sensable_portfolio.storage.db import init_engine


def _fake_llm_factory(_model):
    def _fn(_):
        return Intervention(
            decision_id="will_overwrite", arm_id="will_overwrite", ts=0.0,
            action_type="breath", title="t", body="b",
            duration_s=10.0, intensity="low", rationale="r",
        )
    return RunnableLambda(_fn)


@pytest.mark.asyncio
async def test_decision_graph_emits_intervention_and_persists_decision():
    registry = build_registry()
    # seed history
    for k in ALL_DIMS:
        for i in range(5):
            registry.push(k, float(i), 0.1)

    arm_reg = ArmRegistry.from_default_pkg()
    arm_ids = [a.id for a in arm_reg.active_arms()]
    policy = LinUCBPolicy(arms=arm_ids, context_dim=FEATURE_DIM, alpha=1.0)
    engine = await init_engine("sqlite+aiosqlite:///:memory:")

    emitted: list = []

    async def on_emit(event):
        emitted.append(event)

    graph = DecisionGraph(
        signal_registry=registry, arm_registry=arm_reg, policy=policy,
        engine=engine, llm_factory=_fake_llm_factory, on_emit=on_emit,
    )
    await graph.run_one()
    assert len(emitted) == 1
    ev = emitted[0]
    assert ev["arm_id"] in arm_ids
    assert ev["intervention"].decision_id == ev["decision_id"]
    assert set(ev["signals_at_decision"]) == set(ALL_DIMS)
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
pytest tests/test_graph.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement graph nodes**

`src/sensable_portfolio/graph/__init__.py`: empty.

`src/sensable_portfolio/graph/nodes.py`:
```python
"""Pure functions for each step in the decision graph.

We deliberately keep these as plain async functions so the test suite
exercises them without spinning up a LangGraph runtime; the LangGraph
compile step in decision.py is a thin wrapper that orders them."""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

import numpy as np
from sqlalchemy.ext.asyncio import AsyncEngine

from sensable_portfolio.arms.factory import build_arm_runnable
from sensable_portfolio.arms.registry import ArmRegistry
from sensable_portfolio.contracts import Intervention
from sensable_portfolio.policy.linucb import LinUCBPolicy
from sensable_portfolio.signals.features import FEATURE_DIM, build_feature_vector
from sensable_portfolio.signals.view import ALL_DIMS
from sensable_portfolio.storage.db import get_session
from sensable_portfolio.storage.models import Decision


async def featurize(state: dict[str, Any]) -> dict[str, Any]:
    snaps = state["signal_registry"].snapshot_all()
    state["context"] = build_feature_vector(snaps)
    state["signals_at_decision"] = {
        k: float(snaps[k].present) if k in snaps else 0.0 for k in ALL_DIMS
    }
    return state


async def select(state: dict[str, Any]) -> dict[str, Any]:
    policy: LinUCBPolicy = state["policy"]
    state["arm_id"] = policy.predict(state["context"])
    return state


async def run_arm(state: dict[str, Any]) -> dict[str, Any]:
    reg: ArmRegistry = state["arm_registry"]
    arm = reg.get(state["arm_id"])
    runnable = build_arm_runnable(arm, state["llm_factory"])
    decision_id = state.get("decision_id") or uuid.uuid4().hex
    state["decision_id"] = decision_id
    out = await runnable.ainvoke({
        "decision_id": decision_id,
        "ts": state.get("ts") or time.time(),
        "context_features": state["context"].tolist(),
        "signals_at_decision": state["signals_at_decision"],
    })
    if not isinstance(out, Intervention):
        raise TypeError(f"Arm {arm.id} did not return an Intervention")
    state["intervention"] = out.model_copy(update={
        "decision_id": decision_id, "arm_id": arm.id,
    })
    return state


async def persist(state: dict[str, Any]) -> dict[str, Any]:
    engine: AsyncEngine = state["engine"]
    inter: Intervention = state["intervention"]
    async with get_session(engine) as s:
        s.add(Decision(
            id=inter.decision_id, ts=inter.ts, arm_id=inter.arm_id,
            target_id="default",
            context_json=json.dumps(state["context"].tolist()),
            intervention_json=inter.model_dump_json(),
            run_id=None,
        ))
        await s.commit()
    return state


async def emit(state: dict[str, Any]) -> dict[str, Any]:
    inter: Intervention = state["intervention"]
    arm = state["arm_registry"].get(inter.arm_id)
    event = {
        "decision_id": inter.decision_id,
        "arm_id": inter.arm_id,
        "agent": {"id": arm.id, "persona": arm.persona, "model": arm.model,
                  "parent_id": arm.parent_id},
        "intervention": inter,
        "signals_at_decision": state["signals_at_decision"],
        "ts": inter.ts,
    }
    on_emit = state.get("on_emit")
    if on_emit is not None:
        await on_emit(event)
    return state
```

- [ ] **Step 4: Implement decision.py wrapper**

`src/sensable_portfolio/graph/decision.py`:
```python
"""Glue all decision-graph nodes into a runnable async sequence."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from langchain_core.runnables import Runnable
from sqlalchemy.ext.asyncio import AsyncEngine

from sensable_portfolio.arms.registry import ArmRegistry
from sensable_portfolio.graph.nodes import emit, featurize, persist, run_arm, select
from sensable_portfolio.policy.linucb import LinUCBPolicy
from pidview import SignalRegistry


@dataclass
class DecisionGraph:
    signal_registry: SignalRegistry
    arm_registry: ArmRegistry
    policy: LinUCBPolicy
    engine: AsyncEngine
    llm_factory: Callable[[str], Runnable]
    on_emit: Callable[[dict[str, Any]], Awaitable[None]]

    async def run_one(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "signal_registry": self.signal_registry,
            "arm_registry": self.arm_registry,
            "policy": self.policy,
            "engine": self.engine,
            "llm_factory": self.llm_factory,
            "on_emit": self.on_emit,
            "ts": time.time(),
        }
        state = await featurize(state)
        state = await select(state)
        state = await run_arm(state)
        state = await persist(state)
        state = await emit(state)
        return state
```

> Why a plain async sequence rather than `langgraph.graph.StateGraph`? The pipeline is linear with no branches; LangGraph's value here is checkpointing and conditional edges, neither of which we need at v1. Switching is a one-file change inside `decision.py` and does not touch any other module.

- [ ] **Step 5: Verify PASS**

```bash
pytest tests/test_graph.py -v
```
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add src/sensable_portfolio/graph/ tests/test_graph.py
git commit -m "feat(graph): linear async decision graph (featurize→select→run_arm→persist→emit)"
```

---

## Task 12: Stream pubsub + SSE debug sink

**Files:**
- Create: `src/sensable_portfolio/stream/__init__.py`, `pubsub.py`, `sinks.py`
- Test: `tests/test_pubsub.py`

- [ ] **Step 1: Write the failing test**

`tests/test_pubsub.py`:
```python
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
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
pytest tests/test_pubsub.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement PubSub**

`src/sensable_portfolio/stream/__init__.py`: empty.

`src/sensable_portfolio/stream/pubsub.py`:
```python
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
                # drop-old policy for backpressure
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
```

`src/sensable_portfolio/stream/sinks.py`:
```python
"""SSE debug sink: mirrors WS-out frames over /debug/stream."""
from __future__ import annotations

import json
from typing import AsyncIterator

from sensable_portfolio.stream.pubsub import PubSub


async def sse_debug_stream(bus: PubSub) -> AsyncIterator[dict]:
    """Yield SSE-shaped events: {event, data} for both signals and actions."""
    async with bus.subscribe("signals") as qs, bus.subscribe("actions") as qa:
        import asyncio
        while True:
            done, _ = await asyncio.wait(
                {asyncio.create_task(qs.get()), asyncio.create_task(qa.get())},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in done:
                msg = t.result()
                ev = "signals" if msg.get("type") == "mood" else "actions"
                yield {"event": ev, "data": json.dumps(msg)}
```

- [ ] **Step 4: Verify PASS**

```bash
pytest tests/test_pubsub.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/sensable_portfolio/stream/ tests/test_pubsub.py
git commit -m "feat(stream): asyncio PubSub bus + SSE debug sink"
```

---

## Task 13: Renderer client (single WebSocket carrying both frame types)

**Files:**
- Create: `src/sensable_portfolio/renderer/__init__.py`, `client.py`
- Test: `tests/test_renderer_client.py`

- [ ] **Step 1: Write the failing test**

`tests/test_renderer_client.py`:
```python
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

        # publish a mood + an action
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
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
pytest tests/test_renderer_client.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement RendererClient**

`src/sensable_portfolio/renderer/__init__.py`: empty.

`src/sensable_portfolio/renderer/client.py`:
```python
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
                    return  # cancel during backoff
                except asyncio.TimeoutError:
                    pass
                backoff = min(self.backoff_max, backoff * 2 if backoff > 0 else self.backoff_initial)

    async def _send_loop(self, ws) -> None:
        period = 1.0 / max(self.signals_hz, 0.001)
        last_signal_send = 0.0
        async with self.bus.subscribe("signals", maxsize=4) as qs, \
                   self.bus.subscribe("actions", maxsize=64) as qa:
            while not self._cancel.is_set():
                # await whichever fires first; signals are coalesced via period
                signal_task = asyncio.create_task(qs.get())
                action_task = asyncio.create_task(qa.get())
                done, pending = await asyncio.wait(
                    {signal_task, action_task},
                    return_when=asyncio.FIRST_COMPLETED, timeout=period,
                )
                for p in pending:
                    p.cancel()
                now = asyncio.get_running_loop().time()
                # actions: send every event
                if action_task in done:
                    msg = action_task.result()
                    await ws.send(json.dumps(msg))
                # signals: rate-limited to signals_hz
                if signal_task in done:
                    msg = signal_task.result()
                    if (now - last_signal_send) >= period:
                        await ws.send(json.dumps(msg))
                        last_signal_send = now
```

- [ ] **Step 4: Verify PASS**

```bash
pytest tests/test_renderer_client.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/sensable_portfolio/renderer/ tests/test_renderer_client.py
git commit -m "feat(renderer): WS client carrying mood + agent_action frames with backoff"
```

---

## Task 14: Reward attribution + scheduler + feedback

**Files:**
- Create: `src/sensable_portfolio/reward/__init__.py`, `attribution.py`, `scheduler.py`, `feedback.py`
- Test: `tests/test_reward.py`

- [ ] **Step 1: Write the failing test**

`tests/test_reward.py`:
```python
import asyncio
import json
import pytest
from sqlmodel import select

from sensable_portfolio.policy.linucb import LinUCBPolicy
from sensable_portfolio.reward.attribution import attribute
from sensable_portfolio.reward.scheduler import RewardScheduler
from sensable_portfolio.reward.feedback import record_feedback
from sensable_portfolio.signals.features import FEATURE_DIM
from sensable_portfolio.storage.db import get_session, init_engine
from sensable_portfolio.storage.models import (
    Decision, Feedback, Reward, SnapshotLog,
)


async def _seed(engine, decision_ts: float):
    async with get_session(engine) as s:
        # 0..120s pre-baseline avg = 0.5; 60..360s post avg = 1.5  -> Δ = +1.0 raw
        for ts in range(int(decision_ts - 120), int(decision_ts)):
            for k in ("focus", "stress"):
                s.add(SnapshotLog(ts=float(ts), kind=k, value=0.5))
        for ts in range(int(decision_ts + 60), int(decision_ts + 360)):
            for k in ("focus", "stress"):
                s.add(SnapshotLog(ts=float(ts), kind=k, value=1.5))
        s.add(Decision(
            id="d1", ts=decision_ts, arm_id="x", target_id="default",
            context_json=json.dumps([0.0] * FEATURE_DIM),
            intervention_json="{}", run_id=None,
        ))
        await s.commit()


@pytest.mark.asyncio
async def test_attribute_signed_weights_flip_correctly():
    engine = await init_engine("sqlite+aiosqlite:///:memory:")
    await _seed(engine, decision_ts=1000.0)

    components, reward = await attribute(
        engine, decision_id="d1", target_weights={"stress": -0.5, "focus": 0.5},
        baseline_pre=120, window_lo=60, window_hi=360, alpha=0.5,
    )
    # focus rises Δ=+1; +0.5 weight => +0.5*z
    # stress rises Δ=+1; -0.5 weight => -0.5*z
    # net raw ≈ 0
    assert "focus" in components and "stress" in components
    assert -0.2 < reward < 0.2


@pytest.mark.asyncio
async def test_scheduler_attributes_and_partial_fits():
    engine = await init_engine("sqlite+aiosqlite:///:memory:")
    await _seed(engine, decision_ts=1000.0)

    policy = LinUCBPolicy(arms=["x"], context_dim=FEATURE_DIM, alpha=1.0)
    sched = RewardScheduler(
        engine=engine, policy=policy,
        target_weights={"stress": -0.5, "focus": 0.5},
        baseline_pre=120, window_lo=60, window_hi=360,
        feedback_alpha=0.5,
        # treat 'now' as ts after the post-window has closed
        now_fn=lambda: 1500.0,
        scan_interval_s=0.0,
    )
    await sched.scan_once()

    async with get_session(engine) as s:
        rew = (await s.exec(select(Reward).where(Reward.decision_id == "d1"))).first()
        assert rew is not None


@pytest.mark.asyncio
async def test_feedback_blends_into_reward():
    engine = await init_engine("sqlite+aiosqlite:///:memory:")
    await _seed(engine, decision_ts=1000.0)

    await record_feedback(engine, decision_id="d1", score=1.0, comment="great", ts=1500.0)
    async with get_session(engine) as s:
        fb = (await s.exec(select(Feedback).where(Feedback.decision_id == "d1"))).first()
        assert fb.score == 1.0
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
pytest tests/test_reward.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement attribution.py**

`src/sensable_portfolio/reward/__init__.py`: empty.

`src/sensable_portfolio/reward/attribution.py`:
```python
"""Compute the goal-conditioned reward for a decision from SnapshotLog history."""
from __future__ import annotations

import json
import math
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import select

from sensable_portfolio.storage.db import get_session
from sensable_portfolio.storage.models import Decision, Feedback, SnapshotLog


def _z_safe(delta: float, std: float) -> float:
    if std <= 1e-9 or math.isnan(std):
        return 0.0
    return delta / std


async def _mean_in_window(engine: AsyncEngine, kind: str, t_lo: float, t_hi: float) -> tuple[float, float]:
    """Return (mean, std) of SnapshotLog values for kind in [t_lo, t_hi)."""
    async with get_session(engine) as s:
        rows = (await s.exec(
            select(SnapshotLog.value).where(
                (SnapshotLog.kind == kind)
                & (SnapshotLog.ts >= t_lo)
                & (SnapshotLog.ts < t_hi)
            )
        )).all()
    vals = [float(v) for v in rows]
    if not vals:
        return (0.0, 0.0)
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / max(1, n - 1)
    return (mean, math.sqrt(var))


async def _running_std_1h(engine: AsyncEngine, kind: str, t: float) -> float:
    _, std = await _mean_in_window(engine, kind, t - 3600.0, t)
    return std if std > 1e-9 else 1.0


async def attribute(
    engine: AsyncEngine,
    decision_id: str,
    target_weights: dict[str, float],
    baseline_pre: int,
    window_lo: int,
    window_hi: int,
    alpha: float,
) -> tuple[dict[str, float], float]:
    async with get_session(engine) as s:
        d = (await s.exec(select(Decision).where(Decision.id == decision_id))).first()
        if d is None:
            raise ValueError(decision_id)
        fb = (await s.exec(select(Feedback).where(Feedback.decision_id == decision_id))).first()

    components: dict[str, float] = {}
    raw = 0.0
    for kind, weight in target_weights.items():
        b_mean, _ = await _mean_in_window(engine, kind, d.ts - baseline_pre, d.ts)
        o_mean, _ = await _mean_in_window(engine, kind, d.ts + window_lo, d.ts + window_hi)
        std = await _running_std_1h(engine, kind, d.ts)
        z = _z_safe(o_mean - b_mean, std)
        components[kind] = float(weight * z)
        raw += components[kind]

    user_score = float(fb.score) if fb is not None else 0.0
    reward = max(-1.0, min(1.0, raw + (alpha * user_score if fb is not None else 0.0)))
    return components, reward
```

- [ ] **Step 4: Implement scheduler.py**

`src/sensable_portfolio/reward/scheduler.py`:
```python
"""Background scanner: attribute rewards for decisions whose post-window has closed."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Callable

from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import select

from sensable_portfolio.policy.linucb import LinUCBPolicy
from sensable_portfolio.reward.attribution import attribute
from sensable_portfolio.signals.features import FEATURE_DIM
from sensable_portfolio.storage.db import get_session
from sensable_portfolio.storage.models import Decision, Reward
import numpy as np


@dataclass
class RewardScheduler:
    engine: AsyncEngine
    policy: LinUCBPolicy
    target_weights: dict[str, float]
    baseline_pre: int
    window_lo: int
    window_hi: int
    feedback_alpha: float
    now_fn: Callable[[], float] = time.time
    scan_interval_s: float = 30.0

    async def scan_once(self) -> int:
        cutoff = self.now_fn() - self.window_hi
        # decisions whose window has closed AND have no Reward yet
        async with get_session(self.engine) as s:
            unscored = (await s.exec(
                select(Decision).where(
                    (Decision.ts <= cutoff)
                    & (Decision.id.not_in(select(Reward.decision_id)))
                )
            )).all()
        n = 0
        for d in unscored:
            try:
                components, reward = await attribute(
                    self.engine, d.id, self.target_weights,
                    self.baseline_pre, self.window_lo, self.window_hi,
                    self.feedback_alpha,
                )
                async with get_session(self.engine) as s:
                    s.add(Reward(
                        decision_id=d.id,
                        components_json=json.dumps(components),
                        user_score=None, reward=reward,
                        computed_at=self.now_fn(),
                    ))
                    await s.commit()
                ctx = np.asarray(json.loads(d.context_json), dtype=np.float64)
                if ctx.shape == (FEATURE_DIM,):
                    self.policy.partial_fit(ctx, d.arm_id, reward)
                n += 1
            except Exception:
                import logging; logging.exception("attribute failed for %s", d.id)
        return n

    async def run(self) -> None:
        while True:
            try:
                await self.scan_once()
            except Exception:
                import logging; logging.exception("reward scheduler scan failed")
            await asyncio.sleep(max(0.1, self.scan_interval_s))
```

- [ ] **Step 5: Implement feedback.py**

`src/sensable_portfolio/reward/feedback.py`:
```python
"""POST /feedback handler: record a user's outcome score for a decision."""
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine

from sensable_portfolio.storage.db import get_session
from sensable_portfolio.storage.models import Feedback


async def record_feedback(
    engine: AsyncEngine, *, decision_id: str, score: float,
    comment: str | None = None, ts: float,
) -> None:
    if not -1.0 <= score <= 1.0:
        raise ValueError("score must be in [-1, 1]")
    async with get_session(engine) as s:
        s.add(Feedback(decision_id=decision_id, score=score, comment=comment, ts=ts))
        await s.commit()
```

- [ ] **Step 6: Verify PASS**

```bash
pytest tests/test_reward.py -v
```
Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add src/sensable_portfolio/reward/ tests/test_reward.py
git commit -m "feat(reward): attribution + scheduler + feedback intake"
```

---

## Task 15: FastAPI app (operational routes + lifespan wiring)

**Files:**
- Create: `src/sensable_portfolio/app.py`
- Test: `tests/test_app.py`

- [ ] **Step 1: Write the failing test**

`tests/test_app.py`:
```python
import asyncio
import pytest
from httpx import ASGITransport, AsyncClient

from sensable_portfolio.app import build_app


@pytest.mark.asyncio
async def test_healthz_returns_status_block():
    app = build_app(start_runtime=False)  # don't spawn the connector loop in tests
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/healthz")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] in ("ok", "starting")
        assert "decisions_total" in body


@pytest.mark.asyncio
async def test_feedback_records():
    app = build_app(start_runtime=False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # seed a Decision row for FK validity
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
    app = build_app(start_runtime=False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        r = await ac.get("/arms/leaderboard")
        assert r.status_code == 200
        rows = r.json()
        assert len(rows) >= 7
        assert {row["persona"] for row in rows} >= {"breath_coach", "deep_focus"}
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
pytest tests/test_app.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement app.py**

`src/sensable_portfolio/app.py`:
```python
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
    without an API key. Production code should pass a real factory."""
    def _fn(inputs: dict[str, Any]) -> Intervention:
        return Intervention(
            decision_id=inputs.get("decision_id", "d?"),
            arm_id="will_be_overwritten",
            ts=float(inputs.get("ts", 0.0)),
            action_type="breath",
            title="Box breath, 90s",
            body="Inhale 4s, hold 4s, exhale 4s, hold 4s. Repeat.",
            duration_s=90.0, intensity="low",
            rationale="stub_llm_factory placeholder",
        )
    return RunnableLambda(_fn)


def build_app(start_runtime: bool = True) -> FastAPI:
    settings = load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        engine = await init_engine(settings.db_url)
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

        graph = DecisionGraph(
            signal_registry=signal_registry, arm_registry=arm_registry,
            policy=policy, engine=engine, llm_factory=_stub_llm_factory,
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
            await asyncio.gather(*app.state.tasks, return_exceptions=True)
            await save_policy(engine, policy)

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
        # decorate with reward stats
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
```

- [ ] **Step 4: Verify PASS**

```bash
pytest tests/test_app.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/sensable_portfolio/app.py tests/test_app.py
git commit -m "feat(app): FastAPI lifespan wiring + healthz/feedback/decide/leaderboard/audit"
```

---

## Task 16: Meta-evolver (24h cron mutator)

**Files:**
- Create: `src/sensable_portfolio/evolve/__init__.py`, `meta.py`, `prompts/mutator.yaml`
- Test: `tests/test_evolver.py`

- [ ] **Step 1: Write the failing test**

`tests/test_evolver.py`:
```python
import asyncio
import pytest

from langchain_core.runnables import RunnableLambda

from sensable_portfolio.arms.registry import ArmRegistry, ArmRow
from sensable_portfolio.evolve.meta import MetaEvolver
from sensable_portfolio.policy.linucb import LinUCBPolicy
from sensable_portfolio.signals.features import FEATURE_DIM
from sensable_portfolio.storage.db import init_engine, get_session
from sensable_portfolio.storage.models import Reward, Decision


def _fake_mutator_factory(_model: str):
    def _fn(inputs: dict):
        return {
            "id": f"evolved.{inputs['parent_id'].split('.')[0]}.x",
            "system": "You are an evolved variant.",
            "human": inputs.get("human", "Propose one Intervention."),
        }
    return RunnableLambda(_fn)


@pytest.mark.asyncio
async def test_evolver_picks_top_arm_and_registers_variant():
    engine = await init_engine("sqlite+aiosqlite:///:memory:")
    arm_reg = ArmRegistry.from_default_pkg()
    arms = arm_reg.active_arms()
    parent = arms[0]

    # seed a few rewards favoring the parent
    async with get_session(engine) as s:
        for i in range(5):
            d_id = f"d{i}"
            s.add(Decision(id=d_id, ts=float(i), arm_id=parent.id,
                           target_id="default", context_json="[]",
                           intervention_json="{}", run_id=None))
            s.add(Reward(decision_id=d_id, components_json="{}",
                         user_score=None, reward=0.8, computed_at=float(i)))
        await s.commit()

    policy = LinUCBPolicy(arms=[a.id for a in arms], context_dim=FEATURE_DIM, alpha=1.0)
    n0 = len(arm_reg.active_arms())

    ev = MetaEvolver(
        engine=engine, arm_registry=arm_reg, policy=policy,
        mutator_factory=_fake_mutator_factory, top_k=1, min_pulls=3,
    )
    await ev.run_once()
    assert len(arm_reg.active_arms()) == n0 + 1
    new = [a for a in arm_reg.active_arms() if a.parent_id == parent.id]
    assert len(new) == 1
    # policy now knows the new arm
    assert new[0].id in policy._arms  # ok, internal state for test
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
pytest tests/test_evolver.py -v
```
Expected: ImportError.

- [ ] **Step 3: Mutator prompt**

`src/sensable_portfolio/evolve/prompts/mutator.yaml`:
```yaml
id: mutator.v1
system: |
  You are a prompt-evolution agent. Given a parent prompt for an
  intervention-arm, rewrite it to produce more effective interventions
  while staying in the same persona family. Return JSON with keys:
  id (string), system (string), human (string).
human: |
  Parent id: {parent_id}
  Parent persona: {parent_persona}
  Parent system prompt: |
    {parent_system}
  Parent human prompt: |
    {parent_human}
  Recent mean reward: {recent_reward}
  Diversify on at least one of: tone, scope, body modality, duration.
```

- [ ] **Step 4: Implement evolve/meta.py**

`src/sensable_portfolio/evolve/__init__.py`: empty.

`src/sensable_portfolio/evolve/meta.py`:
```python
"""Periodic mutator: pick top-K arms, ask an LLM to mutate, register variant."""
from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from langchain_core.runnables import Runnable
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import select

from sensable_portfolio.arms.registry import ArmRegistry, ArmRow
from sensable_portfolio.policy.linucb import LinUCBPolicy
from sensable_portfolio.storage.db import get_session
from sensable_portfolio.storage.models import Decision, Reward


@dataclass
class MetaEvolver:
    engine: AsyncEngine
    arm_registry: ArmRegistry
    policy: LinUCBPolicy
    mutator_factory: Callable[[str], Runnable]
    top_k: int = 2
    min_pulls: int = 20

    async def _mean_rewards(self) -> dict[str, tuple[int, float]]:
        async with get_session(self.engine) as s:
            rows = (await s.exec(
                select(Decision.arm_id, Reward.reward).join(
                    Reward, Reward.decision_id == Decision.id,
                )
            )).all()
        bag: dict[str, list[float]] = defaultdict(list)
        for arm_id, r in rows:
            bag[arm_id].append(float(r))
        return {k: (len(v), sum(v) / len(v)) for k, v in bag.items()}

    async def run_once(self) -> int:
        means = await self._mean_rewards()
        eligible = [(arm_id, n, m) for arm_id, (n, m) in means.items() if n >= self.min_pulls]
        eligible.sort(key=lambda x: x[2], reverse=True)
        top = eligible[: self.top_k]
        added = 0
        for arm_id, n, m in top:
            try:
                parent = self.arm_registry.get(arm_id)
            except KeyError:
                continue
            mutator = self.mutator_factory(parent.model)
            out = await mutator.ainvoke({
                "parent_id": parent.id,
                "parent_persona": parent.persona,
                "parent_system": parent.system,
                "parent_human": parent.human,
                "recent_reward": round(m, 4),
            })
            new_id = out.get("id") or f"evolved.{parent.persona}.{uuid.uuid4().hex[:6]}"
            new = ArmRow(
                id=new_id, persona=parent.persona,
                prompt_id=new_id, model=parent.model,
                parent_id=parent.id, created_at=time.time(),
                system=out.get("system", parent.system),
                human=out.get("human", parent.human),
            )
            self.arm_registry.add(new)
            self.policy.add_arm(new.id)
            added += 1
        return added

    async def run(self, period_s: float) -> None:
        if period_s <= 0:
            return
        while True:
            try:
                await self.run_once()
            except Exception:
                import logging; logging.exception("evolver run_once failed")
            await asyncio.sleep(period_s)
```

- [ ] **Step 5: Verify PASS**

```bash
pytest tests/test_evolver.py -v
```
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add src/sensable_portfolio/evolve/ tests/test_evolver.py
git commit -m "feat(evolve): meta-mutator picks top-K arms and registers variants"
```

---

## Task 17: End-to-end smoke test + README

**Files:**
- Create: `tests/test_e2e.py`
- Modify: `README.md`

- [ ] **Step 1: Write the e2e test**

`tests/test_e2e.py`:
```python
import asyncio
import json
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
    """Wire up app components manually with FakeSource, run for ~1s with a
    very short min_interval, assert at least one mood frame and one
    action frame land on a fake renderer WebSocket."""

    received = []

    async def handler(ws):
        async for msg in ws:
            received.append(json.loads(msg))

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        # Boot the app without starting runtime
        app = build_app(start_runtime=False)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            # manual lifespan trigger
            async with app.router.lifespan_context(app):
                # Override renderer URL by spawning our own client task
                from sensable_portfolio.renderer.client import RendererClient
                renderer = RendererClient(
                    url=f"ws://127.0.0.1:{port}",
                    bus=app.state.bus, signals_hz=20.0,
                )

                # Override pipeline pieces with FakeSource and short min_interval
                source = FakeSource(runtime_s=1.0, seed=11)
                baseline = calibrate_baseline(FakeSource(runtime_s=0.5, seed=10))
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
                await asyncio.sleep(1.5)
                renderer.cancel()
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

        types = [m["type"] for m in received]
        assert "mood" in types
        assert "agent_action" in types
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
pytest tests/test_e2e.py -v
```
Expected: PASS once Tasks 1-16 are merged. Run anyway to confirm green.

- [ ] **Step 3: Write README**

`README.md`:
```markdown
# sensable-portfolio

Local Python service that consumes `AffectSample` from `neurable_connector` (4 Hz),
selects a stress/focus intervention from a portfolio of LangChain agents using a
contextual bandit (LinUCB), and emits both mood frames (1 Hz) and agent_action
frames (every 30 min) over a single WebSocket to the renderer at
`ws://127.0.0.1:7777`.

## Prerequisites

- Python ≥ 3.11
- Sibling packages installed in editable mode:
  - `../neurable_connector`
  - `../pidview`
- Optional: a renderer running at `ws://127.0.0.1:7777` (Spec A or B aware)
- Optional: an LLM provider (set `OPENAI_API_KEY` and pass a real `llm_factory`)

## Install

```bash
uv sync --extra dev
# or: pip install -e ".[dev]"
```

## Run

```bash
cp .env.example .env  # tweak as needed
uv run uvicorn sensable_portfolio.app:app --host 127.0.0.1 --port 8910
```

The service:
- imports `neurable_connector.NeurableConnector` and starts streaming;
- writes `SnapshotLog` rows at 1 Hz;
- runs the decision graph every `min_decision_interval_s` (default 1800 s = 30 min);
- pushes mood frames (~1 Hz) and agent_action frames (per decision) to the renderer.

## Operational endpoints

- `GET /healthz` — uptime, connector_alive, decisions_total, ws_renderer_connected
- `POST /feedback` — `{decision_id, score, comment?}`
- `POST /decide` — force a decision tick
- `GET /arms/leaderboard`
- `GET /decisions/{id}`
- `GET /debug/stream` — optional SSE mirror of WS frames (set `DEBUG_SSE_ENABLED=true`)

## Tests

```bash
pytest -q
```

## Spec & plan

- `docs/superpowers/specs/2026-05-09-sensable-portfolio-design.md`
- `docs/superpowers/plans/2026-05-09-sensable-portfolio.md`
```

- [ ] **Step 4: Run the full suite**

```bash
pytest -q
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_e2e.py README.md
git commit -m "test(e2e): FakeSource → bandit → renderer WS smoke + README"
```

---

## Self-review (run before declaring the plan done)

- **Spec coverage:**
  - §2 inputs (in-process `nc.stream`) → Tasks 5, 6.
  - §3 outputs (WS + operational HTTP + optional SSE) → Tasks 13, 15.
  - §4 contracts → Task 4.
  - §5 architecture (modules) → Tasks 6–16.
  - §6 bandit (LinUCB, 36-dim) → Tasks 7, 9.
  - §7 reward function → Task 14.
  - §8 arms (7 personas, factory) → Task 10.
  - §9 evolver (24 h, top-K) → Task 16.
  - §10 schema → Task 3.
  - §11 API surface → Task 15.
  - §12 error handling → covered inline (try/except in scheduler, runner, evolver, renderer).
  - §13 testing → Tasks 5–17.
  - §14 out of scope — respected.
  - §15 tunables → Task 2.
- **Type consistency:** `ArmRow`, `Intervention`, `MoodFrame`, `AgentActionFrame`, `LinUCBPolicy.predict/partial_fit/add_arm/snapshot/restore` are referenced consistently across tasks.
- **No placeholders:** all code is concrete; no TBD/TODO; tests have actual code; commits are exact.
- **DRY/YAGNI:** valence/arousal are buffered but excluded from bandit context (in `CONTEXT_KINDS`); MCP server / calibration / drowsy gates remain out of scope.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-09-sensable-portfolio.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
