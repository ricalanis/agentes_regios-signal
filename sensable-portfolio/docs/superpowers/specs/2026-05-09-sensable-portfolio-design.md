# sensable-portfolio — design spec

**Status:** locked 2026-05-09
**Path:** `/Users/ricalanis/Dev/agentes_regios/sensable-portfolio/`
**Sibling deps (path-installed):** `agentes_regios/neurable_connector`, `agentes_regios/pidview`

## 1. Goal

Run a portfolio of LangChain agents that each propose a stress-reducing / focus-restoring intervention from the live affective signal stream, select among them with a contextual bandit (exploration/exploitation), measure the impact of the chosen intervention on the signals, and continuously deliver the latest winning recommendation over a local API. The portfolio is flexible: it grows over iterations as a meta-evolver mutates top-performing arms.

The v1 reward target is locked: lower stress, higher focus.

## 2. Inputs (in-process, no network ingest)

The connector is an existing local Python package. Ingest happens by importing and iterating, not by hosting a server.

```python
from neurable_connector import NeurableConnector, AffectSample, MW75Source, Baseline
from pidview import SignalRegistry, Snapshot
```

`NeurableConnector.stream()` is an async generator yielding `AffectSample` at 4 Hz (every 250 ms). Each sample carries 8 typed dims (`focus, stress, valence, arousal, joy, calm, excitement, neutral`) plus an opaque `features` debug dict. Every sample is fanned out to a `pidview.SignalRegistry` via `registry.push(name, t, value)` per dim; the registry produces `Snapshot` objects with `present, differential, integral, history, stats` precomputed. We never re-derive these.

A configurable `Source` selects between the real MW75 board and `FakeMW75` (synthetic) for tests and demos.

## 3. Outputs

**Primary output channel:** WebSocket client to `ws://127.0.0.1:7777` (the renderer at the other end). Signals and actions ride the same connection as two frame types. Fire-and-forget; reconnect with 1s→5s backoff; silent close on either side.

**Mood frames** (Spec A, sent at `renderer_signals_hz`, default **1 Hz**, drawn from the latest `AffectSample`):
```json
{
  "v": 1,
  "type": "mood",
  "vector": {
    "focus": -0.1128, "stress": -1.9065,
    "valence": -1.9584, "arousal": 0.7499,
    "joy": 0.0, "calm": 0.0, "excitement": 0.0511, "neutral": 0.4819
  },
  "ts": 1778364427578
}
```
All 8 dims of `AffectSample` go in `vector`. Spec A explicitly allows any subset or additional keys, so `valence`, `arousal`, `neutral` (not in the reference palette) are fine — Spec A consumers ignore unknown keys.

**Action frames** (Spec B–shaped, sent on each decision tick — cadence = `min_decision_interval_s`, default **1800 s = 30 min**):
```json
{
  "v": 1,
  "type": "agent_action",
  "ts": 1778364427578,
  "decision_id": "uuid",
  "agent": {"id": "breath_coach.v1", "persona": "breath_coach", "model": "claude-sonnet-4-6", "parent_id": null},
  "intervention": { "action_type": "breath", "title": "...", "body": "...", "duration_s": 90, "intensity": "low", "rationale": "..." },
  "signals_at_decision": { "focus": ..., "stress": ..., "valence": ..., "arousal": ..., "joy": ..., "calm": ..., "excitement": ..., "neutral": ... }
}
```
Spec A renderers ignore unknown `type` (per spec note that Spec B lands additively). Spec B–aware renderers route on `type`.

**Secondary output channel: FastAPI on `127.0.0.1:8910`** — operational/debug only:
- `POST /feedback` — outcome score from any consumer (renderer, CLI, dashboard).
- `POST /decide` — force a decision tick (debug; bypasses `min_decision_interval_s` gate).
- `GET /decisions/{id}` — audit a past decision + reward + feedback.
- `GET /arms/leaderboard` — bandit state.
- `GET /healthz` — `{status, uptime, connector_alive, ws_renderer_connected, decisions_total, last_decision_ts}`.
- `GET /debug/stream` — optional SSE mirror of the frames we push to 7777, for local dev when no renderer is up.

The renderer at 7777 is **not** hosted by us; it's the other side. We only emit. Port 8910 was picked to avoid 7777 (renderer) and 8787 (existing breakneurable daemon).

## 4. Recommendation contract

```python
class Recommendation(Protocol):
    schema_version: int
    decision_id: str
    arm_id: str
    ts: float

class Intervention(BaseModel):           # the only concrete impl in v1
    schema_version: Literal[1] = 1
    decision_id: str
    arm_id: str
    ts: float
    action_type: str                     # "breath", "micro_break", "reframe", ...
    title: str
    body: str
    duration_s: float
    intensity: Literal["low","med","high"]
    rationale: str
```

Producers (arms) and consumers (bandit/storage/SSE/renderer-client) depend on the `Recommendation` Protocol, not on `Intervention` directly. Future shapes (UI specs, multi-step plans) drop in without rewriting consumers.

## 5. Architecture

```
┌──────────────────────┐
│  neurable_connector  │  in-process; async stream() @ 4 Hz
└──────────┬───────────┘
           │ AffectSample
           ▼
┌─────────────────────────────────────────────────────────────────┐
│  sensable-portfolio                                             │
│                                                                 │
│  connector/runner.py                                            │
│    async for s in nc.stream():                                  │
│      registry.push(<each dim>, s.t, value)                      │
│      snapshot_log.write(s)               # 1 Hz subset → SQLite │
│      new_data.set()                                             │
│                                                                 │
│  tick/scheduler.py                                              │
│    on new_data + min_interval_s gate (default 90s) →            │
│      featurize from registry.snapshot_all()                     │
│      policy.predict(context) → arm_id                           │
│      run_arm(arm_id, context) → Intervention                    │
│      persist Decision; pubsub.publish(decision)                 │
│                                                                 │
│  reward/scheduler.py (background, 30 s scan)                    │
│    scan unscored decisions where now() ≥ ts + window_s →        │
│      Δ_k = post_window_mean(k) - baseline_pre_mean(k)           │
│      z   = Δ / running_std_1h(k)                                │
│      reward = clip(-0.5·z(stress) + 0.5·z(focus) + α·u, ±1)     │
│      policy.partial_fit; persist Reward                         │
│                                                                 │
│  evolve/meta.py (cron, default hourly)                          │
│    pick top-K by EWMA reward → LLM mutates parent prompt →      │
│    insert Arm row with parent_id, fresh (A=I, b=0)              │
│                                                                 │
│  app.py  FastAPI on :8910                                       │
│    GET  /recommendations/stream    SSE                          │
│    POST /feedback                                               │
│    POST /decide                                                 │
│    GET  /arms/leaderboard                                       │
│    GET  /decisions/{id}                                         │
│    GET  /healthz                                                │
│                                                                 │
│  renderer/client.py (optional)                                  │
│    WS client → ws://127.0.0.1:7777, Spec A mood frames @ ~5 Hz  │
│    backoff 1s→5s, fire-and-forget                               │
└─────────────────────────────────────────────────────────────────┘
```

### Module layout

```
src/sensable_portfolio/
├── app.py                # FastAPI + lifespan
├── config.py             # pydantic-settings (PORT=8910, MIN_INTERVAL_S=90, RENDERER_WS_URL, …)
├── contracts.py          # Recommendation Protocol + Intervention
├── connector/
│   ├── runner.py         # async loop: nc.stream() → registry.push() + snapshot_log + new_data
│   └── source.py         # build NeurableConnector w/ MW75Source or FakeMW75 from env
├── signals/
│   ├── view.py           # SignalRegistry facade
│   └── features.py       # FeatureVector builder from per-kind Snapshots
├── tick/
│   └── scheduler.py      # asyncio.Event + min_interval_s debounce
├── arms/
│   ├── registry.py       # YAML-backed catalog (id, persona, prompt_id, model, parent_id, retired_at)
│   ├── prompts/          # 5–7 seed personas (breath_coach, micro_break, reframe_cbt,
│   │                     #                    body_scan, env_tweak, social_nudge, deep_focus)
│   └── factory.py        # (arm_row) → LangChain Runnable (LLM + structured output → Intervention)
├── policy/
│   ├── base.py           # Policy Protocol
│   ├── linucb.py         # MABWiser LinUCB disjoint, Gaussian reward in [-1,1]
│   └── persistence.py    # snapshot/restore via SQLite blob; fall back to replay
├── graph/
│   ├── nodes.py          # featurize, select, run_arm, emit, persist
│   └── decision.py       # LangGraph compile + SqliteSaver checkpointer
├── reward/
│   ├── attribution.py    # baseline_pre / window_lo / window_hi / z-score / clip
│   ├── scheduler.py      # background scanner, 30 s cadence
│   └── feedback.py       # POST /feedback handler
├── evolve/
│   ├── meta.py           # cron mutator: pick top-K → LLM mutate → registry insert
│   └── prompts/          # mutator template
├── stream/
│   ├── pubsub.py         # in-process asyncio fan-out
│   └── sinks.py          # SSESink today; AgentActionSink slot reserved for Spec B
├── renderer/
│   └── client.py         # WebSocket client to ws://127.0.0.1:7777, backoff, fire-and-forget
├── storage/
│   ├── models.py         # SQLModel: Decision, Reward, Feedback, Arm, PolicySnapshot, SnapshotLog
│   └── db.py             # aiosqlite, WAL
└── observability/
    └── langsmith.py      # optional run tagging + create_feedback (env-gated)
```

## 6. Bandit

- Library: **MABWiser**, `LinUCB` disjoint, Gaussian reward.
- Reward range: clipped to `[-1, 1]`.
- Context vector (**36-dim**): for each of the 6 registered kinds {focus, stress, joy, calm, excitement, neutral} we concatenate `[present, differential, integral, history_last_3]` from the Snapshot — 6 dims × 6 kinds = 36. `valence`/`arousal` are buffered but excluded from context in v1; opt in via `config/tuning.yaml` (`bandit.context_kinds`).
- Cold start: LinUCB's optimism term naturally inflates fresh arms; no special-casing.
- Arm growth: disjoint design — new arm = fresh `(A=I_d, b=0)`; existing arms untouched.
- Persistence: pickle the MABWiser state to `PolicySnapshot.blob` every 50 decisions and on shutdown. On startup, restore; if restore fails, replay `Decision`+`Reward` to rebuild.
- Upgrade path (not in v1): swap to discounted / sliding-window LinTS for non-stationarity. Behind the `Policy` Protocol — single-class swap.

## 7. Reward function

```
target.weights = {stress: -0.5, focus: +0.5}        # config/targets.yaml; locked v1

For each k in target.weights:
    baseline_mean(k) = mean(snapshot_log[k] in [t - baseline_pre,  t])
    outcome_mean(k)  = mean(snapshot_log[k] in [t + window_lo,     t + window_hi])
    Δ_k_raw = outcome_mean(k) - baseline_mean(k)
    Δ_k_z   = Δ_k_raw / running_std_1h(k)
    component_k = target.weights[k] * Δ_k_z

raw    = Σ component_k
reward = clip(raw + α * user_score, -1, 1)          # α=0.5 if feedback present, else 0
```

Defaults (overridable in `config/tuning.yaml`): `window_s = 300`, `baseline_pre = 120`, `window_lo = 60`, `window_hi = 360`, `α = 0.5`, `min_decision_interval_s = 90`.

## 8. Arms

- Seed personas (YAML, 5–7): `breath_coach, micro_break, reframe_cbt, body_scan, env_tweak, social_nudge, deep_focus`.
- Arm row schema: `id, persona, prompt_id, model, parent_id, created_at, retired_at`.
- Factory: `(arm_row) → LangChain Runnable` that takes `FeatureVector + recent_brief` and emits a structured `Intervention` (Pydantic schema → LLM structured output).
- Hot-reload: `arms/registry.py` watches `arms.yaml` + DB; new rows are picked up on the next decision tick.

## 9. Meta-evolver

- Cron: default every 60 min (configurable; off in tests).
- Picks top-K arms by EWMA reward over the last 24 h (default K=2).
- Calls a "mutator" LLM with the parent prompt, the recent contexts where the parent did well, recent rewards, and a diversity nudge ("differ from siblings on axis X").
- Inserts new `Arm` row with `parent_id`, fresh `(A=I, b=0)`.
- Optional retirement: arms with `pulls > 50` and `EWMA_reward < -0.1` are marked `retired_at`. Off by default in v1.

## 10. Persistence (SQLModel + aiosqlite, WAL)

```python
class Decision(SQLModel, table=True):
    id: str = Field(primary_key=True)
    ts: float
    arm_id: str
    target_id: str = "default"
    context_json: str
    intervention_json: str
    run_id: str | None
    # idx: (ts), (arm_id, ts)

class Reward(SQLModel, table=True):
    decision_id: str = Field(primary_key=True, foreign_key="decision.id")
    components_json: str
    user_score: float | None
    reward: float
    computed_at: float

class Feedback(SQLModel, table=True):
    decision_id: str = Field(primary_key=True, foreign_key="decision.id")
    score: float
    comment: str | None
    ts: float

class Arm(SQLModel, table=True):
    id: str = Field(primary_key=True)
    persona: str
    prompt_id: str
    model: str
    parent_id: str | None
    created_at: float
    retired_at: float | None

class PolicySnapshot(SQLModel, table=True):
    id: int | None = Field(primary_key=True)
    ts: float
    algo: str
    blob: bytes

class SnapshotLog(SQLModel, table=True):
    id: int | None = Field(primary_key=True)
    ts: float
    kind: str
    value: float
    # idx: (kind, ts) — used for reward attribution
```

`SnapshotLog` is written at ~1 Hz from `connector/runner.py` (downsample of the 4 Hz stream) and is the source of truth for reward attribution windows.

## 11. API surface

| method | path | purpose |
|---|---|---|
| GET | `/debug/stream` | (optional) SSE mirror of WS frames sent to 7777 — local dev only |
| POST | `/feedback` | `{decision_id, score: float in [-1,1], comment?}` |
| POST | `/decide` | force a decision tick (bypasses min_interval; debug/manual) |
| GET | `/arms/leaderboard` | `[{arm_id, pulls, mean_reward, last_pulled, retired}]` |
| GET | `/decisions/{id}` | full decision + reward (if attributed) + feedback (if any) |
| GET | `/healthz` | `{status, uptime, connector_alive, decisions_total, last_decision_ts, ws_renderer_connected}` |

## 12. Error handling

- **Connector death** → restart loop with 1s→5s backoff; surface in `/healthz`.
- **LLM call failure** in `run_arm` → log; emit a sentinel "no-op" Intervention with `arm_id` annotated `failed`. Bandit gets a small negative reward (default -0.1) when the post-window closes.
- **Pydantic validation failure** on Intervention → treat as failed call.
- **Renderer disconnect** → silent reconnect with backoff; never block decision loop.
- **DB write failure** → log + continue; in-memory bandit state preserved; snapshot retry on next tick.
- **Late frames** (`ts < oldest_open_reward_window`) → drop; do not retroactively change stored features.

## 13. Testing

- A `FakeSource` fixture in `tests/conftest.py` implements the `neurable_connector.Source` Protocol with deterministic synthetic input (mirroring the `FakeMW75` defined inline in `/tmp/dump_affect_signal.py`). Reuses `Baseline.fit(...)` from `neurable_connector` for the calibration step.
- `FakeArm` fixture: deterministic arm whose Intervention is a function of context.
- `test_policy.py`: 5 simulated arms with known reward functions over context regimes; assert LinUCB picks the right arm in the right regime within N pulls.
- `test_reward_attribution.py`: inject decisions at fixed t, fast-forward synthetic snapshot log, assert attribution computes the expected Δ.
- `test_decision_graph.py`: end-to-end through LangGraph with FakeArm + FakeMW75; assert Decision row + SSE emit + reward eventually lands.
- `test_renderer_client.py`: spin up an ephemeral `websockets.serve`, point client at it, assert frames arrive and reconnect on close.
- `test_api.py`: SSE, `/feedback`, `/decide`.

## 14. Out of scope (v1)

- Multi-tenancy / `session_id` on the API surface.
- HTTP `POST /signals` ingest (replaced by in-process `nc.stream()`).
- MCP server.
- `Calibration`, drowsy gate, valence gate, cross-session correction (upstream concerns).
- Spec B emit (`type:"agent_action"`); the slot is reserved in `stream/sinks.py` and `renderer/client.py`, but no behavior.
- Dashboard / UI.

## 15. Tunables (default → file)

| key | default | file |
|---|---|---|
| `port` | 8910 | `config.py` |
| `min_decision_interval_s` | 1800 (= 30 min) | `config/tuning.yaml` |
| `renderer_signals_hz` | 1.0 | `config/tuning.yaml` |
| `window_s` / `baseline_pre` / `window_lo` / `window_hi` | 300 / 120 / 60 / 360 | `config/tuning.yaml` |
| `feedback_alpha` | 0.5 | `config/tuning.yaml` |
| `target` | `{stress: -0.5, focus: +0.5}` | `config/targets.yaml` |
| `evolver_cron_h` | 24 (off in tests) | `config/tuning.yaml` |
| `policy_snapshot_every` | 50 | `config/tuning.yaml` |
| `renderer_ws_url` | `ws://127.0.0.1:7777` | env / `.env` |
| `renderer_enabled` | `true` | env |
| `debug_sse_enabled` | `false` | env |
