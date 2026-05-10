# IO contract — sensable-portfolio

Three boundaries in the data path. Versioning posture: upstream packages
(`neurable_connector`, `pidview`) have no on-wire schema version;
sensable-portfolio enforces shape contracts at module load and emits
versioned frames downstream.

## Boundary 1: `neurable_connector` → sensable-portfolio (in-process)

Producer: `NeurableConnector.stream()` (async generator, ~4 Hz).

Frame: `AffectSample` (frozen dataclass)
- `t: float` — unix seconds
- 8 scalar dims: `focus, stress, valence, arousal, joy, calm, excitement, neutral`
- `features: dict[str, float]` — debug payload, not part of the contract

Consumer: `connector/runner.py` pushes each dim into a `pidview.SignalRegistry`.

Contract enforcement:
- `signals/view.py` derives `ALL_DIMS` from `dataclasses.fields(AffectSample)` and asserts on the canonical 8 names at module load. Drift fails with a clear `ImportError` at boot.

## Boundary 2: sensable-portfolio ↔ `pidview` (in-process)

Calls:
- `SignalRegistry.register(name: str, history_seconds: float, integral_tau: float)`
- `SignalRegistry.push(name: str, t: float, x: float)` — `t` in unix seconds
- `SignalRegistry.snapshot_all() -> dict[str, Snapshot]`

Frame: `Snapshot` (frozen dataclass, per-signal)
- `name: str`
- `t: float` — last-sample unix seconds
- `present, differential, integral: float`
- `history: numpy.ndarray` of shape `(N, 2)`, each row `[t, x]`
- `stats: dict[str, float]`

Contract enforcement:
- `signals/features.py::_verify_pidview_snapshot_shape()` runs at module load and asserts `history.ndim == 2 and history.shape[1] == 2`.

## Boundary 3: sensable-portfolio → renderer (WebSocket out)

Producer: `renderer/client.py` over `ws://127.0.0.1:7777`.

Frames (Pydantic v2, `extra="forbid"`):
- `MoodFrame` — `{v:1, type:"mood", vector:{8 keys, ALL_DIMS}, ts:int_ms}`
- `AgentActionFrame` — `{v:1, type:"agent_action", ts:int_ms, decision_id, agent, intervention, signals_at_decision}`

Time conversion: `contracts.unix_seconds_to_ms(t: float) -> int` is the single
conversion site for unix-seconds (internal) → unix-ms (wire). Truncation,
not rounding, to match prior wire output byte-for-byte.

**Vocabulary divergence (intentional):** Renderer Spec A reference keys are
`joy, calm, focus, stress, sadness, excitement, anger, awe`. We emit
`focus, stress, valence, arousal, joy, calm, excitement, neutral`. We
overlap on 5 (joy, calm, focus, stress, excitement), add 3 the renderer
ignores (valence, arousal, neutral), and omit 3 the upstream sensor
doesn't produce (sadness, anger, awe). Spec A explicitly permits "any
subset or arbitrary additional keys", so this is intentional and not a
contract gap.

Schema version: each frame carries `v: Literal[1]`. Spec B (additive)
will introduce `type:"text"` and bidirectional ack frames; the version
field stays `1` unless a backward-incompatible change lands.
