# sensable-portfolio

Local Python service that consumes `AffectSample` from `neurable_connector` (4 Hz),
selects a stress/focus intervention from a portfolio of LangChain agents using a
contextual bandit (LinUCB), and emits both mood frames (1 Hz) and agent_action
frames (every 30 min) over a single WebSocket to the renderer at
`ws://127.0.0.1:7777`.

## Prerequisites

- Python >= 3.11
- Sibling packages installed in editable mode (declared as path deps in `pyproject.toml`):
  - `../neurable_connector`
  - `../pidview`
- Optional: a renderer at `ws://127.0.0.1:7777` (Spec A or B aware)
- Optional: an LLM provider (see "LLM providers" below)

## Install

```bash
uv sync --extra dev          # baseline (stub LLM, offline tests)
uv sync --extra dev --extra all   # also install Ollama + Anthropic providers
```

## Run

```bash
cp .env.example .env  # tweak as needed
uv run uvicorn sensable_portfolio.app:app --host 127.0.0.1 --port 8910
```

Service:
- imports `neurable_connector.NeurableConnector` and starts streaming;
- writes `SnapshotLog` rows at 1 Hz;
- runs the decision graph every `min_decision_interval_s` (default 1800 s = 30 min);
- pushes mood + agent_action frames to the renderer.

## Operational endpoints

- `GET /healthz`
- `POST /feedback` — `{decision_id, score, comment?}`
- `POST /decide` — force a decision tick (debug)
- `GET /arms/leaderboard`
- `GET /decisions/{id}`
- `GET /debug/stream` — optional SSE mirror of WS frames (set `DEBUG_SSE_ENABLED=true`)

## LLM providers

Three providers are wired:

| `LLM_PROVIDER` | Description |
|---|---|
| `stub` (default) | Deterministic in-process stub. No network. Used by tests. |
| `ollama` | Local Ollama daemon. Supports both local models AND Ollama Cloud-routed `:cloud` tags (e.g. `kimi-k2:1t-cloud`) once you've run `ollama signin`. |
| `anthropic` | Claude via the Anthropic API. Requires `ANTHROPIC_API_KEY`. |

### Use Kimi via Ollama Cloud

```bash
ollama signin
ollama pull kimi-k2:1t-cloud
uv sync --extra dev --extra ollama
LLM_PROVIDER=ollama OLLAMA_MODEL=kimi-k2:1t-cloud uv run uvicorn sensable_portfolio.app:app --port 8910
```

### Use Claude

```bash
uv sync --extra dev --extra anthropic
LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-... uv run uvicorn sensable_portfolio.app:app --port 8910
```

Per-arm overrides: edit `src/sensable_portfolio/arms/prompts/<persona>.yaml` and change `model: default` to a specific tag (e.g. `model: claude-sonnet-4-6` or `model: qwen2.5:14b-instruct`).

## Tests

```bash
uv run pytest -q
```

## Spec & plan

- `docs/superpowers/specs/2026-05-09-sensable-portfolio-design.md`
- `docs/superpowers/plans/2026-05-09-sensable-portfolio.md`
