"""Settings + tunables loader."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

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
    anthropic_api_key: str | None = None

    # LLM provider selection
    llm_provider: Literal["stub", "ollama", "anthropic"] = "stub"

    # Ollama: local daemon OR direct Cloud API (set OLLAMA_API_KEY for direct).
    # Direct Cloud API:  base_url="https://ollama.com",        model="kimi-k2.6" (bare).
    # Local daemon path: base_url="http://localhost:11434",    model="kimi-k2:1t-cloud"
    # (requires `ollama signin` and a daemon recent enough to support :cloud tags).
    ollama_enabled: bool = False
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "kimi-k2:1t-cloud"
    ollama_api_key: str | None = None

    # Anthropic (Claude)
    anthropic_model: str = "claude-haiku-4-5-20251001"


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    return Settings()
