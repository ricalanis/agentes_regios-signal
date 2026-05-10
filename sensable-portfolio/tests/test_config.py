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
    assert s.target_weights == {"stress": -0.5, "focus": 0.5}
    assert s.ollama_enabled is False
    assert s.ollama_base_url == "http://localhost:11434"
    assert s.ollama_model == "kimi-k2:1t-cloud"
    assert s.llm_provider == "stub"
    assert s.anthropic_model == "claude-haiku-4-5-20251001"


def test_env_override(monkeypatch):
    # invalidate the lru_cache so monkeypatch is honored
    load_settings.cache_clear()
    monkeypatch.setenv("RENDERER_ENABLED", "false")
    monkeypatch.setenv("DB_URL", "sqlite+aiosqlite:///:memory:")
    s = load_settings()
    assert s.renderer_enabled is False
    assert s.db_url == "sqlite+aiosqlite:///:memory:"
    load_settings.cache_clear()  # leave it clean for sibling tests
