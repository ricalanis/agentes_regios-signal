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


@pytest.fixture(autouse=True)
def _isolate_settings_env(monkeypatch, tmp_path):
    """Tests must not be polluted by the project `.env`.

    pydantic-settings resolves `env_file=".env"` relative to the current
    working directory at `Settings()` instantiation. We chdir to a tmp
    dir (no .env present) and clear every sensable-portfolio env var so
    each test sees pristine defaults regardless of the developer's
    local config."""
    for k in (
        "LLM_PROVIDER",
        "OLLAMA_ENABLED", "OLLAMA_API_KEY", "OLLAMA_BASE_URL", "OLLAMA_MODEL",
        "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL",
        "OPENAI_API_KEY", "LANGSMITH_API_KEY",
        "RENDERER_ENABLED", "RENDERER_WS_URL", "RENDERER_SIGNALS_HZ",
        "DB_URL",
        "DEBUG_SSE_ENABLED",
        "PORT",
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(tmp_path)  # cwd has no .env
    from sensable_portfolio.config import load_settings
    load_settings.cache_clear()
    yield
    load_settings.cache_clear()
