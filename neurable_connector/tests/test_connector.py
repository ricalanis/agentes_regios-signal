"""Connector tests with a fake source injected."""
from __future__ import annotations

import asyncio
from typing import Iterator

import numpy as np
import pytest

from neurable_connector import (
    AffectSample,
    Baseline,
    EEGFrame,
    FocusStressSample,
    NeurableConnector,
)
from neurable_connector.types import FS_HZ


class _FakeSource:
    """Iterable yielding synthetic frames; supports close()."""

    def __init__(self, n_seconds: float, fs: float = float(FS_HZ), seed: int = 7):
        self.n = int(round(n_seconds * fs))
        self.fs = fs
        self.seed = seed
        self.closed = False

    def __iter__(self) -> Iterator[EEGFrame]:
        rng = np.random.default_rng(self.seed)
        for i in range(self.n):
            t = i / self.fs
            phase = 2 * np.pi * 10.0 * t
            ch = 50.0 * np.sin(phase) + 1.0 * rng.standard_normal(12)
            yield EEGFrame(t=float(t), samples=ch.astype(np.float64))

    def close(self) -> None:
        self.closed = True


def _flat_baseline() -> Baseline:
    return Baseline(
        means={"posterior_alpha": 0.0, "mu_lap": 0.0, "posterior_hfd": 0.0},
        stds={"posterior_alpha": 1.0, "mu_lap": 1.0, "posterior_hfd": 1.0},
    )


def test_connector_emits_about_4hz_over_5_seconds():
    """5 s of synthetic frames -> ~20 samples (4 Hz, 1 s warm-up)."""
    src = _FakeSource(n_seconds=5.0)
    bl = _flat_baseline()

    async def run():
        samples: list[FocusStressSample] = []
        async with NeurableConnector(source=src, baseline=bl) as nc:
            async for s in nc.stream():
                samples.append(s)
        return samples

    out = asyncio.run(run())
    # 5 s @ 500 Hz, window 500 samples, hop 125 samples.
    # First emit at sample 500, then every 125 -> emits at 500,625,...,2500.
    # That's 17 emits.
    assert 14 <= len(out) <= 21, f"got {len(out)} samples"
    for s in out:
        assert isinstance(s, AffectSample)
        # FocusStressSample is the legacy alias of AffectSample.
        assert isinstance(s, FocusStressSample)
        assert np.isfinite(s.focus)
        assert np.isfinite(s.stress)
        assert np.isfinite(s.valence)
        assert np.isfinite(s.arousal)
        assert s.joy >= 0.0
        assert s.calm >= 0.0
        assert s.excitement >= 0.0
        assert 0.0 <= s.neutral <= 1.0
        assert "posterior_alpha" in s.features
        assert "posterior_asymmetry" in s.features
        assert "beta_alpha_ratio" in s.features


def test_connector_calibrate_baseline_with_fake_source():
    src = _FakeSource(n_seconds=2.0)
    nc = NeurableConnector(source=src)
    bl = nc.calibrate_baseline(duration_s=2.0)
    assert nc.baseline is bl
    for k in ("posterior_alpha", "mu_lap", "posterior_hfd"):
        assert k in bl.means
        assert k in bl.stds


def test_stream_requires_baseline():
    src = _FakeSource(n_seconds=1.0)

    async def run():
        async with NeurableConnector(source=src) as nc:
            async for _ in nc.stream():
                pass

    with pytest.raises(RuntimeError):
        asyncio.run(run())


def test_async_context_closes_source():
    src = _FakeSource(n_seconds=1.0)
    bl = _flat_baseline()

    async def run():
        async with NeurableConnector(source=src, baseline=bl) as nc:
            assert nc.baseline is bl

    asyncio.run(run())
    assert src.closed is True
