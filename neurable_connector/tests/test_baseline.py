"""Baseline fit + JSON save/load roundtrip + old-format compatibility."""
from __future__ import annotations

import json

import numpy as np

from neurable_connector import Baseline, EEGFrame
from neurable_connector.baseline import FEATURE_KEYS
from neurable_connector.types import FS_HZ


def _synthetic_frames(n_seconds: float = 5.0, fs: float = float(FS_HZ)):
    rng = np.random.default_rng(123)
    n = int(round(n_seconds * fs))
    t = np.arange(n) / fs
    frames = []
    for i in range(n):
        # 10 Hz alpha tone + small white noise across all 12 channels.
        ch = 50.0 * np.sin(2 * np.pi * 10.0 * t[i]) + 1.0 * rng.standard_normal(12)
        frames.append(EEGFrame(t=float(t[i]), samples=ch.astype(np.float64)))
    return frames


def test_baseline_fit_produces_finite_means_and_stds():
    frames = _synthetic_frames()
    bl = Baseline.fit(frames, fs=float(FS_HZ))
    for k in FEATURE_KEYS:
        assert k in bl.means
        assert k in bl.stds
        assert np.isfinite(bl.means[k])
        assert np.isfinite(bl.stds[k])
        assert bl.stds[k] > 0
    # New fields populated and finite.
    for f in ("mu_asym", "sigma_asym", "mu_ba", "sigma_ba"):
        assert np.isfinite(getattr(bl, f))
    assert bl.sigma_asym > 0
    assert bl.sigma_ba > 0


def test_baseline_save_load_roundtrip(tmp_path):
    frames = _synthetic_frames()
    bl = Baseline.fit(frames, fs=float(FS_HZ))
    path = tmp_path / "baseline.json"
    bl.save(path)
    loaded = Baseline.load(path)
    for k in FEATURE_KEYS:
        assert loaded.means[k] == bl.means[k]
        assert loaded.stds[k] == bl.stds[k]
    assert loaded.mu_asym == bl.mu_asym
    assert loaded.sigma_asym == bl.sigma_asym
    assert loaded.mu_ba == bl.mu_ba
    assert loaded.sigma_ba == bl.sigma_ba


def test_baseline_load_old_format_defaults_with_warning(tmp_path, capsys):
    """Old baseline JSON (no asym/ba fields) loads with defaults + stderr warning."""
    path = tmp_path / "old_baseline.json"
    payload = {
        "means": {"posterior_alpha": 1.0, "mu_lap": 2.0, "posterior_hfd": 0.5},
        "stds": {"posterior_alpha": 0.5, "mu_lap": 1.0, "posterior_hfd": 0.2},
    }
    path.write_text(json.dumps(payload))

    loaded = Baseline.load(path)
    assert loaded.mu_asym == 0.0
    assert loaded.sigma_asym == 1.0
    assert loaded.mu_ba == 0.0
    assert loaded.sigma_ba == 1.0
    # Existing fields preserved.
    assert loaded.means["posterior_alpha"] == 1.0
    assert loaded.stds["mu_lap"] == 1.0

    captured = capsys.readouterr()
    assert "older version" in captured.err
    assert "asymmetry/arousal" in captured.err
    # Single warning, not multiple.
    assert captured.err.count("neurable_connector:") == 1
