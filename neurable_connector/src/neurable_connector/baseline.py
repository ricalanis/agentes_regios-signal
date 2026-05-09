"""Baseline: per-feature mean/std fit over an eyes-open recording."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from .pipeline import compute_features
from .types import EEGFrame, FS_HZ


DEFAULT_BASELINE_PATH = Path.home() / ".breakneurable" / "baseline.json"

FEATURE_KEYS = ("posterior_alpha", "mu_lap", "posterior_hfd")

OLD_BASELINE_WARNING = (
    "neurable_connector: baseline file is from older version, defaulting "
    "asymmetry/arousal stats - recalibrate for full accuracy."
)


@dataclass
class Baseline:
    """Per-feature means and stds for z-scoring."""
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    # Posterior alpha asymmetry stats (valence latent).
    mu_asym: float = 0.0
    sigma_asym: float = 1.0
    # Beta/alpha ratio stats (arousal latent).
    mu_ba: float = 0.0
    sigma_ba: float = 1.0

    @classmethod
    def fit(
        cls,
        frames: Iterable[EEGFrame],
        fs: float = float(FS_HZ),
        window_s: float = 1.0,
        hop_s: float = 0.25,
    ) -> "Baseline":
        """Fit baseline by sliding the same 1 s / 0.25 s window over frames."""
        win = int(round(fs * window_s))
        hop = int(round(fs * hop_s))
        samples: list[np.ndarray] = []
        for fr in frames:
            samples.append(np.asarray(fr.samples, dtype=np.float64))
        if len(samples) < win:
            raise ValueError(
                f"Need at least {win} frames for fitting; got {len(samples)}"
            )
        arr = np.stack(samples, axis=0)  # (N, 12)

        per_key: dict[str, list[float]] = {k: [] for k in FEATURE_KEYS}
        asym_vals: list[float] = []
        ba_vals: list[float] = []
        i = 0
        while i + win <= arr.shape[0]:
            feats = compute_features(arr[i : i + win], fs=fs)
            for k in FEATURE_KEYS:
                v = feats.get(k, float("nan"))
                if np.isfinite(v):
                    per_key[k].append(v)
            a = feats.get("posterior_asymmetry", float("nan"))
            if np.isfinite(a):
                asym_vals.append(float(a))
            b = feats.get("beta_alpha_ratio", float("nan"))
            if np.isfinite(b):
                ba_vals.append(float(b))
            i += hop

        means: dict[str, float] = {}
        stds: dict[str, float] = {}
        for k in FEATURE_KEYS:
            vals = np.array(per_key[k], dtype=np.float64)
            if vals.size == 0:
                means[k] = 0.0
                stds[k] = 1.0
                continue
            mu = float(np.mean(vals))
            sigma = float(np.std(vals, ddof=0))
            if not np.isfinite(sigma) or sigma <= 1e-12:
                sigma = 1.0
            means[k] = mu
            stds[k] = sigma

        mu_asym, sigma_asym = _mu_sigma(asym_vals)
        mu_ba, sigma_ba = _mu_sigma(ba_vals)
        return cls(
            means=means,
            stds=stds,
            mu_asym=mu_asym,
            sigma_asym=sigma_asym,
            mu_ba=mu_ba,
            sigma_ba=sigma_ba,
        )

    def save(self, path: str | os.PathLike | None = None) -> Path:
        """Persist as JSON; defaults to ~/.breakneurable/baseline.json."""
        p = Path(path) if path is not None else DEFAULT_BASELINE_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "means": self.means,
            "stds": self.stds,
            "mu_asym": self.mu_asym,
            "sigma_asym": self.sigma_asym,
            "mu_ba": self.mu_ba,
            "sigma_ba": self.sigma_ba,
        }
        p.write_text(json.dumps(payload, indent=2))
        return p

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> "Baseline":
        """Load from JSON; defaults to ~/.breakneurable/baseline.json.

        Older baselines without asymmetry/arousal stats load with
        mu=0/sigma=1 defaults and emit a single stderr warning.
        """
        p = Path(path) if path is not None else DEFAULT_BASELINE_PATH
        data = json.loads(Path(p).read_text())
        missing = any(
            k not in data for k in ("mu_asym", "sigma_asym", "mu_ba", "sigma_ba")
        )
        if missing:
            print(OLD_BASELINE_WARNING, file=sys.stderr)
        return cls(
            means=dict(data["means"]),
            stds=dict(data["stds"]),
            mu_asym=float(data.get("mu_asym", 0.0)),
            sigma_asym=float(data.get("sigma_asym", 1.0)),
            mu_ba=float(data.get("mu_ba", 0.0)),
            sigma_ba=float(data.get("sigma_ba", 1.0)),
        )


def _mu_sigma(vals: list[float]) -> tuple[float, float]:
    """Mean and std of a list with sane fallbacks."""
    if not vals:
        return 0.0, 1.0
    arr = np.array(vals, dtype=np.float64)
    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=0))
    if not np.isfinite(sigma) or sigma <= 1e-12:
        sigma = 1.0
    return mu, sigma
