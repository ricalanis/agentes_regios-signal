"""Score formulas: focus, stress, valence, arousal, and label intensities."""
from __future__ import annotations

import math

from .baseline import Baseline
from .types import AffectSample


def _z(value: float, mean: float, std: float) -> float:
    if std <= 0:
        return 0.0
    return (value - mean) / std


def _relu(x: float) -> float:
    return max(0.0, float(x))


def _gauss(x: float, mu: float, sigma: float) -> float:
    return math.exp(-((float(x) - mu) / sigma) ** 2)


def compute_focus(features: dict[str, float], baseline: Baseline) -> float:
    """focus = -z_alpha_post + 0.5 * (-z_mu)."""
    z_alpha = _z(
        features["posterior_alpha"],
        baseline.means["posterior_alpha"],
        baseline.stds["posterior_alpha"],
    )
    z_mu = _z(
        features["mu_lap"],
        baseline.means["mu_lap"],
        baseline.stds["mu_lap"],
    )
    return -z_alpha + 0.5 * (-z_mu)


def compute_stress(features: dict[str, float], baseline: Baseline) -> float:
    """stress = -z_alpha_post + 0.5 * z_hfd_post."""
    z_alpha = _z(
        features["posterior_alpha"],
        baseline.means["posterior_alpha"],
        baseline.stds["posterior_alpha"],
    )
    z_hfd = _z(
        features["posterior_hfd"],
        baseline.means["posterior_hfd"],
        baseline.stds["posterior_hfd"],
    )
    return -z_alpha + 0.5 * z_hfd


def compute_valence(features: dict[str, float], baseline: Baseline) -> float:
    return _z(features["posterior_asymmetry"], baseline.mu_asym, baseline.sigma_asym)


def compute_arousal(features: dict[str, float], baseline: Baseline) -> float:
    return _z(features["beta_alpha_ratio"], baseline.mu_ba, baseline.sigma_ba)


def compute_joy(valence: float, arousal: float) -> float:
    return _relu(valence) * _gauss(arousal, mu=0.5, sigma=1.0)


def compute_calm(valence: float, arousal: float) -> float:
    return _relu(valence) * _relu(-arousal)


def compute_excitement(valence: float, arousal: float) -> float:
    return _gauss(valence, mu=0.5, sigma=1.5) * _relu(arousal)


def compute_neutral(
    focus: float, stress: float, joy: float, calm: float, excitement: float
) -> float:
    s = focus**2 + stress**2 + joy**2 + calm**2 + excitement**2
    return math.exp(-s / 5.0)


def compute_all(
    t: float, features: dict[str, float], baseline: Baseline
) -> AffectSample:
    """Single entry point: features + baseline -> AffectSample."""
    focus = compute_focus(features, baseline)
    stress = compute_stress(features, baseline)
    valence = compute_valence(features, baseline)
    arousal = compute_arousal(features, baseline)
    joy = compute_joy(valence, arousal)
    calm = compute_calm(valence, arousal)
    excitement = compute_excitement(valence, arousal)
    neutral = compute_neutral(focus, stress, joy, calm, excitement)
    return AffectSample(
        t=t,
        focus=focus,
        stress=stress,
        valence=valence,
        arousal=arousal,
        joy=joy,
        calm=calm,
        excitement=excitement,
        neutral=neutral,
        features=features,
    )
