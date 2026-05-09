"""Score formula tests with synthetic features and baselines."""
from __future__ import annotations

import math

from neurable_connector import (
    AffectSample,
    Baseline,
    compute_all,
    compute_arousal,
    compute_calm,
    compute_excitement,
    compute_focus,
    compute_joy,
    compute_neutral,
    compute_stress,
    compute_valence,
)


def _baseline(
    mu_alpha=0.0, sd_alpha=1.0,
    mu_mu=0.0, sd_mu=1.0,
    mu_hfd=0.0, sd_hfd=1.0,
    mu_asym=0.0, sd_asym=1.0,
    mu_ba=0.0, sd_ba=1.0,
) -> Baseline:
    return Baseline(
        means={
            "posterior_alpha": mu_alpha,
            "mu_lap": mu_mu,
            "posterior_hfd": mu_hfd,
        },
        stds={
            "posterior_alpha": sd_alpha,
            "mu_lap": sd_mu,
            "posterior_hfd": sd_hfd,
        },
        mu_asym=mu_asym,
        sigma_asym=sd_asym,
        mu_ba=mu_ba,
        sigma_ba=sd_ba,
    )


def _flat_features(
    alpha: float = 0.0,
    mu: float = 0.0,
    hfd: float = 0.0,
    asym: float = 0.0,
    ba: float = 0.0,
) -> dict[str, float]:
    return {
        "posterior_alpha": alpha,
        "mu_lap": mu,
        "posterior_hfd": hfd,
        "posterior_asymmetry": asym,
        "beta_alpha_ratio": ba,
    }


def test_focus_formula_exact():
    """focus = -z_alpha + 0.5 * (-z_mu)."""
    bl = _baseline(mu_alpha=2.0, sd_alpha=2.0, mu_mu=10.0, sd_mu=5.0)
    feats = {"posterior_alpha": 6.0, "mu_lap": 20.0, "posterior_hfd": 0.0}
    z_alpha = (6.0 - 2.0) / 2.0  # = 2
    z_mu = (20.0 - 10.0) / 5.0  # = 2
    expected = -z_alpha + 0.5 * (-z_mu)  # = -3
    assert math.isclose(compute_focus(feats, bl), expected, rel_tol=1e-9)


def test_stress_formula_exact():
    """stress = -z_alpha + 0.5 * z_hfd."""
    bl = _baseline(mu_alpha=1.0, sd_alpha=0.5, mu_hfd=1.5, sd_hfd=0.25)
    feats = {"posterior_alpha": 0.5, "mu_lap": 0.0, "posterior_hfd": 2.0}
    z_alpha = (0.5 - 1.0) / 0.5  # = -1
    z_hfd = (2.0 - 1.5) / 0.25  # = 2
    expected = -z_alpha + 0.5 * z_hfd  # = 1 + 1 = 2
    assert math.isclose(compute_stress(feats, bl), expected, rel_tol=1e-9)


def test_focus_sign_alpha_decrease_increases_focus():
    """Posterior alpha below baseline -> z_alpha negative -> focus positive."""
    bl = _baseline(mu_alpha=10.0, sd_alpha=1.0, mu_mu=0.0, sd_mu=1.0)
    feats = {"posterior_alpha": 8.0, "mu_lap": 0.0, "posterior_hfd": 0.0}
    assert compute_focus(feats, bl) > 0.0


def test_stress_sign_hfd_increase_increases_stress():
    """HFD above baseline pushes stress up, alpha at baseline has no effect."""
    bl = _baseline(mu_hfd=1.0, sd_hfd=0.1)
    feats = {"posterior_alpha": 0.0, "mu_lap": 0.0, "posterior_hfd": 1.2}
    assert compute_stress(feats, bl) > 0.0


# -- valence/arousal/labels ---------------------------------------------


def test_valence_z_against_baseline():
    bl = _baseline(mu_asym=0.2, sd_asym=0.4)
    feats = _flat_features(asym=1.0)
    expected = (1.0 - 0.2) / 0.4  # = 2.0
    assert math.isclose(compute_valence(feats, bl), expected, rel_tol=1e-9)


def test_arousal_z_against_baseline():
    bl = _baseline(mu_ba=0.5, sd_ba=0.25)
    feats = _flat_features(ba=1.0)
    expected = (1.0 - 0.5) / 0.25  # = 2.0
    assert math.isclose(compute_arousal(feats, bl), expected, rel_tol=1e-9)


def test_joy_dominates_for_positive_valence_mid_arousal():
    """valence > 0, arousal ~ +0.5 -> joy is the largest of the four labels."""
    v, a = 1.5, 0.5
    joy = compute_joy(v, a)
    calm = compute_calm(v, a)
    exc = compute_excitement(v, a)
    neutral = compute_neutral(0.0, 0.0, joy, calm, exc)
    assert joy > calm
    assert joy > exc
    assert joy > neutral


def test_calm_dominates_for_positive_valence_negative_arousal():
    """valence > 0, arousal < 0 -> calm wins."""
    v, a = 1.5, -1.0
    joy = compute_joy(v, a)
    calm = compute_calm(v, a)
    exc = compute_excitement(v, a)
    neutral = compute_neutral(0.0, 0.0, joy, calm, exc)
    assert calm > joy
    assert calm > exc
    assert calm > neutral


def test_excitement_dominates_for_zero_valence_high_arousal():
    """valence ~ 0, arousal > 1 -> excitement wins."""
    v, a = 0.0, 2.0
    joy = compute_joy(v, a)
    calm = compute_calm(v, a)
    exc = compute_excitement(v, a)
    neutral = compute_neutral(0.0, 0.0, joy, calm, exc)
    assert exc > joy
    assert exc > calm
    assert exc > neutral


def test_neutral_close_to_one_when_all_zs_zero():
    """All zs near 0 -> neutral close to 1, others small."""
    v, a = 0.0, 0.0
    joy = compute_joy(v, a)
    calm = compute_calm(v, a)
    exc = compute_excitement(v, a)
    neutral = compute_neutral(0.0, 0.0, joy, calm, exc)
    assert neutral > 0.95
    assert joy < 1e-6
    assert calm < 1e-6
    # Excitement uses Gauss(v=0, mu=0.5, sigma=1.5)*relu(0) = 0.
    assert exc < 1e-6


def test_neutral_drops_when_focus_high_but_valence_arousal_zero():
    """High focus z but no valence/arousal: neutral drops, labels stay near 0."""
    v, a = 0.0, 0.0
    joy = compute_joy(v, a)
    calm = compute_calm(v, a)
    exc = compute_excitement(v, a)
    focus_high = 2.0
    n_zero = compute_neutral(0.0, 0.0, joy, calm, exc)
    n_focus = compute_neutral(focus_high, 0.0, joy, calm, exc)
    assert n_focus < n_zero
    assert joy < 1e-6
    assert calm < 1e-6
    assert exc < 1e-6


def test_compute_all_returns_affectsample_with_all_fields():
    bl = _baseline(mu_asym=0.0, sd_asym=1.0, mu_ba=0.0, sd_ba=1.0)
    feats = _flat_features(asym=1.0, ba=0.5)
    out = compute_all(123.45, feats, bl)
    assert isinstance(out, AffectSample)
    assert out.t == 123.45
    # All numeric fields finite.
    for f in ("focus", "stress", "valence", "arousal",
              "joy", "calm", "excitement", "neutral"):
        v = getattr(out, f)
        assert isinstance(v, float)
        assert math.isfinite(v)
    assert 0.0 <= out.neutral <= 1.0
    assert out.joy >= 0.0
    assert out.calm >= 0.0
    assert out.excitement >= 0.0
    assert out.features is feats
