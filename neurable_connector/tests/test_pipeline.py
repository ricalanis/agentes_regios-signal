"""Offline pipeline tests against synthetic signals."""
from __future__ import annotations

import numpy as np

from neurable_connector.pipeline import (
    BANDS,
    POSTERIOR_ALPHA_IDX,
    POSTERIOR_HFD_IDX,
    band_power,
    beta_alpha_ratio,
    channel_band_powers,
    compute_features,
    filter_window,
    higuchi_fd,
    mu_laplacian_signals,
    posterior_asymmetry,
    welch_psd,
)
from neurable_connector.types import FS_HZ


def _sine(freq_hz: float, n: int, fs: float, amp: float = 50.0) -> np.ndarray:
    t = np.arange(n) / fs
    return amp * np.sin(2 * np.pi * freq_hz * t)


def test_alpha_dominates_for_10hz_sinusoid():
    """A pure 10 Hz sinusoid puts power in the alpha band, not theta or beta."""
    fs = float(FS_HZ)
    n = int(fs * 1.0)
    rng = np.random.default_rng(0)
    x = np.zeros((n, 12))
    for c in range(12):
        x[:, c] = _sine(10.0, n, fs) + 0.5 * rng.standard_normal(n)
    feats = compute_features(x, fs=fs)
    # Compute band means by re-running the inner pipeline pieces.
    x_filt = filter_window(x, fs=fs)
    f, pxx = welch_psd(x_filt, fs=fs)
    alpha = channel_band_powers(f, pxx, BANDS["alpha"])
    theta = channel_band_powers(f, pxx, BANDS["theta"])
    beta = channel_band_powers(f, pxx, BANDS["beta"])
    assert alpha.mean() > 5.0 * theta.mean()
    assert alpha.mean() > 5.0 * beta.mean()
    # Sanity: features dict has expected keys.
    for k in ("posterior_alpha", "posterior_hfd", "mu_lap"):
        assert k in feats


def test_higuchi_high_for_white_noise_low_for_sine():
    """HFD ~= 2 for white noise, ~= 1 for a clean sinusoid."""
    rng = np.random.default_rng(42)
    n = 500
    noise = rng.standard_normal(n)
    sine = _sine(10.0, n, fs=500.0, amp=1.0)
    hfd_noise = higuchi_fd(noise, k_max=10)
    hfd_sine = higuchi_fd(sine, k_max=10)
    assert 1.7 < hfd_noise < 2.1
    assert 0.9 < hfd_sine < 1.4
    assert hfd_noise > hfd_sine + 0.4


def test_mu_laplacian_uses_correct_indices():
    """C5-CP5 is x[:,5]-x[:,3]; C6-CP6 is x[:,11]-x[:,9]."""
    n, ch = 50, 12
    x = np.zeros((n, ch))
    x[:, 5] = 7.0
    x[:, 3] = 2.0
    x[:, 11] = -1.0
    x[:, 9] = 4.0
    lap = mu_laplacian_signals(x)
    assert lap.shape == (n, 2)
    assert np.allclose(lap[:, 0], 5.0)  # 7 - 2
    assert np.allclose(lap[:, 1], -5.0)  # -1 - 4


def test_posterior_indices_match_spec():
    """Posterior alpha = TP7,P7,TP8,P8 -> 2,4,8,10. Posterior HFD = P7,P8 -> 4,10."""
    assert POSTERIOR_ALPHA_IDX == (2, 4, 8, 10)
    assert POSTERIOR_HFD_IDX == (4, 10)


def test_filter_window_removes_dc_and_60hz():
    """1 Hz HP kills DC; notch attenuates a 60 Hz tone."""
    fs = float(FS_HZ)
    n = int(fs * 1.0)
    t = np.arange(n) / fs
    base = 100.0 * np.ones((n, 12))
    sixty = 50.0 * np.sin(2 * np.pi * 60.0 * t)[:, None] * np.ones((1, 12))
    ten = 10.0 * np.sin(2 * np.pi * 10.0 * t)[:, None] * np.ones((1, 12))
    x = base + sixty + ten
    y = filter_window(x, fs=fs)
    # DC gone.
    assert abs(y.mean()) < 1.0
    # 60 Hz attenuated relative to 10 Hz in the PSD.
    f, pxx = welch_psd(y, fs=fs)
    p10 = band_power(f, pxx[:, 0], 9.0, 11.0)
    p60 = band_power(f, pxx[:, 0], 59.0, 61.0)
    assert p10 > 10.0 * p60


def test_band_power_zero_outside_range():
    f = np.linspace(0, 100, 1001)
    pxx = np.zeros_like(f)
    pxx[(f >= 8.0) & (f < 13.0)] = 1.0
    assert band_power(f, pxx, 14.0, 30.0) == 0.0
    assert band_power(f, pxx, 8.0, 13.0) > 0.0


def test_posterior_asymmetry_positive_when_right_alpha_stronger():
    """10 Hz alpha 2x stronger on TP8/P8 than TP7/P7 -> asymmetry > 0."""
    fs = float(FS_HZ)
    n = int(fs * 1.0)
    rng = np.random.default_rng(0)
    x = np.zeros((n, 12))
    sine = _sine(10.0, n, fs, amp=1.0)
    base = 0.1 * rng.standard_normal((n, 12))
    x[:] = base
    # Left posterior TP7 (2), P7 (4): amplitude 25.
    x[:, 2] += 25.0 * sine
    x[:, 4] += 25.0 * sine
    # Right posterior TP8 (8), P8 (10): amplitude 50 -> 4x power, 2x amp.
    x[:, 8] += 50.0 * sine
    x[:, 10] += 50.0 * sine
    feats = compute_features(x, fs=fs)
    assert feats["posterior_asymmetry"] > 0.5


def test_posterior_asymmetry_zero_when_balanced():
    """Equal alpha on left and right posterior -> asymmetry ~ 0."""
    fs = float(FS_HZ)
    n = int(fs * 1.0)
    rng = np.random.default_rng(1)
    x = np.zeros((n, 12))
    sine = _sine(10.0, n, fs, amp=1.0)
    base = 0.1 * rng.standard_normal((n, 12))
    x[:] = base
    for c in (2, 4, 8, 10):
        x[:, c] += 30.0 * sine
    feats = compute_features(x, fs=fs)
    assert abs(feats["posterior_asymmetry"]) < 0.1


def test_posterior_asymmetry_monotonic_in_ratio():
    """Direct call: feeding stronger right alpha gives strictly larger asym."""
    alpha_balanced = np.array([1.0] * 12)
    alpha_right_2x = np.ones(12)
    for i in (8, 10):
        alpha_right_2x[i] = 2.0
    alpha_right_4x = np.ones(12)
    for i in (8, 10):
        alpha_right_4x[i] = 4.0
    a0 = posterior_asymmetry(alpha_balanced)
    a1 = posterior_asymmetry(alpha_right_2x)
    a2 = posterior_asymmetry(alpha_right_4x)
    assert a0 < a1 < a2
    assert abs(a0) < 1e-9


def test_beta_alpha_ratio_higher_for_strong_beta_signal():
    """20 Hz strong + 10 Hz weak -> larger b/a than the inverse case."""
    fs = float(FS_HZ)
    n = int(fs * 1.0)
    rng = np.random.default_rng(2)
    base = 0.1 * rng.standard_normal((n, 12))

    # Beta-dominant signal: strong 20 Hz, weak 10 Hz.
    x_beta = base + (
        50.0 * _sine(20.0, n, fs, amp=1.0)[:, None]
        + 5.0 * _sine(10.0, n, fs, amp=1.0)[:, None]
    )
    # Alpha-dominant signal: strong 10 Hz, weak 20 Hz.
    x_alpha = base + (
        5.0 * _sine(20.0, n, fs, amp=1.0)[:, None]
        + 50.0 * _sine(10.0, n, fs, amp=1.0)[:, None]
    )

    feats_beta = compute_features(x_beta, fs=fs)
    feats_alpha = compute_features(x_alpha, fs=fs)
    assert feats_beta["beta_alpha_ratio"] > feats_alpha["beta_alpha_ratio"]
    assert feats_beta["beta_alpha_ratio"] > 1.0
    assert feats_alpha["beta_alpha_ratio"] < 1.0


def test_beta_alpha_ratio_uses_central_parietal_indices():
    """Direct math: only 3,4,5,9,10,11 contribute."""
    beta = np.zeros(12)
    alpha = np.ones(12)
    # Set beta only at non-listed channels: should give zero.
    for i in (0, 1, 2, 6, 7, 8):
        beta[i] = 10.0
    assert beta_alpha_ratio(beta, alpha) == 0.0
    # Set beta at listed channels: should give 10/1.
    beta = np.zeros(12)
    for i in (3, 4, 5, 9, 10, 11):
        beta[i] = 10.0
    assert abs(beta_alpha_ratio(beta, alpha) - 10.0) < 1e-9
