"""DSP: filter, Welch PSD, band powers, Higuchi FD, mu Laplacian."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt, welch

from .types import FS_HZ


# Posterior alpha sites: TP7, P7, TP8, P8 -> indices 2,4,8,10
POSTERIOR_ALPHA_IDX: tuple[int, ...] = (2, 4, 8, 10)
# Posterior HFD sites: P7, P8 -> indices 4, 10
POSTERIOR_HFD_IDX: tuple[int, ...] = (4, 10)
# Posterior asymmetry: right TP8, P8 (8, 10) vs left TP7, P7 (2, 4).
ASYM_LEFT_IDX: tuple[int, ...] = (2, 4)
ASYM_RIGHT_IDX: tuple[int, ...] = (8, 10)
# Beta/alpha ratio sites: CP5, P7, C5, CP6, P8, C6 -> 3,4,5,9,10,11.
BETA_ALPHA_IDX: tuple[int, ...] = (3, 4, 5, 9, 10, 11)

BANDS: dict[str, tuple[float, float]] = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


def _design_filters(fs: float):
    hp_sos = butter(4, 1.0 / (fs / 2.0), btype="highpass", output="sos")
    notch_b, notch_a = iirnotch(60.0, 30.0, fs)
    return hp_sos, notch_b, notch_a


def filter_window(x: np.ndarray, fs: float = float(FS_HZ)) -> np.ndarray:
    """Filter (N, 12) window: DC-remove, 1 Hz HP Butterworth, 60 Hz notch."""
    if x.ndim != 2:
        raise ValueError(f"x must be 2D (N, CH); got shape {x.shape}")
    y = x - x.mean(axis=0, keepdims=True)
    hp_sos, notch_b, notch_a = _design_filters(fs)
    y = sosfiltfilt(hp_sos, y, axis=0)
    y = filtfilt(notch_b, notch_a, y, axis=0)
    return y


def band_power(freqs: np.ndarray, pxx: np.ndarray, lo: float, hi: float) -> float:
    """Trapezoid integral of PSD over [lo, hi)."""
    mask = (freqs >= lo) & (freqs < hi)
    if not mask.any():
        return 0.0
    return float(np.trapezoid(pxx[mask], freqs[mask]))


def welch_psd(x: np.ndarray, fs: float = float(FS_HZ)):
    """scipy.signal.welch on (N, CH); returns (freqs, pxx) with pxx shape (F, CH)."""
    nperseg = min(512, x.shape[0])
    f, pxx = welch(x, fs=fs, nperseg=nperseg, axis=0)
    return f, pxx


def channel_band_powers(
    freqs: np.ndarray, pxx: np.ndarray, band: tuple[float, float]
) -> np.ndarray:
    """Per-channel band power vector; pxx shape (F, CH)."""
    lo, hi = band
    n_ch = pxx.shape[1]
    return np.array([band_power(freqs, pxx[:, c], lo, hi) for c in range(n_ch)])


def higuchi_fd(x: np.ndarray, k_max: int = 10) -> float:
    """Higuchi 1988 fractal dimension of 1-D signal x."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < k_max + 1:
        return float("nan")
    lk = np.zeros(k_max)
    log_k = np.zeros(k_max)
    for k in range(1, k_max + 1):
        lm = []
        for m in range(k):
            # Number of subsequence steps for offset m, scale k.
            n_max = (n - m - 1) // k
            if n_max < 1:
                continue
            idx = m + np.arange(n_max + 1) * k
            diffs = np.abs(np.diff(x[idx]))
            norm = (n - 1) / (n_max * k)
            lm.append(diffs.sum() * norm / k)
        if not lm:
            return float("nan")
        lk[k - 1] = np.mean(lm)
        log_k[k - 1] = np.log(1.0 / k)
    log_lk = np.log(lk + 1e-30)
    slope, _ = np.polyfit(log_k, log_lk, 1)
    return float(slope)


def mu_laplacian_signals(x_filt: np.ndarray) -> np.ndarray:
    """Two virtual channels: C5 - CP5 and C6 - CP6. Input (N, 12) filtered."""
    left = x_filt[:, 5] - x_filt[:, 3]
    right = x_filt[:, 11] - x_filt[:, 9]
    return np.stack([left, right], axis=1)


def posterior_asymmetry(alpha_per_channel: np.ndarray) -> float:
    """log(alpha_R) - log(alpha_L) over right TP8+P8 vs left TP7+P7."""
    a = np.asarray(alpha_per_channel, dtype=np.float64)
    right = float(a[list(ASYM_RIGHT_IDX)].sum())
    left = float(a[list(ASYM_LEFT_IDX)].sum())
    return float(np.log(right + 1e-30) - np.log(left + 1e-30))


def beta_alpha_ratio(
    beta_per_channel: np.ndarray,
    alpha_per_channel: np.ndarray,
) -> float:
    """Mean of beta/alpha across central+parietal sites (both hemispheres)."""
    b = np.asarray(beta_per_channel, dtype=np.float64)
    a = np.asarray(alpha_per_channel, dtype=np.float64)
    idx = list(BETA_ALPHA_IDX)
    ratios = b[idx] / (a[idx] + 1e-30)
    return float(np.mean(ratios))


def compute_features(window: np.ndarray, fs: float = float(FS_HZ)) -> dict[str, float]:
    """Run full per-window pipeline, return feature dict."""
    x_filt = filter_window(window, fs=fs)
    freqs, pxx = welch_psd(x_filt, fs=fs)

    alpha = channel_band_powers(freqs, pxx, BANDS["alpha"])
    theta = channel_band_powers(freqs, pxx, BANDS["theta"])
    beta = channel_band_powers(freqs, pxx, BANDS["beta"])

    # Posterior alpha: log-mean across TP7, P7, TP8, P8.
    post_alpha_vals = alpha[list(POSTERIOR_ALPHA_IDX)]
    posterior_alpha = float(np.mean(np.log(post_alpha_vals + 1e-30)))

    # Posterior HFD: log-mean across P7, P8 of time-domain HFD per channel.
    hfd_vals = np.array(
        [higuchi_fd(x_filt[:, c]) for c in POSTERIOR_HFD_IDX], dtype=np.float64
    )
    posterior_hfd = float(np.mean(np.log(hfd_vals + 1e-30)))

    # Mu Laplacian: alpha-band power on the two Laplacian virtual channels.
    mu_signals = mu_laplacian_signals(x_filt)
    mu_freqs, mu_pxx = welch_psd(mu_signals, fs=fs)
    mu_alpha_left = band_power(mu_freqs, mu_pxx[:, 0], *BANDS["alpha"])
    mu_alpha_right = band_power(mu_freqs, mu_pxx[:, 1], *BANDS["alpha"])
    mu_lap = float(np.mean([mu_alpha_left, mu_alpha_right]))

    asym = posterior_asymmetry(alpha)
    ba = beta_alpha_ratio(beta, alpha)

    return {
        "posterior_alpha": posterior_alpha,
        "posterior_hfd": posterior_hfd,
        "mu_lap": mu_lap,
        "theta_mean": float(np.mean(theta)),
        "alpha_mean": float(np.mean(alpha)),
        "beta_mean": float(np.mean(beta)),
        "posterior_asymmetry": asym,
        "beta_alpha_ratio": ba,
    }
