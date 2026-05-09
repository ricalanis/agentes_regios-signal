"""Compose neurable_connector + pidview into a one-line-per-sample monitor of
all 8 affective dimensions.

Each AffectSample is decomposed into:
  - signed z-scores: focus, stress, valence, arousal
    (focus + stress get the full P/D/I treatment as the primary signals)
  - non-negative intensities: joy, calm, excitement, neutral

All 8 are pushed through a SignalRegistry so each gets its own P/D/H/I view.
The line printed shows P for everything plus D and I for focus/stress.

Run with a real MW75 headset (default):

    python examples/focus_stress_pid.py

Run in simulated mode (no headset required, uses a noise-driven fake source):

    python examples/focus_stress_pid.py --simulated
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from typing import Iterator

import numpy as np

from neurable_connector import (
    Baseline,
    EEGFrame,
    FS_HZ,
    NeurableConnector,
    Source,
)
from pidview import SignalRegistry, Snapshot


SIGNED_DIMS = ("focus", "stress", "valence", "arousal")
INTENSITY_DIMS = ("joy", "calm", "excitement", "neutral")
ALL_DIMS = SIGNED_DIMS + INTENSITY_DIMS


# -- Simulated source ---------------------------------------------------------


class _FakeMW75Source:
    """Yields noise-driven EEGFrames at FS_HZ for ~runtime_s seconds.

    Implements the Source protocol: an iterable of EEGFrame. The signal is
    bandlimited pink-ish noise plus a small alpha bump, enough to exercise the
    pipeline end-to-end without a real headset.
    """

    def __init__(self, runtime_s: float = 30.0, seed: int = 0):
        self.runtime_s = float(runtime_s)
        self._rng = np.random.default_rng(seed)
        self._t0 = time.time()

    def __iter__(self) -> Iterator[EEGFrame]:
        n_ch = 12
        fs = float(FS_HZ)
        dt = 1.0 / fs
        t = self._t0
        end = self._t0 + self.runtime_s
        # Tiny alpha bump (~10 Hz) to keep posterior alpha non-degenerate.
        alpha_phase = self._rng.uniform(0.0, 2 * np.pi, size=n_ch)
        i = 0
        # Throttle to roughly real time so the streaming loop has time to emit.
        emit_every = max(1, int(fs / 50))  # sleep every ~20 ms of generated data
        while t < end:
            phase = 2 * np.pi * 10.0 * (i / fs) + alpha_phase
            sample = (
                self._rng.standard_normal(n_ch).astype(np.float64) * 5.0
                + np.sin(phase) * 1.5
            )
            yield EEGFrame(t=t, samples=sample)
            i += 1
            t += dt
            if i % emit_every == 0:
                # Sleep just enough to let the asyncio loop drain.
                time.sleep(emit_every * dt * 0.5)


def _fit_synthetic_baseline(seconds: float = 5.0, seed: int = 1) -> Baseline:
    """Fit a Baseline on a few seconds of synthetic frames (simulated mode only)."""
    fake = _FakeMW75Source(runtime_s=seconds, seed=seed)
    frames = list(fake)
    return Baseline.fit(frames, fs=float(FS_HZ))


# -- Main ---------------------------------------------------------------------


_HEADER = (
    "# t=rel.sec | F/S = focus/stress (P/D/I) | v=valence a=arousal | "
    "J=joy C=calm E=excite N=neutral"
)


def _format_line(t_rel: float, snaps: dict[str, Snapshot]) -> str:
    foc, stress, val, aro = (snaps[k] for k in SIGNED_DIMS)
    j, c, e, n = (snaps[k].present for k in INTENSITY_DIMS)
    return (
        f"{t_rel:+6.2f}s  "
        f"F={foc.present:+.2f}/{foc.differential:+.2f}/{foc.integral:+.2f}  "
        f"S={stress.present:+.2f}/{stress.differential:+.2f}/{stress.integral:+.2f}  "
        f"v={val.present:+.2f} a={aro.present:+.2f}  "
        f"| J={j:.2f} C={c:.2f} E={e:.2f} N={n:.2f}"
    )


async def run(simulated: bool, max_seconds: float | None) -> int:
    reg = SignalRegistry()
    for name in ALL_DIMS:
        reg.register(name, history_seconds=600.0, integral_tau=60.0)

    if simulated:
        # Build the connector around a fake source. Skip the long calibration
        # and inject a baseline fit on a few seconds of synthetic frames.
        runtime = max_seconds if max_seconds is not None else 30.0
        # Baseline gets its own short fake stream; the streaming connector gets
        # a fresh one (sources are single-shot per iteration).
        baseline = _fit_synthetic_baseline(seconds=5.0, seed=1)
        stream_source: Source = _FakeMW75Source(runtime_s=runtime + 5.0, seed=2)
        connector = NeurableConnector(source=stream_source, baseline=baseline)
    else:
        connector = NeurableConnector()

    started = time.monotonic()
    try:
        async with connector as nc:
            if not simulated:
                print(
                    "Calibrating baseline. Sit quietly for the first 90 s eyes "
                    "open, then 90 s eyes closed. Total: ~180 s.",
                    file=sys.stderr,
                    flush=True,
                )
                nc.calibrate_baseline(duration_s=180.0)
                print("Baseline calibrated. Streaming...", file=sys.stderr, flush=True)

            print(_HEADER, file=sys.stderr, flush=True)
            t0: float | None = None
            async for s in nc.stream():
                if t0 is None:
                    t0 = s.t
                for name in ALL_DIMS:
                    reg.push(name, s.t, getattr(s, name))
                snaps = reg.snapshot_all()
                print(_format_line(s.t - t0, snaps), flush=True)
                if max_seconds is not None and (time.monotonic() - started) >= max_seconds:
                    return 0
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--simulated",
        action="store_true",
        help="Use a noise-driven fake source instead of the real MW75 headset.",
    )
    ap.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Stop after this many seconds (useful for smoke tests).",
    )
    args = ap.parse_args()
    try:
        return asyncio.run(run(simulated=args.simulated, max_seconds=args.max_seconds))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
