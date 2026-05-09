"""SignalView: PID-style view over a single scalar time series."""
from __future__ import annotations

from typing import Callable, Optional
import sys

import numpy as np

from pidview.types import Snapshot


class SignalView:
    """Maintain history, leaky integral, and short-window slope for one signal."""

    def __init__(
        self,
        name: str,
        history_seconds: float = 600.0,
        integral_tau: Optional[float] = 60.0,
        differential_window_seconds: float = 2.0,
    ) -> None:
        self.name = name
        self.history_seconds = float(history_seconds)
        self.integral_tau = None if integral_tau is None else float(integral_tau)
        self.differential_window_seconds = float(differential_window_seconds)

        # history is list of (t, x); kept sorted-by-time (push order = monotonic).
        self._hist: list[tuple[float, float]] = []
        self._integral: float = 0.0
        self._integral_t: Optional[float] = None
        self._subs: list[Callable[[Snapshot], None]] = []

    def push(self, t: float, x: float) -> None:
        """Append a sample; update integral; evict old history; notify subscribers."""
        t = float(t)
        x = float(x)

        if self._integral_t is not None and t < self._integral_t:
            # Out-of-order timestamps are not supported; raise to surface the bug.
            raise ValueError(
                f"non-monotonic timestamp: t={t} < previous t={self._integral_t}"
            )

        if self._integral_t is None:
            # First sample initializes integral to 0; no dt available.
            self._integral = 0.0
            self._integral_t = t
        else:
            dt = t - self._integral_t
            if dt > 0.0:
                if self.integral_tau is None:
                    # Trapezoidal accumulation, no decay.
                    x_prev = self._hist[-1][1]
                    self._integral += 0.5 * (x_prev + x) * dt
                else:
                    # Leaky integrator: dI/dt = x - I/tau.
                    # Use forward Euler with current x (consistent with discrete update).
                    self._integral += dt * (x - self._integral / self.integral_tau)
            self._integral_t = t

        self._hist.append((t, x))

        # Evict samples older than history window relative to the latest t.
        cutoff = t - self.history_seconds
        # Strict less-than: keep samples with t >= cutoff.
        i = 0
        for i, (ti, _) in enumerate(self._hist):
            if ti >= cutoff:
                break
        else:
            i = len(self._hist)
        if i > 0:
            self._hist = self._hist[i:]

        snap = self.snapshot()
        for fn in list(self._subs):
            try:
                fn(snap)
            except Exception as e:  # noqa: BLE001
                print(f"pidview subscriber error: {e!r}", file=sys.stderr)

    def snapshot(self) -> Snapshot:
        """Return an immutable Snapshot of the current state."""
        if not self._hist:
            empty = np.zeros((0, 2), dtype=np.float64)
            return Snapshot(
                name=self.name,
                t=0.0,
                present=0.0,
                differential=0.0,
                integral=0.0,
                history=empty,
                stats={
                    "mean": 0.0,
                    "std": 0.0,
                    "p10": 0.0,
                    "p50": 0.0,
                    "p90": 0.0,
                    "slope": 0.0,
                },
            )

        arr = np.asarray(self._hist, dtype=np.float64)  # (N, 2)
        t_latest = float(arr[-1, 0])
        present = float(arr[-1, 1])

        # Differential: slope over (t_latest - window, t_latest].
        window_lo = t_latest - self.differential_window_seconds
        mask = arr[:, 0] > window_lo
        differential = 0.0
        if int(mask.sum()) >= 2:
            ts = arr[mask, 0]
            xs = arr[mask, 1]
            differential = float(np.polyfit(ts, xs, deg=1)[0])

        # Stats over the entire current history.
        xs_all = arr[:, 1]
        ts_all = arr[:, 0]
        stats = {
            "mean": float(np.mean(xs_all)),
            "std": float(np.std(xs_all)),
            "p10": float(np.percentile(xs_all, 10)),
            "p50": float(np.percentile(xs_all, 50)),
            "p90": float(np.percentile(xs_all, 90)),
            "slope": (
                float(np.polyfit(ts_all, xs_all, deg=1)[0])
                if len(xs_all) >= 2
                else 0.0
            ),
        }

        return Snapshot(
            name=self.name,
            t=t_latest,
            present=present,
            differential=differential,
            integral=float(self._integral),
            history=arr.copy(),
            stats=stats,
        )

    def subscribe(self, fn: Callable[[Snapshot], None]) -> Callable[[], None]:
        """Register fn for post-push notification; returns an unsubscribe callable."""
        self._subs.append(fn)

        def unsubscribe() -> None:
            try:
                self._subs.remove(fn)
            except ValueError:
                pass

        return unsubscribe
