"""Build a 36-dim FeatureVector for the bandit context from per-kind Snapshots."""
from __future__ import annotations

import numpy as np
from pidview import Snapshot, SignalView

from sensable_portfolio.signals.view import CONTEXT_KINDS

PER_KIND = 6   # present, differential, integral, last_3 history values
FEATURE_DIM = len(CONTEXT_KINDS) * PER_KIND  # 6 * 6 = 36


def _verify_pidview_snapshot_shape() -> None:
    """Module-load contract check: Snapshot.history must be (N, 2) [t, x] rows."""
    v = SignalView("__contract_check", history_seconds=10.0, integral_tau=1.0)
    v.push(0.0, 1.0)
    v.push(1.0, 2.0)
    h = v.snapshot().history
    if h.ndim != 2 or h.shape[1] != 2:
        raise ImportError(
            f"pidview.Snapshot.history shape drift: got {h.shape}, expected (N, 2). "
            "sensable_portfolio.signals.features assumes columns [t, x]."
        )


_verify_pidview_snapshot_shape()


def _last_3_values(snap: Snapshot) -> tuple[float, float, float]:
    h = snap.history
    if h.shape[0] == 0:
        return (0.0, 0.0, 0.0)
    vals = h[:, 1]
    if len(vals) >= 3:
        a, b, c = vals[-3], vals[-2], vals[-1]
    elif len(vals) == 2:
        a, b, c = 0.0, vals[-2], vals[-1]
    else:
        a, b, c = 0.0, 0.0, vals[-1]
    return (float(a), float(b), float(c))


def build_feature_vector(snapshots: dict[str, Snapshot]) -> np.ndarray:
    out = np.zeros(FEATURE_DIM, dtype=np.float64)
    for i, kind in enumerate(CONTEXT_KINDS):
        snap = snapshots.get(kind)
        if snap is None:
            continue
        a, b, c = _last_3_values(snap)
        offset = i * PER_KIND
        out[offset + 0] = float(snap.present)
        out[offset + 1] = float(snap.differential)
        out[offset + 2] = float(snap.integral)
        out[offset + 3] = a
        out[offset + 4] = b
        out[offset + 5] = c
    return out
