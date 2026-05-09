"""Snapshot dataclass shared between SignalView and consumers."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Snapshot:
    """Immutable point-in-time view of a signal."""

    name: str
    t: float
    present: float
    differential: float
    integral: float
    history: np.ndarray
    stats: dict[str, float] = field(default_factory=dict)
