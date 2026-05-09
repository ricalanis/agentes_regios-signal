"""Public dataclasses, constants, exceptions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Protocol, runtime_checkable

import numpy as np


CH_NAMES: tuple[str, ...] = (
    "FT7", "T7", "TP7", "CP5", "P7", "C5",
    "FT8", "T8", "TP8", "CP6", "P8", "C6",
)
FS_HZ: int = 500


@dataclass(frozen=True)
class EEGFrame:
    """One sample across 12 channels at unix time t."""
    t: float
    samples: np.ndarray  # shape (12,) float64


@dataclass(frozen=True)
class AffectSample:
    """Within-subject z-scored affective sample (focus, stress, valence,
    arousal) plus non-negative intensity labels (joy, calm, excitement,
    neutral)."""
    t: float
    # signed z-scores
    focus: float
    stress: float
    valence: float       # +ve = right > left posterior alpha
    arousal: float       # +ve = high beta/alpha ratio
    # non-negative intensity scores (0 = absent, larger = stronger)
    joy: float
    calm: float
    excitement: float
    neutral: float       # in [0, 1]; peaks when others are quiet
    features: dict[str, float] = field(default_factory=dict)


# Legacy alias for source compatibility.
FocusStressSample = AffectSample


class MW75Unavailable(RuntimeError):
    """Raised when the MW75 subprocess cannot produce data."""


@runtime_checkable
class Source(Protocol):
    """Anything that yields EEGFrames."""
    def __iter__(self) -> Iterator[EEGFrame]: ...
