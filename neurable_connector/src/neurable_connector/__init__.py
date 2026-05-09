"""neurable_connector: MW75 EEG -> affective z-scores + labels."""
from .baseline import Baseline
from .connector import NeurableConnector
from .scores import (
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
from .source import MW75Source
from .types import (
    AffectSample,
    CH_NAMES,
    EEGFrame,
    FocusStressSample,
    FS_HZ,
    MW75Unavailable,
    Source,
)

__all__ = [
    "AffectSample",
    "Baseline",
    "CH_NAMES",
    "EEGFrame",
    "FS_HZ",
    "FocusStressSample",
    "MW75Source",
    "MW75Unavailable",
    "NeurableConnector",
    "Source",
    "compute_all",
    "compute_arousal",
    "compute_calm",
    "compute_excitement",
    "compute_focus",
    "compute_joy",
    "compute_neutral",
    "compute_stress",
    "compute_valence",
]
