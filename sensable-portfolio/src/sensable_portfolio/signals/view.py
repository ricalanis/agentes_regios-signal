"""Thin facade over pidview.SignalRegistry for our 8 affective dims.

Boot-time contract check: the public dim names we ship to the renderer
(`ALL_DIMS`) are asserted against `neurable_connector.AffectSample`
fields at module-load. Drift fails with a clear ImportError at boot,
not silently at runtime."""
from __future__ import annotations

from dataclasses import fields

from neurable_connector import AffectSample
from pidview import SignalRegistry

# Names that exist on AffectSample but are NOT scalar dim values.
_NON_SCALAR = {"t", "features"}

_OBSERVED = tuple(f.name for f in fields(AffectSample) if f.name not in _NON_SCALAR)
_REQUIRED = ("focus", "stress", "valence", "arousal", "joy", "calm", "excitement", "neutral")

_missing = [n for n in _REQUIRED if n not in _OBSERVED]
_extra = [n for n in _OBSERVED if n not in _REQUIRED]
if _missing or _extra:
    raise ImportError(
        "neurable_connector.AffectSample shape drift: "
        f"missing={_missing}, unexpected={_extra}. "
        "Update sensable_portfolio.signals.view._REQUIRED to match upstream."
    )

# Public surface — preserved for callers; order is the wire-stable one.
ALL_DIMS: tuple[str, ...] = _REQUIRED
SIGNED_DIMS: tuple[str, ...] = ("focus", "stress", "valence", "arousal")
INTENSITY_DIMS: tuple[str, ...] = ("joy", "calm", "excitement", "neutral")

# Bandit-context kinds (subset of ALL_DIMS); valence/arousal excluded in v1.
CONTEXT_KINDS: tuple[str, ...] = ("focus", "stress", "joy", "calm", "excitement", "neutral")


def build_registry(history_seconds: float = 600.0, integral_tau: float = 60.0) -> SignalRegistry:
    reg = SignalRegistry()
    for name in ALL_DIMS:
        reg.register(name, history_seconds=history_seconds, integral_tau=integral_tau)
    return reg
