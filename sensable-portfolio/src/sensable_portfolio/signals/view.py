"""Thin facade over pidview.SignalRegistry for our 8 affective dims."""
from __future__ import annotations

from pidview import SignalRegistry

SIGNED_DIMS: tuple[str, ...] = ("focus", "stress", "valence", "arousal")
INTENSITY_DIMS: tuple[str, ...] = ("joy", "calm", "excitement", "neutral")
ALL_DIMS: tuple[str, ...] = SIGNED_DIMS + INTENSITY_DIMS

# Bandit-context kinds (subset of ALL_DIMS); valence/arousal excluded in v1.
CONTEXT_KINDS: tuple[str, ...] = ("focus", "stress", "joy", "calm", "excitement", "neutral")


def build_registry(history_seconds: float = 600.0, integral_tau: float = 60.0) -> SignalRegistry:
    reg = SignalRegistry()
    for name in ALL_DIMS:
        reg.register(name, history_seconds=history_seconds, integral_tau=integral_tau)
    return reg
