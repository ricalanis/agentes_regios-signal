"""SignalRegistry: a named collection of SignalViews."""
from __future__ import annotations

from pidview.types import Snapshot
from pidview.view import SignalView


class SignalRegistry:
    """Dict-like container of named SignalViews."""

    def __init__(self) -> None:
        self._views: dict[str, SignalView] = {}

    def register(self, name: str, **view_kwargs) -> SignalView:
        """Create and store a SignalView under name; raise if name exists."""
        if name in self._views:
            raise ValueError(f"signal {name!r} already registered")
        view = SignalView(name, **view_kwargs)
        self._views[name] = view
        return view

    def get(self, name: str) -> SignalView:
        """Return the SignalView for name; raise KeyError if missing."""
        if name not in self._views:
            raise KeyError(name)
        return self._views[name]

    def push(self, name: str, t: float, x: float) -> None:
        """Forward a (t, x) sample to the named view."""
        self.get(name).push(t, x)

    def snapshot_all(self) -> dict[str, Snapshot]:
        """Return a snapshot dict for every registered view."""
        return {name: view.snapshot() for name, view in self._views.items()}

    def names(self) -> list[str]:
        """Return registered names in insertion order."""
        return list(self._views.keys())

    def __contains__(self, name: object) -> bool:
        return name in self._views
