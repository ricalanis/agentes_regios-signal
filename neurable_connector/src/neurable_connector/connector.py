"""NeurableConnector: async streaming of AffectSamples from a Source."""
from __future__ import annotations

import asyncio
import threading
from collections import deque
from typing import AsyncIterator

import numpy as np

from .baseline import Baseline
from .pipeline import compute_features
from .scores import compute_all
from .source import MW75Source
from .types import AffectSample, EEGFrame, FS_HZ, Source


_SENTINEL = object()


class NeurableConnector:
    """Async streamer of AffectSamples from a Source."""

    def __init__(
        self,
        source: Source | None = None,
        baseline: Baseline | None = None,
        fs: float = float(FS_HZ),
        window_s: float = 1.0,
        hop_s: float = 0.25,
    ):
        self.source: Source = source if source is not None else MW75Source()
        self.baseline: Baseline | None = baseline
        self.fs = float(fs)
        self.window_samples = int(round(self.fs * window_s))
        self.hop_samples = int(round(self.fs * hop_s))
        self._closed = False

    # -- context management ------------------------------------------------
    async def __aenter__(self) -> "NeurableConnector":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """Release the underlying source if it has a close() method."""
        if self._closed:
            return
        self._closed = True
        close_fn = getattr(self.source, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:
                pass

    # -- baseline ----------------------------------------------------------
    def calibrate_baseline(
        self,
        duration_s: float = 180.0,
    ) -> Baseline:
        """Synchronously collect frames and fit a baseline."""
        target_n = int(round(duration_s * self.fs))
        frames: list[EEGFrame] = []
        it = iter(self.source)
        for fr in it:
            frames.append(fr)
            if len(frames) >= target_n:
                break
        baseline = Baseline.fit(
            frames,
            fs=self.fs,
            window_s=self.window_samples / self.fs,
            hop_s=self.hop_samples / self.fs,
        )
        self.baseline = baseline
        return baseline

    # -- streaming ---------------------------------------------------------
    async def stream(self) -> AsyncIterator[AffectSample]:
        """Yield AffectSample at hop rate (4 Hz default)."""
        if self.baseline is None:
            raise RuntimeError(
                "Baseline not set; call calibrate_baseline() or pass baseline=..."
            )

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=4096)

        def producer() -> None:
            try:
                for fr in iter(self.source):
                    fut = asyncio.run_coroutine_threadsafe(queue.put(fr), loop)
                    try:
                        fut.result()
                    except Exception:
                        return
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(_SENTINEL), loop)

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        ring: deque[np.ndarray] = deque(maxlen=self.window_samples)
        last_t: float = 0.0
        since_emit = 0
        try:
            while True:
                item = await queue.get()
                if item is _SENTINEL:
                    return
                fr: EEGFrame = item
                ring.append(np.asarray(fr.samples, dtype=np.float64))
                last_t = fr.t
                since_emit += 1
                if len(ring) < self.window_samples:
                    continue
                if since_emit < self.hop_samples:
                    continue
                since_emit = 0
                window = np.stack(ring, axis=0)
                features = compute_features(window, fs=self.fs)
                yield compute_all(last_t, features, self.baseline)
        finally:
            self.close()
