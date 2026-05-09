"""Single live test: MW75 reachable -> 5 s of plausible data."""
from __future__ import annotations

import time

import numpy as np

from neurable_connector import EEGFrame


def test_live_mw75_pulls_5s_of_plausible_data(live_mw75):
    """Pull 5 s, assert no NaN, >=2000 frames, monotonic timestamps."""
    src = live_mw75
    frames: list[EEGFrame] = []
    start = time.monotonic()
    for fr in iter(src):
        frames.append(fr)
        if time.monotonic() - start >= 5.0:
            break
    src.close()

    assert len(frames) >= 2000, f"only got {len(frames)} frames in 5 s"
    samples = np.stack([f.samples for f in frames], axis=0)
    assert samples.shape[1] == 12
    assert not np.any(np.isnan(samples)), "NaN in EEG samples"
    ts = np.array([f.t for f in frames], dtype=np.float64)
    assert np.all(np.diff(ts) > 0), "timestamps not strictly increasing"
