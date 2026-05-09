"""Shared fixtures: live_mw75 with auto-skip on unavailability."""
from __future__ import annotations

import os

import pytest

from neurable_connector import MW75Source, MW75Unavailable


DEFAULT_BIN = "/Users/ricalanis/Dev/breakneurable/mw75/target/release/mw75-csv"


@pytest.fixture
def live_mw75():
    """Yield an MW75Source if the headset is reachable; skip otherwise."""
    binary = os.environ.get("MW75_CSV_BIN", DEFAULT_BIN)
    src = MW75Source(binary=binary, timeout_s=5.0)
    try:
        # Probe by pulling one frame; if it fails, skip.
        it = iter(src)
        try:
            next(it)
        except MW75Unavailable as e:
            pytest.skip(f"MW75 unavailable: {e}")
        except StopIteration:
            pytest.skip("MW75 yielded no frames")
        # Re-create a fresh source for the test (the probe consumed the iterator).
        src.close()
        fresh = MW75Source(binary=binary, timeout_s=5.0)
        yield fresh
        fresh.close()
    finally:
        src.close()
