"""Shared fixtures: live_mw75 with auto-skip on unavailability."""
from __future__ import annotations

import pytest

from neurable_connector import MW75Source, MW75Unavailable


@pytest.fixture
def live_mw75():
    """Yield an MW75Source if the headset is reachable; skip otherwise.

    Binary discovery: MW75_CSV_BIN env var > package-local native/bin/mw75-csv >
    'mw75-csv' on PATH (see MW75Source._default_binary).
    """
    src = MW75Source(timeout_s=5.0)
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
        fresh = MW75Source(timeout_s=5.0)
        yield fresh
        fresh.close()
    finally:
        src.close()
