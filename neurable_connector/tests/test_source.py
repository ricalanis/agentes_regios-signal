"""Regression tests for MW75Source timeout / discovery."""
from __future__ import annotations

import os
import time

import pytest

from neurable_connector import MW75Source, MW75Unavailable


def _write_script(path, body: str) -> str:
    path.write_text(body)
    path.chmod(0o755)
    return str(path)


def test_timeout_when_subprocess_silent(tmp_path):
    """Subprocess alive but producing no output -> MW75Unavailable within timeout."""
    silent = _write_script(tmp_path / "silent.sh", "#!/bin/sh\nsleep 60\n")
    src = MW75Source(binary=silent, timeout_s=0.4)
    start = time.monotonic()
    with pytest.raises(MW75Unavailable):
        next(iter(src))
    elapsed = time.monotonic() - start
    assert elapsed < 2.0, f"timeout took {elapsed:.2f}s, expected < 2s"


def test_timeout_when_subprocess_exits_silently(tmp_path):
    """Subprocess exits without any output -> MW75Unavailable promptly."""
    quitter = _write_script(tmp_path / "quit.sh", "#!/bin/sh\nexit 0\n")
    src = MW75Source(binary=quitter, timeout_s=2.0)
    start = time.monotonic()
    with pytest.raises(MW75Unavailable):
        next(iter(src))
    elapsed = time.monotonic() - start
    assert elapsed < 1.0


def test_unknown_binary_raises_unavailable():
    src = MW75Source(binary="/no/such/binary-xyzzy", timeout_s=1.0)
    with pytest.raises(MW75Unavailable):
        next(iter(src))


def test_inter_frame_stall_raises(tmp_path):
    """Subprocess emits one valid line then goes silent -> raises after timeout."""
    # 14 fields: ts_us,counter,ch1..ch12
    line = "1700000000000000,0," + ",".join(["0.0"] * 12)
    feeder = _write_script(
        tmp_path / "feeder.sh",
        f'#!/bin/sh\necho "{line}"\nsleep 60\n',
    )
    src = MW75Source(binary=feeder, timeout_s=0.4)
    it = iter(src)
    fr = next(it)
    assert fr is not None
    start = time.monotonic()
    with pytest.raises(MW75Unavailable):
        next(it)
    elapsed = time.monotonic() - start
    assert elapsed < 2.0


def test_default_binary_discovery_prefers_env(monkeypatch, tmp_path):
    """MW75_CSV_BIN env var wins over the package-local default."""
    fake = _write_script(tmp_path / "fake.sh", "#!/bin/sh\nexit 0\n")
    monkeypatch.setenv("MW75_CSV_BIN", fake)
    src = MW75Source()
    assert src.binary == fake
