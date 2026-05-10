"""MW75 subprocess source. Spawns mw75-csv and parses CSV stdout."""
from __future__ import annotations

import os
import selectors
import subprocess
import time
from pathlib import Path
from typing import Iterator

import numpy as np

from .types import CH_NAMES, EEGFrame, MW75Unavailable


N_CH = len(CH_NAMES)
_POLL_INTERVAL_S = 0.2


def _default_binary() -> str:
    """Resolve mw75-csv: env var > package-local native build > PATH lookup."""
    env = os.environ.get("MW75_CSV_BIN")
    if env:
        return env
    # neurable_connector/src/neurable_connector/source.py -> .../neurable_connector/
    here = Path(__file__).resolve()
    candidate = here.parent.parent.parent / "native" / "bin" / "mw75-csv"
    if candidate.exists() and os.access(candidate, os.X_OK):
        return str(candidate)
    return "mw75-csv"


class MW75Source:
    """Iterates EEGFrames from the mw75-csv subprocess."""

    def __init__(self, binary: str | None = None, timeout_s: float = 5.0):
        env = os.environ.get("MW75_CSV_BIN")
        if env:
            self.binary = env
        elif binary is not None:
            self.binary = binary
        else:
            self.binary = _default_binary()
        self.timeout_s = float(timeout_s)
        self._proc: subprocess.Popen | None = None

    def __iter__(self) -> Iterator[EEGFrame]:
        try:
            self._proc = subprocess.Popen(
                [self.binary],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except (FileNotFoundError, PermissionError) as e:
            raise MW75Unavailable(f"Cannot launch {self.binary}: {e}") from e

        proc = self._proc
        assert proc.stdout is not None
        sel = selectors.DefaultSelector()
        sel.register(proc.stdout.fileno(), selectors.EVENT_READ)
        last_activity = time.monotonic()
        first_seen = False
        try:
            while True:
                # Hard timeout: bounds first-frame and inter-frame stalls alike.
                if time.monotonic() - last_activity > self.timeout_s:
                    if not first_seen:
                        raise MW75Unavailable(
                            f"No EEG data within {self.timeout_s}s from {self.binary}"
                        )
                    raise MW75Unavailable(
                        f"{self.binary} stalled (no frame for {self.timeout_s}s)"
                    )
                if proc.poll() is not None:
                    if not first_seen:
                        raise MW75Unavailable(
                            f"{self.binary} exited (code {proc.returncode}) before any data"
                        )
                    return
                events = sel.select(timeout=_POLL_INTERVAL_S)
                if not events:
                    continue
                line = proc.stdout.readline()
                if not line:
                    # EOF or closed pipe.
                    if not first_seen:
                        raise MW75Unavailable(
                            f"{self.binary} closed stdout before any data"
                        )
                    return
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) != 2 + N_CH:
                    continue
                try:
                    ts_us = int(parts[0])
                    _counter = int(parts[1])
                    ch_vals = np.fromiter(
                        (float(p) for p in parts[2:]),
                        dtype=np.float64,
                        count=N_CH,
                    )
                except ValueError:
                    continue
                first_seen = True
                last_activity = time.monotonic()
                yield EEGFrame(t=ts_us / 1e6, samples=ch_vals)
        finally:
            sel.close()
            self.close()

    def close(self) -> None:
        """Terminate the subprocess if running."""
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=1.0)
            except Exception:
                pass
        self._proc = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
