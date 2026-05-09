"""MW75 subprocess source. Spawns mw75-csv and parses CSV stdout."""
from __future__ import annotations

import os
import subprocess
import time
from typing import Iterator

import numpy as np

from .types import CH_NAMES, EEGFrame, MW75Unavailable


N_CH = len(CH_NAMES)


class MW75Source:
    """Iterates EEGFrames from the mw75-csv subprocess."""

    def __init__(self, binary: str = "mw75-csv", timeout_s: float = 5.0):
        self.binary = os.environ.get("MW75_CSV_BIN", binary)
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
        deadline = time.monotonic() + self.timeout_s
        first_seen = False
        try:
            while True:
                if not first_seen and time.monotonic() > deadline:
                    raise MW75Unavailable(
                        f"No EEG data within {self.timeout_s}s from {self.binary}"
                    )
                if proc.poll() is not None and not first_seen:
                    raise MW75Unavailable(
                        f"{self.binary} exited (code {proc.returncode}) before any data"
                    )
                line = proc.stdout.readline()
                if not line:
                    if proc.poll() is not None:
                        if not first_seen:
                            raise MW75Unavailable(
                                f"{self.binary} closed stdout before any data"
                            )
                        return
                    continue
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
                yield EEGFrame(t=ts_us / 1e6, samples=ch_vals)
        finally:
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
