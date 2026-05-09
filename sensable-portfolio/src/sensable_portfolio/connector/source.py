"""Build a NeurableConnector with a real or synthetic Source."""
from __future__ import annotations

from neurable_connector import Baseline, FS_HZ, MW75Source, NeurableConnector, Source


def build_connector(source: Source, baseline: Baseline) -> NeurableConnector:
    return NeurableConnector(source=source, baseline=baseline)


def calibrate_baseline(source: Source) -> Baseline:
    return Baseline.fit(list(source), fs=float(FS_HZ))


def production_source() -> Source:
    """The real device. Tests use FakeSource via conftest; runtime uses MW75Source."""
    return MW75Source()
