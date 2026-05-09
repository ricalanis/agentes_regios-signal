import time
from neurable_connector import EEGFrame, FS_HZ
from .conftest import FakeSource  # noqa


def test_fake_source_yields_eegframes_at_fs_hz():
    src = FakeSource(runtime_s=0.05, seed=42)  # 25 frames @ 500 Hz
    frames = list(src)
    assert all(isinstance(f, EEGFrame) for f in frames)
    assert all(f.samples.shape == (12,) for f in frames)
    assert len(frames) >= int(0.05 * FS_HZ) - 5  # tolerance
    ts = [f.t for f in frames]
    assert all(b > a for a, b in zip(ts, ts[1:]))


def test_fake_source_is_deterministic():
    a = list(FakeSource(runtime_s=0.05, seed=7))
    b = list(FakeSource(runtime_s=0.05, seed=7))
    assert len(a) == len(b)
    for fa, fb in zip(a, b):
        assert (fa.samples == fb.samples).all()
