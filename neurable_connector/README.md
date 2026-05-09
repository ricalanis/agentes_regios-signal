# neurable_connector

Reads EEG from a Master & Dynamic MW75 Neuro headset and emits
`(t, focus, stress)` samples at ~4 Hz, where focus and stress are
within-subject z-scores against an eyes-open baseline. Pure-Python pipeline
(numpy + scipy), async streaming API, no model weights. The package owns the
MW75-specific signal processing; downstream views, plots, and storage are
intentionally somebody else's problem.

## Hard constraint: temporal/parietal-only montage

The MW75 has no frontal or midline electrodes. The 12 channels sit over
temporal, sensorimotor, and parietal sites
(`FT7, T7, TP7, CP5, P7, C5, FT8, T8, TP8, CP6, P8, C6`). Every marker computed
here is validated for those locations:

- Posterior alpha at `P7, P8, TP7, TP8` (visual/attentional engagement and
  stress).
- Sensorimotor mu at the `C5−CP5` and `C6−CP6` Laplacian channels (motor and
  attentional engagement).
- Higuchi fractal dimension at `P7, P8` (signal complexity, rises with stress).

The gamma band is intentionally omitted: the temporal sites overlap the
temporalis muscle, and gamma readings there are dominated by EMG.

Honest expectation, in line with the published literature on this kind of
montage: within-subject `r ≈ 0.30–0.50` against self-reported stress, and
within-subject `AUC ≈ 0.75–0.90` for focus. These numbers describe trends, not
thresholds. Cross-subject comparisons of focus or stress are not meaningful;
the scores are within-subject z-scores by construction.

## Install

From the package directory:

```
pip install -e ".[dev]"     # editable install with pytest
pip install .               # runtime only
```

Requires Python 3.11+, numpy, and scipy.

## Hardware setup

You need the `mw75-csv` binary (the Rust subprocess that owns the BLE link to
the headset and prints CSV samples on stdout). Either:

- Put `mw75-csv` on your `PATH`, or
- Set `MW75_CSV_BIN=/path/to/mw75-csv` in the environment.

Build the binary from upstream
[`eugenehp/mw75`](https://github.com/eugenehp/mw75) (cargo crate). On the
machine this package was developed on, a pre-built binary lives at
`/Users/ricalanis/Dev/breakneurable/mw75/target/release/mw75-csv` — that path is
shown only as an example, not a requirement of the package.

If the subprocess cannot be launched, exits before producing any data, or stays
silent past `timeout_s`, `MW75Source` raises `MW75Unavailable`.

## Quickstart

```python
import asyncio
from neurable_connector import NeurableConnector

async def main():
    async with NeurableConnector() as nc:
        # Eyes-open then eyes-closed rest, 90 s each by convention.
        nc.calibrate_baseline(duration_s=180.0)
        async for s in nc.stream():
            print(f"t={s.t:.2f}  focus={s.focus:+.2f}  stress={s.stress:+.2f}")

asyncio.run(main())
```

To run offline against a custom Source (e.g. a recorded file), pass
`source=...` and `baseline=...` to `NeurableConnector` directly:

```python
from neurable_connector import Baseline, NeurableConnector

baseline = Baseline.load()  # ~/.breakneurable/baseline.json
nc = NeurableConnector(source=my_iterable_of_eegframes, baseline=baseline)
```

## Public API

### `NeurableConnector`

```
NeurableConnector(source=None, baseline=None, fs=500.0,
                  window_s=1.0, hop_s=0.25)
```

- Async context manager: `async with NeurableConnector() as nc: ...`. Releases
  the underlying source on exit.
- `calibrate_baseline(duration_s=180.0) -> Baseline` — pulls frames from the
  source synchronously, fits per-feature mean/std, and stores it on `self`.
- `stream() -> AsyncIterator[FocusStressSample]` — yields one sample every
  `hop_s` seconds (4 Hz with defaults). Raises `RuntimeError` if no baseline
  is set.
- `close()` — terminate the underlying source.

### `MW75Source`

```
MW75Source(binary="mw75-csv", timeout_s=5.0)
```

Iterating spawns the subprocess and yields `EEGFrame`s parsed from
`ts_us,counter,ch1..ch12` CSV lines on stdout. Reads `MW75_CSV_BIN` from the
environment to override `binary`. `close()` terminates the subprocess.

### `Source` (Protocol)

```python
class Source(Protocol):
    def __iter__(self) -> Iterator[EEGFrame]: ...
```

Anything that yields `EEGFrame`s is a valid source. Recorded files, fakes,
synthetic generators, network sources — implement the protocol.

### Dataclasses, constants, exceptions

- `EEGFrame(t: float, samples: np.ndarray)` — frozen. `samples` is shape
  `(12,)` float64, channel order = `CH_NAMES`.
- `FocusStressSample(t, focus, stress, features)` — frozen. `focus` and
  `stress` are within-subject z-scores; `features` is the raw feature dict
  used to compute them (debug only).
- `Baseline` — `means` and `stds` per feature key. `Baseline.fit(frames, fs,
  window_s, hop_s)`, `Baseline.save(path=None)`, `Baseline.load(path=None)`.
  Default path is `~/.breakneurable/baseline.json`.
- `MW75Unavailable(RuntimeError)` — raised by `MW75Source` when the subprocess
  cannot deliver data.
- `CH_NAMES: tuple[str, ...]` — the 12 channels in order.
- `FS_HZ: int = 500` — MW75 sample rate.

Helper functions `compute_focus(features, baseline)` and
`compute_stress(features, baseline)` are also exported for offline scoring of
pre-computed features.

## Notes and sharp edges

- `MW75Source` is single-shot per iteration. Calling `iter()` on it spawns the
  subprocess; once the iterator is exhausted (or `close()` is called), the
  source is done. Build a fresh `MW75Source` for a new session — re-iterating
  an exhausted one will not produce data.
- The gamma band is intentionally omitted because the temporal electrodes sit
  over the temporalis muscle. Any "gamma" you would compute there is mostly
  EMG.
- Score formulas are within-subject z-scores against the eyes-open baseline.
  They share the `−z(posterior_alpha)` term by design — the temporal/parietal
  montage cannot fully separate focus and stress; differentiation comes from
  the second term (`mu_lap` for focus, `posterior_hfd` for stress) and from
  watching trajectories over time. Cross-subject comparisons are not
  meaningful.
- The live test in `tests/test_live_mw75.py` auto-skips when no headset is
  reachable (`MW75Unavailable`), so the test suite is green on machines
  without hardware.
- `calibrate_baseline` is synchronous and blocks the event loop. That is
  intentional: the baseline is a one-shot setup step, not a streaming
  operation.

## Test

```
pytest -q
```

Offline tests pass without any hardware. The single live test
(`tests/test_live_mw75.py`) auto-skips when the MW75 subprocess cannot produce
data.

## License

MIT. See `LICENSE`.
