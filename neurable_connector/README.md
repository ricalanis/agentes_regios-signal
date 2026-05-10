# neurable_connector

Reads EEG from a Master & Dynamic MW75 Neuro headset and emits an
`AffectSample` at ~4 Hz with 8 dimensions: signed z-scores for `focus`,
`stress`, `valence`, `arousal`, plus non-negative intensities for `joy`,
`calm`, `excitement`, and `neutral`. Pure-Python pipeline (numpy + scipy),
async streaming API, no model weights. Self-contained — bring this directory
into any Python project, build the native binary once, and you're streaming.

## Hard constraint: temporal/parietal-only montage

The MW75 has no frontal or midline electrodes. The 12 channels sit over
temporal, sensorimotor, and parietal sites
(`FT7, T7, TP7, CP5, P7, C5, FT8, T8, TP8, CP6, P8, C6`). Every marker computed
here is validated for those locations:

- Posterior alpha at `P7, P8, TP7, TP8` (visual/attentional engagement; stress).
- Posterior alpha asymmetry `log(α_P8+α_TP8) − log(α_P7+α_TP7)` (valence).
- Sensorimotor mu at the `C5−CP5` and `C6−CP6` Laplacian channels (focus).
- β/α ratio at `{C5, CP5, P7, C6, CP6, P8}` (arousal).
- Higuchi fractal dimension at `P7, P8` (signal complexity, rises with stress).

The gamma band is intentionally omitted: the temporal sites overlap the
temporalis muscle, and gamma readings there are dominated by EMG.

Honest expectation, in line with the published literature on this kind of
montage: within-subject `r ≈ 0.30–0.50` for stress vs self-report,
`AUC ≈ 0.75–0.90` for focus, weaker numbers for valence (`r ≈ 0.15–0.30`) and
arousal (`AUC ≈ 0.65–0.75`). Cross-subject any-label classification ≈ 65–75%.
Trends, not thresholds. See `docs/research-affective-extension.md` for the full
literature breakdown.

## Install

```
pip install -e ".[dev]"     # editable install with pytest
pip install .               # runtime only
```

Requires Python 3.11+, numpy, and scipy.

## Hardware setup — build the native binary

The Python `MW75Source` spawns a Rust binary (`mw75-csv`) that owns the BLE
activation and RFCOMM stream to the headset. A small wrapper crate is
included in `native/`; build it once:

```
bash native/build.sh
```

This produces `native/bin/mw75-csv`. Python discovers this path
automatically — no `PATH` edits or env vars needed.

First build downloads the upstream
[`eugenehp/mw75`](https://github.com/eugenehp/mw75) crate from github.com
and compiles ~290 transitive crates (a few minutes). Subsequent builds are
incremental.

Override the binary location with `MW75_CSV_BIN=/path/to/mw75-csv` if you
want to point at a different build. If the subprocess cannot launch, exits
before producing any data, or stays silent past `timeout_s`, `MW75Source`
raises `MW75Unavailable`.

## Quickstart

```python
import asyncio
from neurable_connector import NeurableConnector

async def main():
    async with NeurableConnector() as nc:
        # Eyes-open then eyes-closed rest, 90 s each by convention.
        nc.calibrate_baseline(duration_s=180.0)
        async for s in nc.stream():
            print(
                f"t={s.t:.2f}  "
                f"F={s.focus:+.2f}  S={s.stress:+.2f}  "
                f"v={s.valence:+.2f}  a={s.arousal:+.2f}  "
                f"J={s.joy:.2f}  C={s.calm:.2f}  E={s.excitement:.2f}  N={s.neutral:.2f}"
            )

asyncio.run(main())
```

To run offline against a custom Source (e.g. a recorded file or a synthetic
generator), pass `source=...` and `baseline=...` to `NeurableConnector`
directly:

```python
from neurable_connector import Baseline, NeurableConnector

baseline = Baseline.load()
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
- `stream() -> AsyncIterator[AffectSample]` — yields one sample every
  `hop_s` seconds (4 Hz with defaults). Raises `RuntimeError` if no baseline
  is set.
- `close()` — terminate the underlying source.

### `MW75Source`

```
MW75Source(binary=None, timeout_s=5.0)
```

Iterating spawns the subprocess and yields `EEGFrame`s parsed from
`ts_us,counter,ch1..ch12` CSV lines on stdout. Binary discovery order:
`MW75_CSV_BIN` env var > package-local `native/bin/mw75-csv` > `mw75-csv`
on `PATH`. Pass `binary=` to override.

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
- `AffectSample(t, focus, stress, valence, arousal, joy, calm, excitement,
  neutral, features)` — frozen. The 9 typed fields are the contract;
  `features` is a debug dict.
- `FocusStressSample` — alias for `AffectSample`, kept for source compatibility.
- `Baseline` — per-feature `means`/`stds`. `Baseline.fit(frames, fs,
  window_s, hop_s)`, `Baseline.save(path=None)`, `Baseline.load(path=None)`.
  Default path is `~/.breakneurable/baseline.json`. Old baselines (missing
  the asymmetry/β-α stats) load with `μ=0, σ=1` defaults and a single stderr
  warning.
- `MW75Unavailable(RuntimeError)` — raised by `MW75Source` when the
  subprocess cannot deliver data.
- `CH_NAMES: tuple[str, ...]` — the 12 channels in order.
- `FS_HZ: int = 500` — MW75 sample rate.

Helper functions `compute_focus`, `compute_stress`, `compute_valence`,
`compute_arousal`, `compute_joy`, `compute_calm`, `compute_excitement`,
`compute_neutral`, and `compute_all` are exported for offline scoring of
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
  Focus and stress share the `−z(posterior_alpha)` term — the temporal/parietal
  montage cannot fully separate them; differentiation comes from the second
  term (`mu_lap` for focus, `posterior_hfd` for stress). Joy/calm/excitement
  derive deterministically from valence and arousal; neutral is the
  mathematical complement (peaks when nothing else fires).
- The live test in `tests/test_live_mw75.py` auto-skips when no headset is
  reachable (`MW75Unavailable`), so the test suite is green on machines
  without hardware.
- `calibrate_baseline` is synchronous and blocks the event loop. Intentional:
  the baseline is a one-shot setup step, not a streaming operation.

## Test

```
pytest -q
```

Offline tests pass without hardware. The single live test
(`tests/test_live_mw75.py`) auto-skips when the MW75 subprocess cannot
produce data; with the headset on and `bash native/build.sh` already run, it
exercises the real BLE+RFCOMM path.

## License

The Python package is **MIT** (see `LICENSE`).

The optional `native/` subdirectory builds a binary that statically links the
upstream [`eugenehp/mw75`](https://github.com/eugenehp/mw75) GPL-3.0 crate, so
both `native/src/main.rs` and the resulting `native/bin/mw75-csv` are
**GPL-3.0**. The Python package interacts with the binary only via subprocess
("mere aggregation" per the GPL FAQ), so the MIT license on the Python source
is intact. See `native/README.md` for details.
