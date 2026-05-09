# Portable focus/stress + PID-view stack

Design note for two **independent, portable** Python packages built so they can
be lifted out of this repo and dropped into another project. Neither depends on
existing `src/brain.py`, the daemon, or anything in `mw75/` Rust code beyond the
`mw75-csv` binary as a runtime data source.

Date: 2026-05-09.

## Two packages, one composing example

```
breakneurable/
├── pidview/                       ← System 2 (generic, no EEG knowledge)
│   ├── pyproject.toml             dep: numpy
│   ├── src/pidview/
│   └── tests/
├── neurable_connector/            ← System 1 (MW75 → focus, stress)
│   ├── pyproject.toml             deps: numpy, scipy
│   ├── src/neurable_connector/
│   └── tests/
└── examples/
    └── focus_stress_pid.py        wires both, proves they compose
```

Hard rule: **`neurable_connector` does not import `pidview`.** They compose at
the consumer level (the example), not via imports.

## System 1 — `neurable_connector`

Reads MW75 EEG and emits `(t, focus, stress)` at ~4 Hz.

### Public surface

```python
from neurable_connector import (
    NeurableConnector, MW75Source, Source, EEGFrame,
    AffectSample, FocusStressSample, Baseline, MW75Unavailable,
)
# `FocusStressSample` is a re-export alias of `AffectSample` for legacy code.

# Default usage — spawns mw75-csv subprocess, calibrates, streams forever.
async with NeurableConnector() as nc:
    baseline = nc.calibrate_baseline(duration_s=180.0)   # eyes-open + eyes-closed
    async for s in nc.stream():
        print(s.t, s.focus, s.stress, s.valence, s.arousal,
              s.joy, s.calm, s.excitement, s.neutral)
```

### Dataclasses

```python
@dataclass(frozen=True)
class EEGFrame:
    t: float                       # unix seconds
    samples: np.ndarray            # shape (12,) float64, channel order = CH_NAMES

@dataclass(frozen=True)
class AffectSample:
    t: float
    # signed z-scores
    focus: float                   # within-subject z (negative=low focus)
    stress: float                  # within-subject z (positive=high stress)
    valence: float                 # +ve = right > left posterior alpha
    arousal: float                 # +ve = high beta/alpha ratio
    # non-negative intensity scores (0=absent, larger=stronger)
    joy: float
    calm: float
    excitement: float
    neutral: float                 # in [0, 1]; peaks when others are quiet
    features: dict[str, float]     # raw band powers + HFD + asym + b/a, debug only

# Legacy alias for source compatibility — same dataclass.
FocusStressSample = AffectSample

CH_NAMES = ("FT7","T7","TP7","CP5","P7","C5","FT8","T8","TP8","CP6","P8","C6")
FS_HZ = 500
```

### Source protocol

`Source` is a Protocol that yields `EEGFrame`s. The default implementation
`MW75Source(binary="mw75-csv", timeout_s=5.0)` spawns the Rust subprocess and
parses one line per packet (format is `ts_us,counter,ch1..ch12`, log on stderr).
Anyone can plug in another source by implementing the protocol.

`MW75Source` raises `MW75Unavailable` if the subprocess exits without producing
data within `timeout_s` — this is what the test fixture catches to skip.

### Pipeline (per 1-s window, 0.25-s hop → 4 Hz output)

1. High-pass 1 Hz (4th-order Butterworth, sosfiltfilt) — kills DC drift.
2. Notch 60 Hz (Q=30, iirnotch + filtfilt) — kills mains hum.
3. Welch PSD per channel: `nperseg=min(512, len)`.
4. Band powers via trapezoid on PSD: θ 4–8, α 8–13, β 13–30. **No γ at temporal
   sites** (MW75 sits over temporalis muscle).
5. Posterior α at {P7, P8, TP7, TP8}: log-mean.
6. Higuchi fractal dimension at {P7, P8} on the time-domain window.
7. Sensorimotor μ: Laplacian {C5−CP5, C6−CP6} band power 8–13 Hz.
8. Posterior asymmetry: `log(α[TP8]+α[P8]) - log(α[TP7]+α[P7])` — drives valence.
9. β/α ratio: mean over {CP5, P7, C5, CP6, P8, C6} of `β_i / α_i` — drives arousal.

### Scores

All features are reduced to within-subject z-scores against the eyes-open
baseline (mean μ, std σ per feature). For each window:

```
z_alpha_post = (alpha_post - μ_alpha_post) / σ_alpha_post
z_mu         = (mu_lap     - μ_mu         ) / σ_mu
z_hfd_post   = (hfd_post   - μ_hfd_post   ) / σ_hfd_post

focus  = -z_alpha_post + 0.5 * (-z_mu)        # both are suppression effects
stress = -z_alpha_post + 0.5 *  z_hfd_post    # α decrease + complexity rise
```

Sign rules: alpha posterior z is negated because both focus (visual/attentional
engagement) and stress (Vanhollebeke 2022, g≈0.60) appear as α *decreases*. μ
suppression at C5/C6 Laplacian sites also drops with motor/attentional
engagement. Posterior HFD rises with stress (Kawe 2019).

Note that focus and stress share the α-decrease term — they are not orthogonal.
This is honest: temporal/parietal-only montage cannot fully separate them.
Differentiation comes from the second terms (μ vs HFD) and from the differential
view in `pidview` (focus tends to ramp up; stress tends to plateau or accumulate).

Honest expectation: within-subject r ≈ 0.30–0.50 vs self-reported stress;
within-subject AUC ~0.75–0.90 for focus. Trends, not thresholds.

#### Affective extension (2026-05-09)

Two new latents and four label intensities derive from existing band powers
plus the two new features. All formulas are pure, side-effect-free.

```
z_asym = (posterior_asymmetry - μ_asym) / σ_asym
z_ba   = (beta_alpha_ratio    - μ_ba)   / σ_ba

valence    = z_asym
arousal    = z_ba

joy        = relu(valence) * gauss(arousal, mu=0.5, sigma=1.0)
calm       = relu(valence) * relu(-arousal)
excitement = gauss(valence, mu=0.5, sigma=1.5) * relu(arousal)
neutral    = exp( -(focus^2 + stress^2 + joy^2 + calm^2 + excitement^2) / 5 )
```

Where `relu(x) = max(0, x)` and `gauss(x, mu, sigma) = exp(-((x-mu)/sigma)^2)`.

`neutral` is a Gaussian collapse of the other intensities — operational, not
empirical. It is bounded in `[0, 1]` and peaks when all other channels are
quiet.

Honest expectation for the new dimensions (montage-corrected — see
`research-affective-extension.md`): within-subject r ≈ 0.15–0.30 for valence
vs self-report; within-subject AUC ≈ 0.65–0.75 for binary arousal.
Cross-subject label classification ≈ 65–75%. Trends, not thresholds.

### Baseline

3 min eyes-open + 3 min eyes-closed rest. `Baseline.fit(frames, eo_window,
ec_window)` returns means and stds for each feature. `Baseline.save(path)` /
`Baseline.load(path)` persists JSON to `~/.breakneurable/baseline.json` by
default; path is overridable.

### Tests

- `tests/test_pipeline.py` — synthetic numpy frames (sine waves at known
  frequencies). Validates band-power math, HFD, μ Laplacian. Runs anywhere, no
  device.
- `tests/test_scores.py` — given features and a baseline, the score formulas
  produce expected signs and z magnitudes.
- `tests/test_live_mw75.py` — single live test that constructs `MW75Source`,
  pulls 5 seconds, asserts data is plausible (no NaN, sample rate ≈ 500 Hz).
  Auto-skips on `MW75Unavailable`. This is the ONLY live test; the helper is
  the source itself.
- `tests/conftest.py` — `live_mw75` fixture wrapping `MW75Source` with skip.

## System 2 — `pidview`

Generic. Does not know EEG exists. Any scalar time-series goes in.

### Public surface

```python
from pidview import SignalView, SignalRegistry, Snapshot

view = SignalView("focus", history_seconds=600.0,
                  integral_tau=60.0, differential_window_seconds=2.0)
view.push(t, x)
snap = view.snapshot()           # Snapshot(name, t, present, differential, integral, history, stats)

reg = SignalRegistry()
reg.register("focus", history_seconds=600.0)
reg.register("stress", history_seconds=600.0)
reg.push("focus", t, x)
reg.snapshot_all()               # dict[str, Snapshot]
```

### Snapshot

```python
@dataclass(frozen=True)
class Snapshot:
    name: str
    t: float
    present: float                # P — current x
    differential: float           # D — slope over last `differential_window_seconds`
    integral: float               # I — leaky integrator (or true integral if tau=None)
    history: np.ndarray           # H — shape (N, 2) of (t, x), oldest first
    stats: dict[str, float]       # mean, std, p10, p50, p90, slope (longer window)
```

### Semantics

- **Present:** the most recent pushed value. None if no samples.
- **Differential:** linear-fit slope (least squares) over samples within the
  last `differential_window_seconds`. Robust to noise vs single-step diff.
  Returns 0.0 if fewer than 2 samples in window.
- **Integral:** leaky integrator with time constant `integral_tau` seconds.
  Continuous form: `dI/dt = x - I/tau`. Discrete update on each push using
  actual dt between samples (handles irregular sampling). If `integral_tau is
  None`, accumulate trapezoidal integral with no decay (useful but unbounded).
- **History:** ring buffer keyed by time; samples older than `history_seconds`
  are dropped on push. Returned as a `(N, 2)` ndarray copy.
- **Stats:** computed over the current history. Cheap (numpy aggregates).

### Pub/sub

`view.subscribe(fn)` registers a callback invoked synchronously after each
`push`, receiving the fresh `Snapshot`. Returns an unsubscribe callable.

### Tests

- Push known sequences (constant, ramp, sinusoid) and assert P, D, I, H values
  match closed-form expectations to within float tolerance.
- Subscriber invoked exactly once per push, in order.
- History eviction is correct under irregular sampling.
- Leaky integrator decays with τ as predicted.

## The composing example

`examples/focus_stress_pid.py`:

```python
import asyncio
from neurable_connector import NeurableConnector
from pidview import SignalRegistry

async def main():
    reg = SignalRegistry()
    reg.register("focus", history_seconds=600.0, integral_tau=60.0)
    reg.register("stress", history_seconds=600.0, integral_tau=60.0)

    async with NeurableConnector() as nc:
        nc.calibrate_baseline(duration_s=180.0)
        async for s in nc.stream():
            reg.push("focus",  s.t, s.focus)
            reg.push("stress", s.t, s.stress)
            snaps = reg.snapshot_all()
            print(
                f"focus  P={snaps['focus'].present:+.2f}  "
                f"D={snaps['focus'].differential:+.3f}/s  "
                f"I={snaps['focus'].integral:+.2f}  "
                f"| stress P={snaps['stress'].present:+.2f}  "
                f"D={snaps['stress'].differential:+.3f}/s  "
                f"I={snaps['stress'].integral:+.2f}"
            )

asyncio.run(main())
```

## Out of scope (do not build)

- HTTP/WebSocket server (left as future opt-in module — the libraries stay
  pure).
- IMU / PPG (MW75 lacks them).
- Cross-subject normalization or pretrained models.
- Recording-to-file storage (the existing `src/daemon/` already does that).
- Any UI.

## Subagent dispatch plan

- **Agent A**: build `pidview/` end-to-end with TDD. Pure numpy. No external
  systems. Returns when `pytest` is green and the public API matches this doc.
- **Agent B**: build `neurable_connector/` end-to-end. Offline tests via
  synthetic numpy. The single live test auto-skips when no headset. Returns
  when offline `pytest` is green and the live test imports cleanly.
- **Agent C** (after A and B): write `examples/focus_stress_pid.py` and the two
  package READMEs, and run a final import-level smoke check.

A and B run truly in parallel; the only contract between them is the
dataclasses defined in this document.
