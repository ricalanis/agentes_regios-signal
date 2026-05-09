# pidview

A small, dependency-light view layer for scalar time series. You push `(t, x)`
samples; you get back a `Snapshot` with the present value (P), a short-window
slope (D), a leaky integral (I), and a bounded history buffer (H). Pure numpy,
no domain knowledge baked in. Designed to sit in front of any signal you want
to track over time without dragging in plotting, storage, or threading
machinery.

## Install

From the package directory:

```
pip install -e ".[dev]"     # editable install with pytest
pip install .               # runtime only
```

Requires Python 3.11+ and numpy.

## Usage

```python
import time
from pidview import SignalRegistry

reg = SignalRegistry()
reg.register("focus", history_seconds=600.0, integral_tau=60.0)
reg.register("stress", history_seconds=600.0, integral_tau=60.0)

# Wire a subscriber on one view; runs synchronously after each push.
def log_focus(snap):
    print(f"focus P={snap.present:+.2f} D={snap.differential:+.3f}/s I={snap.integral:+.2f}")

reg.get("focus").subscribe(log_focus)

# Push samples as they arrive. Timestamps are unix seconds (float).
t0 = time.time()
for i in range(20):
    t = t0 + 0.25 * i
    reg.push("focus",  t,  0.10 * i)              # ramp
    reg.push("stress", t, -0.05 * i + 0.2)        # downward ramp
    time.sleep(0.01)

snaps = reg.snapshot_all()
focus = snaps["focus"]
print("history shape:", focus.history.shape)     # (N, 2): columns are (t, x)
print("p50 over window:", focus.stats["p50"])
```

## Public API

### `SignalView`

```
SignalView(name, history_seconds=600.0, integral_tau=60.0,
           differential_window_seconds=2.0)
```

- `push(t, x)` — append a sample, update the integral, evict samples older than
  `history_seconds`, and notify subscribers. `t` must be non-decreasing across
  calls.
- `snapshot()` — return a `Snapshot` of the current state (cheap; numpy
  aggregates over the live history).
- `subscribe(fn)` — register a callback `fn(snapshot)` invoked synchronously
  after every successful `push`. Returns an unsubscribe callable.

### `SignalRegistry`

A dict-like container of named `SignalView`s.

- `register(name, **view_kwargs) -> SignalView` — create and store a view.
  Raises if `name` is already registered.
- `get(name) -> SignalView` — fetch by name (raises `KeyError` if missing).
- `push(name, t, x)` — forward a sample to the named view.
- `snapshot_all() -> dict[str, Snapshot]` — snapshot every registered view.
- `names() -> list[str]` — registered names in insertion order.
- `name in registry` — membership test.

### `Snapshot`

Frozen dataclass:

- `name: str`
- `t: float` — timestamp of the latest sample.
- `present: float` — most recent x.
- `differential: float` — least-squares slope over the last
  `differential_window_seconds`. `0.0` if fewer than 2 samples in window.
- `integral: float` — leaky integral with time constant `integral_tau`. If
  `integral_tau is None`, this is a trapezoidal accumulation with no decay.
- `history: np.ndarray` — `(N, 2)` array of `(t, x)`, oldest first. Copy of the
  internal buffer.
- `stats: dict[str, float]` — `mean`, `std`, `p10`, `p50`, `p90`, `slope`
  (linear fit over the entire current history).

### Defaults

- `history_seconds = 600.0` (10 minutes)
- `integral_tau = 60.0` seconds (set `None` for unbounded trapezoidal accum)
- `differential_window_seconds = 2.0`

## Notes and sharp edges

- The leaky integrator uses forward-Euler discretization
  (`I += dt * (x - I/tau)`). Stable and accurate for typical streaming rates,
  from multi-Hz down to sub-Hz `dt`. At very sparse sampling (`dt > tau`) the
  response can ring or overshoot. Pick a smaller `tau`, or push at a finer
  cadence.
- Out-of-order timestamps raise `ValueError`. Push in monotonic time. If you
  have multiple producers, sort or buffer upstream before pushing.
- `subscribe` callbacks run synchronously inside `push`. Do not block in them.
  Exceptions raised by a subscriber are caught and logged to stderr; they do
  not interrupt the push.
- `Snapshot.history` is a copy. Mutating it does not affect the view.
- `integral_tau=None` switches to trapezoidal accumulation (no decay). The
  result is unbounded; use it only when you want a true running integral.

## Test

```
pytest -q
```

## License

MIT. See `LICENSE`.
