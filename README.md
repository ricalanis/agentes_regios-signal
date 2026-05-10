# agentes_regios

Portable Python stack for turning a Master & Dynamic MW75 Neuro EEG headset
into an 8-dimensional affective signal — focus, stress, valence, arousal, joy,
calm, excitement, neutral — and viewing each dimension through a generic
PID-style lens (Present / Differential / History / Integral).

Two independent packages, no shared code:

| Package              | Role                                       | Deps          |
|----------------------|--------------------------------------------|---------------|
| `neurable_connector` | MW75 EEG → 8 affective z-scores at 4 Hz    | numpy, scipy  |
| `pidview`            | Generic P/D/H/I view over any scalar stream| numpy         |

They compose at the consumer level (`examples/focus_stress_pid.py`). Either can
be lifted out and used standalone in another project — `cp -r pidview` or
`cp -r neurable_connector` is enough.

## Quickstart

### Simulated (no hardware needed)

```bash
# one shared venv for the example
python3 -m venv .venv
source .venv/bin/activate
pip install -e ./pidview -e ./neurable_connector

python examples/focus_stress_pid.py --simulated --max-seconds 5
```

You'll see a header on stderr and ~20 lines on stdout, e.g.:

```
+0.50s  F=+2.69/+4.51/+1.04  S=+3.16/+4.37/+1.51  v=+1.47 a=-0.74  | J=0.32 C=1.08 E=0.00 N=0.02
```

`F`/`S` columns show **P/D/I** (present, differential per second, leaky
integral with τ=60 s). `v`/`a` are the latent valence/arousal z-scores.
`J`/`C`/`E`/`N` are the four affective intensities.

### Real headset

The `mw75-csv` binary is built from a small Rust wrapper crate that lives
inside this repo at `neurable_connector/native/`. Build it once:

```bash
bash neurable_connector/native/build.sh
```

This installs `neurable_connector/native/bin/mw75-csv`. Python's `MW75Source`
discovers it automatically — no `PATH` edits or env vars needed. First build
takes a few minutes (cargo fetches ~290 transitive crates from github.com).

Then run:

```bash
python examples/focus_stress_pid.py
```

Calibration takes ~3 minutes (90 s eyes open, 90 s eyes closed). After that
the stream prints one line every 250 ms.

**Requires:** Rust toolchain (https://rustup.rs), macOS or Linux with
Bluetooth, and a paired MW75 headset.

**License note:** the binary statically links the GPL-3.0 upstream
`eugenehp/mw75` library, so `mw75-csv` is GPL-3.0. The Python wrapper stays
MIT — it only spawns the binary as a subprocess. See
`neurable_connector/native/README.md` for details.

## Output contract

### Connector — `AffectSample` (one per 250 ms)

```json
{
  "t": 1778364427.578,
  "focus": -0.11,
  "stress": -1.91,
  "valence": -1.96,
  "arousal":  0.75,
  "joy": 0.00,
  "calm": 0.00,
  "excitement": 0.05,
  "neutral": 0.48,
  "features": { "...raw band powers, HFD, asymmetry, β/α ratio for debugging..." }
}
```

The 9 typed fields are the contract. `features` is a debug dict — its keys may
change as the pipeline evolves.

### Generic view — `pidview.Snapshot` (one per signal, on every push)

```json
{
  "name": "joy",
  "t": 1778364427.578,
  "present": 0.0,
  "differential": -0.12,
  "integral": 0.45,
  "history": [[t, x], ...],
  "stats": { "mean": 0.23, "std": 0.50, "p10": 0, "p50": 0, "p90": 0.68, "slope": -0.12 }
}
```

`history` is a numpy `(N, 2)` array of `[t, x]`, oldest first, kept within
`history_seconds` of the latest push. `stats` is computed over that window.

## The eight dimensions

| Dim          | Sign     | Range            | Means…                                            |
|--------------|----------|------------------|---------------------------------------------------|
| `focus`      | signed z | typically ±3     | Posterior α desync + sensorimotor μ suppression   |
| `stress`     | signed z | typically ±3     | Posterior α desync + posterior HFD complexity     |
| `valence`    | signed z | typically ±3     | Right > left posterior α (positive = positive)    |
| `arousal`    | signed z | typically ±3     | Central+parietal β/α ratio (positive = aroused)   |
| `joy`        | ≥ 0      | 0 to ~3          | Positive valence + arousal near +0.5 z            |
| `calm`       | ≥ 0      | 0 to ~3          | Positive valence + low arousal                    |
| `excitement` | ≥ 0      | 0 to ~3          | Non-negative valence + high arousal               |
| `neutral`    | [0, 1]   | peaks at 1       | Mathematical complement; high when others are quiet |

`focus` and `stress` are the original signals. `valence` and `arousal` are the
new latents. `joy`/`calm`/`excitement` are deterministic functions of those
latents (see `docs/research-affective-extension.md` for formulas + citations).
`neutral` is `exp(-(F² + S² + J² + C² + E²) / 5)` — operational, not empirical.

## Honest performance expectations

MW75 has **no frontal/midline electrodes**. The classical affect markers
(Frontal Alpha Asymmetry, frontal-midline θ) are not available. Everything
here uses temporal/sensorimotor/parietal-validated literature only, which has
smaller effects.

Within-subject:
- focus AUC ≈ 0.75–0.90 (binary)
- stress r ≈ 0.30–0.50 vs self-report
- valence r ≈ 0.15–0.30 vs self-report
- arousal AUC ≈ 0.65–0.75 (binary)

Cross-subject any-label classification ≈ 65–75%. **Trends, not thresholds.**

## Layout

```
agentes_regios/
├── README.md                              ← you are here
├── docs/
│   ├── portable-stack-design.md           binding spec for both packages
│   └── research-affective-extension.md    literature + formulas + citations
├── pidview/                               package 1 — generic P/D/H/I view
│   ├── README.md, LICENSE, pyproject.toml
│   ├── src/pidview/
│   └── tests/                             pytest, 22 passing
├── neurable_connector/                    package 2 — MW75 → 8-dim affect
│   ├── README.md, LICENSE, pyproject.toml
│   ├── src/neurable_connector/
│   └── tests/                             pytest, 30 passing + 1 live (auto-skip)
└── examples/
    └── focus_stress_pid.py                composes both, --simulated for headset-free
```

## Non-goals

- No HTTP/WebSocket server (libraries stay pure; build your own transport).
- No model training or cross-subject normalization.
- No IMU / PPG (MW75 lacks them).
- No recording-to-disk (consume the stream and persist however you like).
- No UI.

## Sharp edges to know about

- **MW75Source is single-shot per `__iter__`** — build a fresh one per session.
- **Out-of-order timestamps** in `pidview` raise `ValueError`. Push monotonic time.
- **Leaky integrator is forward-Euler** — fine for streaming at multi-Hz / sub-Hz dt;
  for very sparse pushes (dt > τ) it can ring. Pick a smaller τ or push more often.
- **Baseline is per-session, per-subject.** Cross-subject z-scores are not meaningful.
- **γ band is intentionally omitted.** Temporal MW75 sites overlap temporalis muscle.

## License

MIT for both packages. See `pidview/LICENSE` and `neurable_connector/LICENSE`.
