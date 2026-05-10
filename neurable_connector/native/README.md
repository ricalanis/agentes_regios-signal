# neurable_connector / native

Small Rust wrapper crate that builds the `mw75-csv` binary that the Python
`MW75Source` spawns to stream EEG from the Master & Dynamic MW75 Neuro
headset (BLE activation + RFCOMM data).

## Quick build

```bash
bash native/build.sh
```

This runs `cargo install --path . --root . --force` which produces
`native/bin/mw75-csv`. Python's `MW75Source` discovers this path
automatically (no env var or PATH edits needed).

First build downloads the upstream `mw75` crate from github.com and compiles
its dependency tree (~290 crates, a few minutes). Subsequent builds are
incremental.

## What's here

- `Cargo.toml` — declares the crate, dep on
  `mw75 = { git = "https://github.com/eugenehp/mw75", features = ["rfcomm"] }`
- `src/main.rs` — the wrapper. Connects to MW75 over BLE, activates EEG,
  switches to RFCOMM, and writes one CSV line per EEG packet to stdout
  (`ts_us,counter,ch1,ch2,...,ch12`). Logs go to stderr.
- `build.sh` — convenience wrapper around `cargo install`.

## License

| Part                                | License    |
|-------------------------------------|------------|
| The wrapper source (`src/main.rs`)  | GPL-3.0    |
| The compiled `bin/mw75-csv` binary  | GPL-3.0    |
| Upstream `mw75` library it links    | GPL-3.0    |
| The Python `neurable_connector`     | MIT        |

The wrapper source links the GPL upstream library statically, so the
resulting binary is GPL-3.0. The Python package interacts with the binary
only by spawning it as a subprocess and reading stdout — this is "mere
aggregation" per the [GPL FAQ](https://www.gnu.org/licenses/gpl-faq.en.html#MereAggregation),
so the Python source can stay MIT.

If you redistribute the built binary (e.g. inside a packaged application),
you must comply with GPL-3.0: provide source for the wrapper, attribute the
upstream `mw75` crate, and offer the same to recipients. The full GPL-3.0
text is at <https://www.gnu.org/licenses/gpl-3.0.html>.
