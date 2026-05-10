#!/usr/bin/env bash
# Build the mw75-csv binary into ./bin/mw75-csv (relative to this script).
#
# This is a small Rust wrapper crate that depends on the upstream
# https://github.com/eugenehp/mw75 library (GPL-3.0). The wrapper source
# (src/main.rs) is your own; the resulting binary is GPL-3.0 because it
# statically links the GPL upstream library.
#
# The Python neurable_connector package remains MIT — it spawns this binary
# as a subprocess only ("mere aggregation" per the GPL FAQ; the two are not
# linked).
#
# Requirements:
#   - Rust toolchain (https://rustup.rs)
#   - Internet (cargo fetches the upstream mw75 crate from github.com)
#   - macOS or Linux (RFCOMM feature)
set -euo pipefail
cd "$(dirname "$0")"

cargo install --path . --root . --force

echo
echo "Built: $(pwd)/bin/mw75-csv"
ls -la bin/mw75-csv
