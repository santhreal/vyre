#!/usr/bin/env bash
# P1 inventory #53  -  SPIR-V parity must be a first-class gate.
#
# The suite validates every emitted blob with spirv-val. It used to fall back to
# asserting the blob held at least five words and carried a plausible version
# word whenever that binary was absent, so a machine without the validator
# reported success for every emission and this gate proved nothing there.
#
# The validator is now required, the target is registered behind the `spirv-val`
# feature so a default `cargo test --workspace` skips it instead of running the
# old vacuous path, and this gate is the thing that enables it. SPIR-V emission
# is pure computation, so no device is involved: only the validator.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v spirv-val >/dev/null 2>&1; then
    echo "SPIR-V gate requires spirv-val, which is not on PATH." >&2
    echo "Fix: install spirv-tools (Debian/Ubuntu: apt-get install -y spirv-tools; macOS: brew install spirv-tools)." >&2
    echo "Do not skip SPIR-V validation: an unvalidated blob with a correct header is exactly what this gate catches." >&2
    exit 1
fi

spirv-val --version

source scripts/lib/cargo_runner.sh
vyre_select_cargo_runner

"$CARGO_RUNNER" test -p vyre-driver-spirv --features spirv-val --test spirv_parity -- --nocapture
