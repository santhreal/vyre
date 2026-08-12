#!/usr/bin/env bash
# Compatibility entry point for the canonical primitive-admission gate.
#
# Primitive admission is derived from the semantic operation inventory and
# docs/optimization/PRIMITIVE_ADMISSION.toml. Source-file shape is not a
# primitive contract, so this adapter intentionally accepts no file paths.

set -euo pipefail

if [[ "$#" -ne 0 ]]; then
    echo "FAIL: path-scoped primitive checks are obsolete." >&2
    echo "Fix: register the operation and its real composition edges, or add an owner-reviewed family exception to docs/optimization/PRIMITIVE_ADMISSION.toml." >&2
    exit 2
fi

exec cargo run --quiet -p xtask --bin xtask -- primitive-admission-gate
