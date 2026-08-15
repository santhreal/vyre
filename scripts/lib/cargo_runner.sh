#!/usr/bin/env bash
# Shared cargo runner selection for release, CI, and benchmark gates.
#
# The workspace prefers `./cargo_full` when it is available, but release
# scripts must still be executable in checkouts where the wrapper is absent.
# In that case they fall back to `cargo`. Job count and target directory are
# declared once in `.cargo/config.toml`, so this file sets neither: a runner
# that exported its own job count built a different build than a bare cargo
# invocation in the same checkout, and defeated the shared compilation cache.

vyre_select_cargo_runner() {
    if [[ -n "${VYRE_CARGO_RUNNER:-}" ]]; then
        CARGO_RUNNER="$VYRE_CARGO_RUNNER"
    elif [[ -x ./cargo_full ]]; then
        CARGO_RUNNER="./cargo_full"
    else
        CARGO_RUNNER="cargo"
    fi
    export CARGO_RUNNER
}
