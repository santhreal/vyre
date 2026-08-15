#!/usr/bin/env bash
# Every tracked .rs file must be valid Rust syntax.
#
# The class this closes: a tracked .rs file that no target compiles is never
# parsed by anything, so it can hold text that is not Rust at all and every
# build stays green. `examples/libs-template/tests/cat_a_conform.rs` is the
# proof: it is Liquid, not Rust, and only the scaffolding template renders it.
# Reachability alone does not catch that class, because the answer for an
# orphan is "no target", and the answer for a template is also "no target".
# This gate answers the other half: whatever a file is declared to be, it must
# at least parse.
#
# This is a PARSE gate, not a format gate. rustfmt is used as the parser
# because it is the only parser shipped with the toolchain that reads one file
# without a crate around it. Formatting differences print as `Diff in <path>`
# and are ignored; only a line beginning with `error` fails this gate. Nothing
# is written: --check is read-only, and no `--emit` is ever passed.
#
# rustfmt parses the child modules a file declares, so a broken file is reported
# alongside its parents up to the crate root. Every one of those lines names a
# real path and one fix clears all of them.
#
# rustfmt is required, not optional. `2>/dev/null || true` around a missing
# binary makes "no toolchain" and "clean tree" the same result, so a missing
# rustfmt is fatal here.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v rustfmt >/dev/null 2>&1; then
    echo "check-every-source-file-parses: rustfmt is not installed." >&2
    echo "    Fix: install the rustfmt component (rustup component add rustfmt). This gate" >&2
    echo "    parses every tracked .rs file with it, and a missing parser is not a clean tree." >&2
    exit 1
fi

python3 "$ROOT/scripts/lib/check_every_source_file_parses.py" "$ROOT"
