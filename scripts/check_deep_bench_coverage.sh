#!/usr/bin/env bash
# Every benchmark case a tracked file names must still be registered, and every
# measured dimension must have a registered case.
#
# The canonical harness is the vyre-bench registry, not scattered Criterion
# targets. This gate reads the registry and holds two things to it:
#
#   - Each dimension below names a representative case. A dimension whose case
#     was renamed or deleted stops being measured, and the harness says nothing
#     because a registry with fewer cases is still a valid registry.
#   - Every `--case <id>` in a tracked file must resolve to a registered case.
#     Most of those references live in gpu-parity.yml, which only runs on the
#     self-hosted GPU runner, and in release evidence manifests that only run at
#     release time. A rename breaks them where nobody is watching; here it
#     breaks in PR CI, on a runner that needs no GPU because listing the
#     registry measures nothing.
#
# The gate no longer writes the registry to a mktemp file outside the
# repository, and a cargo failure is fatal instead of an empty scan.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/lib/cargo_runner.sh
vyre_select_cargo_runner

python3 "$ROOT/scripts/lib/check_deep_bench_coverage.py" "$ROOT" "$CARGO_RUNNER"
