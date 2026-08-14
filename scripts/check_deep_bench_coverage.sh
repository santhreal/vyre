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

python3 - "$ROOT" "$CARGO_RUNNER" <<'PY'
import json
import re
import sys
from pathlib import Path
from subprocess import run

root = Path(sys.argv[1]).resolve()
cargo = sys.argv[2]
NAME = "deep-bench-coverage"

# One representative registered case per measured dimension.
REPRESENTATIVE = {
    "throughput": "foundation.dfa_match.256k",
    "latency": "runtime.megakernel.dispatch.256",
    "memory": "primitives.graph.frontier_step.1m",
    "optimizer": "foundation.optimizer.impact",
    "runtime_queueing": "runtime.megakernel.condition.64k",
}

# compile_cache has no vyre-bench case yet; its evidence is a named driver test.
CACHE_CONTRACT = "vyre-driver-cuda/tests/module_cache_contracts.rs"
CACHE_TEST = "repeated_dispatch_reuses_loaded_cuda_module"

# `--case` inside Rust source is prose in a help or error string, never an
# invocation, so those files are not scanned.
REFERENCE_SUFFIXES = (".yml", ".yaml", ".sh", ".json", ".toml", ".md")
CASE_FLAG = re.compile(r"--case[ =]+([A-Za-z0-9_.\-]+)")


def fatal(message: str) -> None:
    sys.exit(f"{NAME}: {message}")


listing = run(
    [cargo, "run", "-q", "-p", "vyre-bench", "--", "list", "--format", "json"],
    cwd=root,
    capture_output=True,
    text=True,
)
if listing.returncode != 0:
    fatal(
        f"`{cargo} run -p vyre-bench -- list --format json` failed with status "
        f"{listing.returncode}. A registry that cannot be listed is unmeasured, "
        f"not covered.\n    {listing.stderr.strip()}"
    )
try:
    registry = json.loads(listing.stdout)
except json.JSONDecodeError as exc:
    fatal(f"vyre-bench registry is not JSON: {exc}")
if not isinstance(registry, list) or not registry:
    fatal("vyre-bench registry lists no cases; every coverage rule would pass vacuously")

registered = {case["id"] for case in registry if isinstance(case, dict) and "id" in case}
if not registered:
    fatal("vyre-bench registry entries carry no `id`; the scan would compare nothing")

tracked = run(["git", "ls-files", "-z"], cwd=root, capture_output=True, text=True)
if tracked.returncode != 0:
    fatal(f"git ls-files failed: {tracked.stderr.strip()}")

problems: list[str] = []

for dimension, case_id in sorted(REPRESENTATIVE.items()):
    if case_id not in registered:
        problems.append(
            f"{dimension}: representative vyre-bench case `{case_id}` is not registered\n"
            f"    Fix: restore the case, or point this dimension at the case that "
            f"replaced it. A dimension with no registered case is not measured."
        )

contract = root / CACHE_CONTRACT
if not contract.is_file():
    problems.append(
        f"compile_cache: executable cache contract `{CACHE_CONTRACT}` does not exist\n"
        f"    Fix: restore the contract, or promote compile_cache to a vyre-bench case."
    )
elif CACHE_TEST not in contract.read_text(encoding="utf-8"):
    problems.append(
        f"compile_cache: `{CACHE_CONTRACT}` no longer defines `{CACHE_TEST}`\n"
        f"    Fix: restore the test, or name the test that pins module-cache reuse now."
    )

references = 0
scanned = 0
for entry in tracked.stdout.split("\0"):
    if not entry or not entry.endswith(REFERENCE_SUFFIXES):
        continue
    path = root / entry
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        continue
    if "--case" not in text:
        continue
    scanned += 1
    for number, line in enumerate(text.splitlines(), 1):
        for case_id in CASE_FLAG.findall(line):
            references += 1
            if case_id not in registered:
                problems.append(
                    f"{entry}:{number}: names `--case {case_id}`, which the vyre-bench "
                    f"registry does not contain\n"
                    f"    Fix: use a registered case id. This invocation would fail "
                    f"wherever it runs, which for gpu-parity.yml and the release "
                    f"evidence manifests is far from PR CI."
                )

if problems:
    print(f"{NAME}: {len(problems)} coverage failure(s).", file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    sys.exit(1)

print(
    f"{NAME}: {len(REPRESENTATIVE) + 1} dimensions covered by the "
    f"{len(registered)}-case vyre-bench registry, and {references} `--case` "
    f"reference(s) across {scanned} tracked file(s) all resolve."
)
PY
