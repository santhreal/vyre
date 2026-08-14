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

python3 - "$ROOT" <<'PY'
import sys
from pathlib import Path, PurePosixPath
from subprocess import run

GATE = "check-every-source-file-parses"
BATCH = 200
EDITION = "2021"
root = Path(sys.argv[1]).resolve()


def tracked(*args: str) -> list[str]:
    done = run(["git", "ls-files", "-z", "--", *args], cwd=root, capture_output=True, text=True)
    if done.returncode != 0:
        sys.exit(f"{GATE}: git ls-files failed: {done.stderr.strip()}")
    return [entry for entry in done.stdout.split("\0") if entry]


def rustfmt(paths: list[PurePosixPath]) -> list[str]:
    """Return every `error...` line rustfmt reports for these files."""
    done = run(
        ["rustfmt", "--edition", EDITION, "--check", "--color", "never", *[str(p) for p in paths]],
        cwd=root,
        capture_output=True,
        text=True,
    )
    lines = (done.stdout + done.stderr).splitlines()
    errors = [line for line in lines if line.startswith("error")]
    if done.returncode not in (0, 1) and not errors:
        sys.exit(
            f"{GATE}: rustfmt exited {done.returncode} without reporting an error line.\n"
            f"    Fix: run it by hand on {[str(p) for p in paths[:3]]} and read the output. A "
            f"parser that fails for an unknown reason must not read as a clean tree."
        )
    return errors


sources = [PurePosixPath(entry) for entry in tracked("*.rs")]
if not sources:
    sys.exit(f"{GATE}: no tracked .rs file found; the scan would pass vacuously")

# The exemption is derived from the tree: a directory holding a tracked template
# manifest (`Cargo.toml.liquid`) is scaffolding, so its sources are rendered
# before they are Rust. Nothing is matched on file contents, so a template that
# happens to be free of placeholders is still exempt and still checked below.
template_roots = [PurePosixPath(entry).parent for entry in tracked("*Cargo.toml.*")]
problems: list[str] = []
exempt: dict[PurePosixPath, PurePosixPath] = {}
for template_root in template_roots:
    owned = [path for path in sources if str(path).startswith(f"{template_root}/")]
    if not owned:
        problems.append(
            f"template manifest in `{template_root}` covers no tracked .rs file\n"
            f"    Fix: delete the template, or delete this exemption. An exemption that "
            f"matches nothing reserves an allowance nothing uses."
        )
    for path in owned:
        exempt[path] = template_root

scanned = [path for path in sources if path not in exempt]
if not scanned:
    sys.exit(f"{GATE}: every tracked .rs file is exempt; the scan would pass vacuously")

# The exemption has to bite. A template whose sources all parse needs no
# exemption, and an exemption nothing uses is how a rule stops covering the
# thing it names.
unused: dict[PurePosixPath, list[PurePosixPath]] = {}
for path, template_root in exempt.items():
    if not rustfmt([path]):
        unused.setdefault(template_root, []).append(path)
for template_root, paths in unused.items():
    if len(paths) == len(
        [path for path, owner in exempt.items() if owner == template_root]
    ):
        problems.append(
            f"every tracked .rs file under template root `{template_root}` parses as Rust: "
            f"{[str(path) for path in sorted(paths)]}\n"
            f"    Fix: delete the template exemption in this gate and scan those files with "
            f"the rest. An exemption whose files no longer need it is slack a real parse "
            f"failure hides in."
        )

failed: list[str] = []
for start in range(0, len(scanned), BATCH):
    batch = scanned[start : start + BATCH]
    if not rustfmt(batch):
        continue
    # A batch reports the offender's path in a `-->` line, but per-file runs
    # name it without parsing rustfmt's diagnostic layout.
    for path in batch:
        for error in rustfmt([path]):
            failed.append(f"`{path}` is not valid Rust: {error}")
            break

for entry in failed:
    problems.append(
        f"{entry}\n"
        f"    Fix: make the file parse, or move it under a scaffolding template root if it "
        f"is not Rust. A tracked .rs file nothing can parse is not code."
    )

if problems:
    print(f"{GATE}: {len(problems)} problem(s)")
    for problem in problems:
        print(f"  - {problem}")
    sys.exit(1)

print(
    f"{GATE}: {len(scanned)} tracked .rs file(s) parse under edition {EDITION}; "
    f"{len(exempt)} exempt as scaffolding templates across "
    f"{len(set(exempt.values()))} template root(s)."
)
PY
