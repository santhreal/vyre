#!/usr/bin/env bash
# Every tracked .rs file must be compiled by some declared cargo target.
#
# The class this closes: a source file nothing declares is not code, it is a
# claim. `vyre-libs/src/visual/glyph_grid/mod.rs` shipped an op registration and
# eight contracts while `docs/catalog/visual.md`, `docs/generated/OP_SCHEMA.json`
# and `docs/optimization/OP_MATRIX.toml` all listed the op as supported; no `mod`
# declaration ever named it, so none of it compiled and none of it ran. The four
# `vyre-driver-cuda/tests/resident_dispatch_contracts/*_contracts.rs` chunks lost
# their parent test file to a deletion and took 15 contracts with them, while
# OP_MATRIX.toml still cited the file as the proving test for elementwise_add.
# Both read as coverage from any distance.
#
# The target set is derived from the tracked manifests at run time, not listed
# here, so a new crate, bin, test, bench or example is picked up by adding it and
# a new orphan cannot be added without turning this gate red.
#
# It reports four failures:
#   1. a tracked .rs file no target reaches,
#   2. a declared target `path` that names no tracked file,
#   3. a `mod` declaration in a reached file that resolves to no tracked file,
#   4. a stale exemption: a template root with no tracked .rs, or a trybuild
#      fixture path matching nothing.
#
# This gate deliberately does NOT invoke cargo. `cargo build` cannot see this
# defect at all: an undeclared file is not part of any target, so a green build
# is exactly what an orphan produces. It reads tracked files with git, tomllib
# and a module-graph walk.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 - "$ROOT" <<'PY'
import posixpath
import re
import sys
import tomllib
from pathlib import Path, PurePosixPath
from subprocess import run

GATE = "check-every-source-file-is-reachable"
root = Path(sys.argv[1]).resolve()


def tracked(*args: str) -> list[str]:
    done = run(["git", "ls-files", "-z", "--", *args], cwd=root, capture_output=True, text=True)
    if done.returncode != 0:
        sys.exit(f"{GATE}: git ls-files failed: {done.stderr.strip()}")
    return [entry for entry in done.stdout.split("\0") if entry]


sources = {PurePosixPath(entry) for entry in tracked("*.rs")}
# A `#[path]` may name Rust code that is not called `.rs`: vyre-aot compiles
# `templates/artifact.rs.tmpl` directly so the shipped template is the tested
# artifact. Module resolution therefore runs against every tracked file.
everything = {PurePosixPath(entry) for entry in tracked()}
manifests = tracked("*Cargo.toml")
templates = tracked("*Cargo.toml.*")
if not sources:
    sys.exit(f"{GATE}: no tracked .rs file found; the scan would pass vacuously")
if not manifests:
    sys.exit(f"{GATE}: no tracked Cargo.toml found; the scan would pass vacuously")

problems: list[str] = []


def norm(path: PurePosixPath) -> PurePosixPath:
    """Collapse `.` and `..` so a #[path] that climbs compares equal to a tracked entry."""
    return PurePosixPath(posixpath.normpath(str(path)))


# ── Targets: every root file cargo compiles, derived from the manifests ───────

RAW_STRING = re.compile(r"b?r(#*)\"")
STRING = re.compile(r"b?\"")
CHAR = re.compile(r"b?'(\\.|[^\\'])'")

TOKEN = re.compile(
    r"#\s*\[\s*path\s*=\s*\"(?P<lit>\d+)\"\s*\]"
    r"|\bmod\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<term>[;{])"
    r"|\binclude!\s*\(\s*\"(?P<inc>\d+)\""
    r"|(?P<open>\{)|(?P<close>\})"
)
TRYBUILD = re.compile(r"\.(?:compile_fail|pass)\s*\(\s*\"(?P<lit>\d+)\"")


def strip_code(text: str) -> tuple[str, list[str]]:
    """Drop comments and replace every literal by its index, so braces and
    #[path] arguments can be read without a full Rust parser."""
    out: list[str] = []
    lits: list[str] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "/" and text.startswith("//", i):
            end = text.find("\n", i)
            i = n if end < 0 else end
            continue
        if ch == "/" and text.startswith("/*", i):
            depth, i = 1, i + 2
            while i < n and depth:
                if text.startswith("/*", i):
                    depth, i = depth + 1, i + 2
                elif text.startswith("*/", i):
                    depth, i = depth - 1, i + 2
                else:
                    i += 1
            out.append(" ")
            continue
        raw = RAW_STRING.match(text, i)
        if raw:
            fence = '"' + raw.group(1)
            end = text.find(fence, raw.end())
            body = text[raw.end() : end] if end >= 0 else text[raw.end() :]
            out.append(f'"{len(lits)}"')
            lits.append(body)
            i = n if end < 0 else end + len(fence)
            continue
        plain = STRING.match(text, i)
        if plain:
            j = plain.end()
            body: list[str] = []
            while j < n and text[j] != '"':
                if text[j] == "\\" and j + 1 < n:
                    body.append(text[j + 1])
                    j += 2
                    continue
                body.append(text[j])
                j += 1
            out.append(f'"{len(lits)}"')
            lits.append("".join(body))
            i = j + 1
            continue
        literal_char = CHAR.match(text, i)
        if literal_char:
            out.append("'c'")
            i = literal_char.end()
            continue
        out.append(ch)
        i += 1
    return "".join(out), lits


def read(rel: PurePosixPath) -> tuple[str, list[str]]:
    return strip_code((root / str(rel)).read_text(encoding="utf-8", errors="replace"))


def declarations(code: str, lits: list[str]) -> tuple[list[tuple[str, str | None, tuple]], list[str]]:
    """Module declarations as (name, #[path] value or None, inline module stack)
    plus every include! argument, in source order."""
    mods: list[tuple[str, str | None, tuple]] = []
    includes: list[str] = []
    stack: list[str] = []
    opened: list[int] = []
    depth = 0
    pending: str | None = None
    for match in TOKEN.finditer(code):
        if match.group("lit") is not None:
            pending = lits[int(match.group("lit"))]
            continue
        if match.group("inc") is not None:
            includes.append(lits[int(match.group("inc"))])
            continue
        if match.group("open"):
            depth += 1
            continue
        if match.group("close"):
            depth -= 1
            if opened and opened[-1] == depth:
                opened.pop()
                stack.pop()
            continue
        name, term = match.group("name"), match.group("term")
        if term == "{":
            stack.append(name)
            opened.append(depth)
            depth += 1
        else:
            mods.append((name, pending, tuple(stack)))
        pending = None
    return mods, includes


def crate_dir(rel: str) -> PurePosixPath:
    return PurePosixPath(rel).parent


roots: list[tuple[PurePosixPath, str]] = []


def declare(label: str, path: PurePosixPath) -> None:
    """Record a declared target root, or a problem when it names no tracked file."""
    path = norm(path)
    if path in sources:
        roots.append((path, label))
        return
    state = "exists but is untracked" if (root / str(path)).is_file() else "does not exist"
    problems.append(
        f"declared target {label} names `{path}`, which {state}\n"
        f"    Fix: restore the file or delete the target entry. A [[test]] or [[bin]] "
        f"path naming nothing is a target that never runs and never says so."
    )


def under(directory: PurePosixPath) -> list[PurePosixPath]:
    return [path for path in sources if path.parent == directory]


def auto_roots(pkgdir: PurePosixPath, kind: str) -> list[PurePosixPath]:
    """Cargo's own autodiscovery: `<kind>/*.rs` plus `<kind>/*/main.rs`."""
    base = pkgdir / kind
    found = list(under(base))
    found += [
        path
        for path in sources
        if path.name == "main.rs" and path.parent.parent == base and path.parent != base
    ]
    return found


for entry in manifests:
    manifest = root / entry
    try:
        data = tomllib.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        problems.append(f"`{entry}` is not readable as TOML: {exc}")
        continue
    package = data.get("package")
    if not package:
        continue
    pkgdir = crate_dir(entry)
    name = package.get("name", pkgdir.name)

    lib = data.get("lib") or {}
    if "path" in lib:
        declare(f"`{entry}` [lib]", pkgdir / lib["path"])
    elif norm(pkgdir / "src/lib.rs") in sources:
        roots.append((norm(pkgdir / "src/lib.rs"), f"`{entry}` src/lib.rs"))

    build = package.get("build")
    if isinstance(build, str):
        declare(f"`{entry}` package.build", pkgdir / build)
    elif build is not False and norm(pkgdir / "build.rs") in sources:
        roots.append((norm(pkgdir / "build.rs"), f"`{entry}` build.rs"))

    for bin_spec in data.get("bin") or []:
        bin_name = bin_spec.get("name", name)
        if "path" in bin_spec:
            declare(f"`{entry}` [[bin]] {bin_name}", pkgdir / bin_spec["path"])
            continue
        guesses = [
            pkgdir / "src/bin" / f"{bin_name}.rs",
            pkgdir / "src/bin" / bin_name / "main.rs",
        ]
        if bin_name == name:
            guesses.insert(0, pkgdir / "src/main.rs")
        found = [norm(guess) for guess in guesses if norm(guess) in sources]
        if found:
            roots.append((found[0], f"`{entry}` [[bin]] {bin_name}"))
        else:
            problems.append(
                f"declared target `{entry}` [[bin]] {bin_name} has no `path` and none of "
                f"{[str(norm(guess)) for guess in guesses]} is tracked\n"
                f"    Fix: add the source file or delete the [[bin]] entry."
            )

    for kind, folder, auto in (
        ("test", "tests", "autotests"),
        ("bench", "benches", "autobenches"),
        ("example", "examples", "autoexamples"),
    ):
        for spec in data.get(kind) or []:
            spec_name = spec.get("name", "<unnamed>")
            if "path" in spec:
                declare(f"`{entry}` [[{kind}]] {spec_name}", pkgdir / spec["path"])
            else:
                guesses = [
                    pkgdir / folder / f"{spec_name}.rs",
                    pkgdir / folder / spec_name / "main.rs",
                ]
                found = [norm(guess) for guess in guesses if norm(guess) in sources]
                if found:
                    roots.append((found[0], f"`{entry}` [[{kind}]] {spec_name}"))
                else:
                    problems.append(
                        f"declared target `{entry}` [[{kind}]] {spec_name} has no `path` and "
                        f"none of {[str(norm(guess)) for guess in guesses]} is tracked\n"
                        f"    Fix: add the source file or delete the [[{kind}]] entry."
                    )
        if package.get(auto) is not False:
            for path in auto_roots(pkgdir, folder):
                roots.append((path, f"`{entry}` autodiscovered {kind}"))

    if package.get("autobins") is not False:
        if norm(pkgdir / "src/main.rs") in sources:
            roots.append((norm(pkgdir / "src/main.rs"), f"`{entry}` src/main.rs"))
        for path in auto_roots(pkgdir, "src/bin"):
            roots.append((path, f"`{entry}` autodiscovered bin"))

if not roots:
    sys.exit(f"{GATE}: no cargo target root resolved; the scan would pass vacuously")

# ── Reachability: walk the module graph out from every target root ────────────

reached: dict[PurePosixPath, str] = {}
missing_mods: list[str] = []
queue: list[tuple[PurePosixPath, PurePosixPath, str]] = []
for path, label in roots:
    if path not in reached:
        reached[path] = label
        queue.append((path, path.parent, label))

while queue:
    rel, mod_base, label = queue.pop()
    try:
        code, lits = read(rel)
    except OSError as exc:
        problems.append(f"`{rel}` is reached from {label} but unreadable: {exc}")
        continue
    mods, includes = declarations(code, lits)

    for name, path_attr, stack in mods:
        if path_attr is not None:
            base = mod_base
            for part in stack:
                base = base / part
            # A #[path] on a module declared at file scope resolves against the
            # file's own directory; inside an inline `mod` block it resolves
            # against the enclosing module directory.
            anchor = base if stack else rel.parent
            candidates = [norm(anchor / path_attr)]
        else:
            base = mod_base
            for part in stack:
                base = base / part
            candidates = [norm(base / f"{name}.rs"), norm(base / name / "mod.rs")]
        hit = next((candidate for candidate in candidates if candidate in everything), None)
        if hit is None:
            missing_mods.append(
                f"`{rel}` declares `mod {name};` but none of "
                f"{[str(candidate) for candidate in candidates]} is tracked\n"
                f"    Fix: add the module file, or delete the declaration."
            )
            continue
        if hit not in reached:
            reached[hit] = f"{label} -> {rel}"
            child_base = hit.parent if hit.name == "mod.rs" else hit.parent / hit.stem
            queue.append((hit, child_base, reached[hit]))

    for raw in includes:
        target = norm(rel.parent / raw)
        if target not in everything:
            continue
        if target not in reached:
            reached[target] = f"{label} -> {rel} include!"
            queue.append((target, mod_base, reached[target]))

# ── Exemptions, each derived from the tree and each required to bite ──────────

exempt: dict[PurePosixPath, str] = {}

for entry in templates:
    template_root = crate_dir(entry)
    owned = [path for path in sources if str(path).startswith(f"{template_root}/")]
    if not owned:
        problems.append(
            f"template manifest `{entry}` covers no tracked .rs file\n"
            f"    Fix: delete the template, or delete this exemption. An exemption that "
            f"matches nothing reserves an allowance nothing uses."
        )
        continue
    for path in owned:
        exempt[path] = f"scaffolding template `{entry}` (no compiling package)"

for path in sorted(reached):
    try:
        code, lits = read(path)
    except OSError:
        continue
    for match in TRYBUILD.finditer(code):
        pattern = lits[int(match.group("lit"))]
        owner = next(
            (crate_dir(entry) for entry in manifests if str(path).startswith(f"{crate_dir(entry)}/")),
            path.parent,
        )
        prefix = norm(owner / pattern)
        named = (
            [candidate for candidate in sources if candidate == prefix]
            if not any(ch in pattern for ch in "*?[")
            else [
                candidate
                for candidate in sources
                if PurePosixPath(candidate).match(str(prefix))
            ]
        )
        if not named:
            problems.append(
                f"`{path}` runs trybuild over `{pattern}`, which matches no tracked file\n"
                f"    Fix: restore the fixture or delete the case. A trybuild path naming "
                f"nothing is a compile-fail test that asserts nothing."
            )
            continue
        for candidate in named:
            exempt[candidate] = f"trybuild fixture compiled at run time by `{path}`"

unreachable = sorted(path for path in sources if path not in reached and path not in exempt)
for path in unreachable:
    problems.append(
        f"`{path}` is compiled by no cargo target\n"
        f"    Fix: declare it (`mod`, `#[path]`, `include!`, or a target entry in the "
        f"owning Cargo.toml), or delete it. A file nothing compiles reads as coverage "
        f"and provides none."
    )

problems.extend(missing_mods)

if problems:
    print(f"{GATE}: {len(problems)} unreachable or unresolvable source file(s)")
    for problem in problems:
        print(f"  - {problem}")
    sys.exit(1)

print(
    f"{GATE}: {len([path for path in reached if path in sources])} tracked .rs file(s) "
    f"reachable from {len(roots)} declared cargo target(s) across "
    f"{len(manifests)} manifest(s); {len(exempt)} exempt (trybuild fixtures and "
    f"scaffolding templates), 0 orphaned."
)
PY
