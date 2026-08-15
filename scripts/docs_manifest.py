#!/usr/bin/env python3
"""Generate navigation and enforce documentation authority from docs/DOCS.toml."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MANIFEST = DOCS / "DOCS.toml"
SUMMARY = DOCS / "SUMMARY.md"
INDEX = DOCS / "INDEX.md"
STATUSES = {"current", "generated", "superseded", "archived"}
AUDIENCES = {"user", "extension", "contributor", "release"}
GENERATIONS = {"manual", "generated"}
KINDS = {
    "architecture",
    "evidence",
    "governance",
    "guide",
    "history",
    "lifecycle",
    "optimization",
    "ownership",
    "reference",
    "release",
    "testing",
}
INTERNAL_MARKERS = (
    (re.compile(r"local://", re.IGNORECASE), "local planning URI"),
    (re.compile(r"\bBACKLOG\.md\b"), "execution backlog"),
    (re.compile(r"\b(?:subagent|agent swarm|worktree protocol)\b", re.IGNORECASE), "agent execution process"),
    (re.compile(r"\b(?:phase|slice|tranche)\s+[A-Z]?\d+\b", re.IGNORECASE), "internal phase identifier"),
)
GENERATED_PROVENANCE = {
    "CLI.md": ("CLI.toml", "../scripts/cli_docs.py"),
    "CRATE_GRAPH.md": ("CRATE_OWNERSHIP.toml", "../scripts/crate_ownership.py"),
    "INDEX.md": ("DOCS.toml", "../scripts/docs_manifest.py"),
    "OWNERSHIP.md": ("CRATE_OWNERSHIP.toml", "../scripts/crate_ownership.py"),
    "generated/OP_INVENTORY.md": ("generated/OP_SCHEMA.json", "../xtask-registry/src/docs/list_ops.rs"),
    "optimization/PASSES.md": (
        "../vyre-foundation/src/optimizer.rs",
        "../xtask-registry/src/docs/optimization_docs.rs",
    ),
}


class ManifestError(ValueError):
    """Documentation authority metadata is incomplete or inconsistent."""


def quoted(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def title_for(path: Path) -> str:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                return line[2:].strip()
    except (OSError, UnicodeError):
        pass
    return path.stem.replace("_", " ").replace("-", " ").title()


def generated_provenance(path: str) -> tuple[str, str] | None:
    if path.startswith("catalog/"):
        return "generated/OP_SCHEMA.json", "../xtask-registry/src/docs/catalog.rs"
    if path.startswith("release/"):
        return "../release/release-train.toml", "../scripts/release_docs.py"
    if path.startswith("testing/"):
        return "testing/TESTING.toml", "../scripts/testing_guides.py"
    return GENERATED_PROVENANCE.get(path)


def classify(path: str, status: str) -> tuple[str, str, str, str]:
    """Return audience, owner, kind, and reader-task section for a page."""
    lower = path.lower()
    if status in {"archived", "superseded"}:
        return "contributor", "historical", "history", "History"
    if path == "INDEX.md" or "DOCUMENTATION" in path:
        return "contributor", "docs-governance", "governance", "Documentation authority"
    if path in {"ARCHITECTURE.md", "CRATE_GRAPH.md", "OWNERSHIP.md"} or "OWNERSHIP" in path:
        return "extension", "architecture", "ownership", "Architecture and ownership"
    if path == "OPTIMIZATION_ARCHITECTURE.md":
        return "extension", "optimization", "optimization", "Optimization"
    if path == "optimization/PASSES.md":
        return "extension", "optimization", "reference", "Optimization"
    if lower.startswith("optimization/"):
        return "contributor", "optimization", "optimization", "Optimization"
    if lower.startswith("testing/") or "TEST" in path or "CONFORM" in path:
        return "contributor", "testing", "testing", "Testing and conformance"
    if lower.startswith("catalog/") or lower.startswith("generated/"):
        return "extension", "operation-registry", "reference", "API and operation reference"
    if any(token in path for token in ("RELEASE", "PERF", "BENCHMARK", "PUBLISH", "LAUNCH")) or lower.startswith("release/"):
        owner = "benchmark" if "PERF" in path or "BENCHMARK" in path else "release-tooling"
        kind = "evidence" if "EVIDENCE" in path or "BENCHMARK" in path else "release"
        return "release", owner, kind, "Performance and release"
    if any(token in path for token in ("RUNTIME", "PERSIST", "RECOVERY", "CACHE", "IO_", "SAFE")):
        return "extension", "runtime", "lifecycle", "Lifecycle and extension contracts"
    if "FRONTEND" in path or "GRAMMAR" in path or lower.startswith("frontend/"):
        return "extension", "frontend", "reference", "Lifecycle and extension contracts"
    if path in {"CLI.md", "COOKBOOK.md", "PORTABILITY.md", "TARGETS.md", "QUICKSTART.md"}:
        return "user", "public-facade", "guide", "User workflows"
    if path in {"API_STABILITY.md", "ERROR_GUIDE.md", "EXTENDING.md"}:
        return "extension", "public-facade", "reference", "Lifecycle and extension contracts"
    return "user", "public-facade", "guide", "User workflows"


def published_markdown() -> list[Path]:
    output = subprocess.run(
        ["git", "ls-files", "--cached", "--", "docs"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        ROOT / path
        for path in output.stdout.splitlines()
        if path.endswith(".md")
        and Path(path).name != "SUMMARY.md"
        and (ROOT / path).is_file()
    ]


def legacy_rows() -> dict[str, dict[str, Any]]:
    if not MANIFEST.exists():
        return {}
    data = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    return {
        str(page["path"]): page
        for page in data.get("page", [])
        if isinstance(page, dict) and "path" in page and "status" in page
    }


def bootstrap() -> None:
    old = legacy_rows()
    owner_rows = [
        ("architecture", "ARCHITECTURE.md"),
        ("benchmark", "optimization/BENCH_TARGETS.toml"),
        ("docs-governance", "DOCS.toml"),
        ("historical", "DOCS.toml"),
        ("operation-registry", "../vyre-foundation/src/operation.rs"),
        ("optimization", "optimization/OWNERSHIP.toml"),
        ("public-facade", "../vyre/src/lib.rs"),
        ("release-tooling", "../scripts/release_docs.py"),
        ("runtime", "../vyre-runtime/src/lib.rs"),
        ("testing", "testing/TESTING.toml"),
    ]
    lines = [
        "# Documentation lifecycle, audience, ownership, and generation authority.",
        "version = 2",
        "",
        "[book]",
        'title = "Vyre"',
        'description = "Whole-program compiler and authenticated artifact lifecycle"',
        "",
    ]
    for owner, authority in owner_rows:
        lines.extend(
            [
                "[[owner]]",
                f"id = {quoted(owner)}",
                f"authority = {quoted(authority)}",
                "",
            ]
        )
    for file in sorted(published_markdown()):
        path = file.relative_to(DOCS).as_posix()
        previous = old.get(path, {})
        status = str(previous.get("status", ""))
        if not status:
            if path == "INDEX.md" or generated_provenance(path):
                status = "generated"
            elif path.startswith(("archive/", "legacy/")):
                status = "archived"
            else:
                raise SystemExit(
                    f"unclassified documentation page {path}; assign a lifecycle before bootstrapping"
                )
        if path == "generated/README.md" and status == "generated":
            status = "current"
        audience, owner, kind, section = classify(path, status)
        provenance = generated_provenance(path) if status == "generated" else None
        authority, generator = provenance if provenance else ("self", "")
        lines.extend(
            [
                "[[page]]",
                f"path = {quoted(path)}",
                f"title = {quoted(title_for(file))}",
                f"status = {quoted(status)}",
                f"audience = {quoted(audience)}",
                f"owner = {quoted(owner)}",
                f"kind = {quoted(kind)}",
                f"section = {quoted(section)}",
                f"authority = {quoted(authority)}",
                f"generation = {quoted('generated' if provenance else 'manual')}",
                *([f"generator = {quoted(generator)}"] if generator else []),
                f"nav = {'true' if status in {'current', 'generated'} else 'false'}",
                "",
            ]
        )
    MANIFEST.write_text("\n".join(lines), encoding="utf-8")


def load_manifest() -> tuple[dict[str, str], list[dict[str, object]]]:
    data = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    if data.get("version") != 2:
        raise ManifestError("docs/DOCS.toml must use version = 2")
    owner_rows = data.get("owner")
    if not isinstance(owner_rows, list):
        raise ManifestError("docs/DOCS.toml must contain [[owner]] entries")
    owners: dict[str, str] = {}
    for index, row in enumerate(owner_rows):
        if not isinstance(row, dict):
            raise ManifestError(f"DOCS.toml owner row {index + 1} must be a table")
        owner = str(row.get("id", ""))
        authority = str(row.get("authority", ""))
        if not owner or not authority:
            raise ManifestError(f"DOCS.toml owner row {index + 1} needs id and authority")
        if owner in owners:
            raise ManifestError(f"duplicate documentation owner: {owner}")
        owners[owner] = authority
    pages = data.get("page")
    if not isinstance(pages, list):
        raise ManifestError("docs/DOCS.toml must contain [[page]] entries")
    return owners, pages


def resolve_metadata_path(value: str) -> Path:
    return (DOCS / value).resolve()


def validate(
    pages: list[dict[str, object]],
    actual: set[str] | None = None,
    owners: dict[str, str] | None = None,
) -> list[str]:
    failures: list[str] = []
    owners = owners or {}
    for owner, authority in owners.items():
        if not resolve_metadata_path(authority).exists():
            failures.append(f"documentation owner {owner}: authority does not exist: {authority}")
    paths = [str(page.get("path", "")) for page in pages]
    counts = Counter(paths)
    failures.extend(
        f"duplicate DOCS.toml page: {path}" for path, count in counts.items() if count > 1
    )
    if actual is None:
        actual = {path.relative_to(DOCS).as_posix() for path in published_markdown()}
    declared = set(paths)
    failures.extend(f"unclassified documentation page: {path}" for path in sorted(actual - declared))
    failures.extend(f"DOCS.toml names missing page: {path}" for path in sorted(declared - actual))

    for page in pages:
        path = str(page.get("path", ""))
        status = str(page.get("status", ""))
        audience = str(page.get("audience", ""))
        owner = str(page.get("owner", ""))
        kind = str(page.get("kind", ""))
        section = str(page.get("section", ""))
        authority = str(page.get("authority", ""))
        generation = str(page.get("generation", ""))
        generator = str(page.get("generator", ""))
        nav = page.get("nav")
        if status not in STATUSES:
            failures.append(f"{path}: invalid lifecycle {status!r}")
        if audience not in AUDIENCES:
            failures.append(f"{path}: invalid or missing audience {audience!r}")
        if owner not in owners:
            failures.append(f"{path}: unknown or deleted documentation owner {owner!r}")
        if kind not in KINDS:
            failures.append(f"{path}: invalid or missing document kind {kind!r}")
        if not section:
            failures.append(f"{path}: missing reader-task section")
        if generation not in GENERATIONS:
            failures.append(f"{path}: invalid or missing generation mode {generation!r}")
        if not authority:
            failures.append(f"{path}: missing authority source")
        elif authority != "self" and not resolve_metadata_path(authority).exists():
            failures.append(f"{path}: authority source does not exist: {authority}")
        if path.startswith(("archive/", "legacy/")) and status != "archived":
            failures.append(f"{path}: historical directories require archived lifecycle")
        if status in {"archived", "superseded"} and nav is not False:
            failures.append(f"{path}: inactive pages must set nav = false")
        if status in {"current", "generated"} and path.endswith(".md") and nav is not True:
            failures.append(f"{path}: active Markdown pages must set nav = true")
        if status == "generated" and generation != "generated":
            failures.append(f"{path}: generated lifecycle requires generated ownership")
        if status != "generated" and generation == "generated":
            failures.append(f"{path}: generated ownership requires generated lifecycle")
        if generation == "generated":
            if not generator:
                failures.append(f"{path}: generated page must name one generator")
            elif not resolve_metadata_path(generator).exists():
                failures.append(f"{path}: generator does not exist: {generator}")
            if authority == "self":
                failures.append(f"{path}: generated page cannot be its own authority")
        elif generator:
            failures.append(f"{path}: manual page cannot name a generator")
        if (
            status in {"current", "generated"}
            and audience in {"user", "extension"}
            and generation == "manual"
        ):
            document = DOCS / path
            try:
                content = document.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            for pattern, label in INTERNAL_MARKERS:
                if pattern.search(content):
                    failures.append(f"{path}: {audience} document leaks {label}")
    return failures


def workspace_facts() -> tuple[int, int]:
    output = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    packages = json.loads(output.stdout)["packages"]
    shipped = sum(
        1
        for package in packages
        for target in package["targets"]
        if set(target["kind"]) & {"lib", "rlib", "cdylib", "staticlib", "bin", "example"}
    )
    return len(packages), shipped


def render_summary(pages: list[dict[str, object]]) -> str:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for page in pages:
        if page.get("nav") is True and page.get("path") != "INDEX.md":
            groups[str(page["section"])].append(page)
    order = [
        "Documentation authority",
        "Architecture and ownership",
        "Lifecycle and extension contracts",
        "Optimization",
        "User workflows",
        "API and operation reference",
        "Testing and conformance",
        "Performance and release",
    ]
    lines = [
        "<!-- Generated from docs/DOCS.toml by scripts/docs_manifest.py. -->",
        "# Summary",
        "",
        "- [Documentation authority and lifecycle](INDEX.md)",
    ]
    for section in [*order, *sorted(set(groups) - set(order))]:
        if section not in groups:
            continue
        lines.extend(["", f"# {section}", ""])
        for page in sorted(groups[section], key=lambda item: (str(item["title"]), str(item["path"]))):
            lines.append(f"- [{page['title']}]({page['path']})")
    lines.append("")
    return "\n".join(lines)


def render_index(
    owners: dict[str, str], pages: list[dict[str, object]]
) -> str:
    package_count, target_count = workspace_facts()
    counts = Counter(str(page["status"]) for page in pages)
    lines = [
        "<!-- Generated from docs/DOCS.toml by scripts/docs_manifest.py. Do not edit. -->",
        "# Documentation Authority and Lifecycle",
        "",
        "Source: [`docs/DOCS.toml`](DOCS.toml).",
        "",
        "Each active page declares its audience, owner, authority source, kind, and",
        "generation mode. Generated pages also declare the generator. Superseded and",
        "archived pages remain lifecycle evidence and are excluded from navigation.",
        "",
        "## Documentation owners",
        "",
        "| Owner | Authority |",
        "| --- | --- |",
    ]
    for owner, authority in sorted(owners.items()):
        lines.append(f"| `{owner}` | [`{authority}`]({authority}) |")
    lines.extend(
        [
            "",
            "## Cargo-derived workspace facts",
            "",
            f"- Workspace packages: {package_count}.",
            f"- Shipped library, binary, and example targets: {target_count}.",
            "- Source: `cargo metadata --no-deps --format-version 1`.",
            "",
            "## Lifecycle counts",
            "",
            *[f"- {status}: {counts[status]}." for status in ("current", "generated", "superseded", "archived")],
            "",
            "## Pages",
            "",
            "| Status | Audience | Owner | Kind | Page | Authority | Generation |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for page in sorted(pages, key=lambda item: str(item["path"])):
        path = str(page["path"])
        authority = str(page["authority"])
        authority_text = "self" if authority == "self" else f"[{authority}]({authority})"
        generation = str(page["generation"])
        generator = str(page.get("generator", ""))
        generation_text = generation if not generator else f"{generation}: [{generator}]({generator})"
        lines.append(
            f"| `{page['status']}` | `{page['audience']}` | `{page['owner']}` | "
            f"`{page['kind']}` | `{path}` | {authority_text} | {generation_text} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_outputs(check: bool) -> int:
    owners, pages = load_manifest()
    failures = validate(pages, owners=owners)
    if failures:
        print("\n".join(f"docs manifest: {failure}" for failure in failures), file=sys.stderr)
        return 1
    outputs = {SUMMARY: render_summary(pages), INDEX: render_index(owners, pages)}
    drift = []
    for path, content in outputs.items():
        if check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                drift.append(path.relative_to(ROOT).as_posix())
        else:
            path.write_text(content, encoding="utf-8")
    if drift:
        print(
            "documentation output drift: " + ", ".join(drift) + ". Fix: run scripts/docs_manifest.py --write.",
            file=sys.stderr,
        )
        return 1
    return 0


def list_active() -> int:
    owners, pages = load_manifest()
    failures = validate(pages, owners=owners)
    if failures:
        print("\n".join(f"docs manifest: {failure}" for failure in failures), file=sys.stderr)
        return 1
    print("docs/SUMMARY.md")
    for page in sorted(pages, key=lambda item: str(item["path"])):
        if page.get("nav") is True:
            print(f"docs/{page['path']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bootstrap", action="store_true")
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--list-active", action="store_true")
    args = parser.parse_args()
    if args.bootstrap:
        bootstrap()
        return 0
    if args.list_active:
        return list_active()
    return write_outputs(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
