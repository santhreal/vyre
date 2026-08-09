#!/usr/bin/env python3
"""Generate mdBook navigation and lifecycle reports from docs/DOCS.toml."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MANIFEST = DOCS / "DOCS.toml"
SUMMARY = DOCS / "SUMMARY.md"
INDEX = DOCS / "INDEX.md"
STATUSES = {"current", "generated", "superseded", "archived"}
GENERATED_SOURCES = {
    "CLI.md": "CLI.toml",
    "CRATE_GRAPH.md": "CRATE_OWNERSHIP.toml",
    "OWNERSHIP.md": "CRATE_OWNERSHIP.toml",
    "RELEASE_CHECKLIST.md": "../scripts/release_docs.py",
    "generated/OP_INVENTORY.md": "generated/OP_SCHEMA.json",
    "generated/README.md": "generated/OP_SCHEMA.json",
    "optimization/XTASK_COMMAND_MATRIX.md": "../xtask/src/command_matrix.rs",
}


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


def generated_source(path: str) -> str:
    if path.startswith("catalog/"):
        return "generated/OP_SCHEMA.json"
    if path.startswith("testing/"):
        return "../scripts/testing_guides.py"
    return GENERATED_SOURCES.get(path, "")


def section_for(path: str, status: str) -> str:
    if path == "INDEX.md":
        return "Authority"
    if path == "ARCHITECTURE.md" or "OWNERSHIP" in path or path == "CRATE_GRAPH.md":
        return "Architecture and ownership"
    prefix = path.split("/", 1)[0] if "/" in path else ""
    if prefix == "optimization":
        return "Optimization"
    if prefix == "testing":
        return "Testing"
    if prefix == "catalog" or prefix == "generated":
        return "Generated reference"
    if prefix:
        return prefix.replace("_", " ").replace("-", " ").title()
    if status == "generated":
        return "Generated reference"
    return "Guides and reference"


def class_for(path: str, status: str) -> str:
    if status in {"archived", "superseded"}:
        return "history"
    if status == "generated":
        return "projection"
    if path == "ARCHITECTURE.md":
        return "architecture"
    if "OWNERSHIP" in path or path == "CRATE_GRAPH.md":
        return "ownership"
    if path.startswith("optimization/"):
        return "optimization"
    if "RELEASE" in path or path.startswith("release/"):
        return "release"
    if path == "DOCUMENTATION_GOVERNANCE.md":
        return "governance"
    return "guide"


def legacy_rows() -> dict[str, str]:
    rows: dict[str, str] = {}
    if MANIFEST.exists():
        data = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
        return {
            str(page["path"]): str(page["status"])
            for page in data.get("page", [])
            if isinstance(page, dict) and "path" in page and "status" in page
        }
    if not INDEX.exists():
        return rows
    pattern = re.compile(
        r"\| `(?P<status>current|generated|superseded|archived)` "
        r"\| [^|]*\| [^|]*\| \[docs/(?P<path>[^\]]+)\]\([^)]*\) \|"
    )
    for line in INDEX.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            rows[match.group("path")] = match.group("status")
    return rows


def bootstrap() -> None:
    old = legacy_rows()
    pages: list[dict[str, object]] = []
    for file in sorted(DOCS.rglob("*.md")):
        path = file.relative_to(DOCS).as_posix()
        if path in {"README.md", "SUMMARY.md"}:
            continue
        if path == "INDEX.md":
            status = "generated"
        elif path in old:
            status = old[path]
        elif path.startswith("testing/"):
            status = "generated"
        elif path.startswith(("archive/", "legacy/")):
            status = "archived"
        else:
            raise SystemExit(
                f"unclassified documentation page {path}; assign a lifecycle before bootstrapping"
            )
        source = generated_source(path) if status == "generated" else path
        if status == "generated" and path == "INDEX.md":
            source = "DOCS.toml"
        if status == "generated" and not source:
            raise SystemExit(f"generated page {path} has no known source")
        pages.append(
            {
                "path": path,
                "title": title_for(file),
                "status": status,
                "class": class_for(path, status),
                "section": section_for(path, status),
                "source": source,
                "nav": status in {"current", "generated"} and path.endswith(".md"),
            }
        )

    lines = [
        "# Documentation lifecycle authority. Generated pages name their source;",
        "# current pages are navigable; superseded and archived pages are excluded.",
        "version = 1",
        "",
        "[book]",
        'title = "Vyre"',
        'description = "Whole-program compiler and authenticated GPU artifact lifecycle"',
        "",
    ]
    for page in pages:
        lines.extend(
            [
                "[[page]]",
                f"path = {quoted(str(page['path']))}",
                f"title = {quoted(str(page['title']))}",
                f"status = {quoted(str(page['status']))}",
                f"class = {quoted(str(page['class']))}",
                f"section = {quoted(str(page['section']))}",
                f"source = {quoted(str(page['source']))}",
                f"nav = {'true' if page['nav'] else 'false'}",
                "",
            ]
        )
    MANIFEST.write_text("\n".join(lines), encoding="utf-8")


def load_pages() -> tuple[dict[str, object], list[dict[str, object]]]:
    data = tomllib.loads(MANIFEST.read_text(encoding="utf-8"))
    if data.get("version") != 1:
        raise ValueError("docs/DOCS.toml must use version = 1")
    pages = data.get("page")
    if not isinstance(pages, list):
        raise ValueError("docs/DOCS.toml must contain [[page]] entries")
    return data, pages


def validate(pages: list[dict[str, object]]) -> list[str]:
    failures: list[str] = []
    paths = [str(page.get("path", "")) for page in pages]
    counts = Counter(paths)
    failures.extend(f"duplicate DOCS.toml page: {path}" for path, count in counts.items() if count > 1)
    actual = {
        path.relative_to(DOCS).as_posix()
        for path in DOCS.rglob("*.md")
        if path.name != "SUMMARY.md"
    }
    declared = set(paths)
    failures.extend(f"unclassified documentation page: {path}" for path in sorted(actual - declared))
    failures.extend(f"DOCS.toml names missing page: {path}" for path in sorted(declared - actual))
    for page in pages:
        path = str(page.get("path", ""))
        status = str(page.get("status", ""))
        nav = page.get("nav")
        source = str(page.get("source", ""))
        if status not in STATUSES:
            failures.append(f"{path}: invalid lifecycle {status!r}")
        if status in {"archived", "superseded"} and nav is not False:
            failures.append(f"{path}: inactive pages must set nav = false")
        if status in {"current", "generated"} and path.endswith(".md") and nav is not True:
            failures.append(f"{path}: active Markdown pages must set nav = true")
        if status == "generated":
            if not source:
                failures.append(f"{path}: generated page must name one source")
            elif not (DOCS / source).resolve().exists():
                failures.append(f"{path}: generated source does not exist: {source}")
    return failures


def workspace_facts() -> tuple[int, int]:
    output = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(output.stdout)
    packages = metadata["packages"]
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
    lines = [
        "<!-- Generated from docs/DOCS.toml by scripts/docs_manifest.py. -->",
        "# Summary",
        "",
        "- [Documentation authority and lifecycle](INDEX.md)",
    ]
    for section in sorted(groups):
        lines.extend(["", f"# {section}", ""])
        for page in sorted(groups[section], key=lambda item: (str(item["title"]), str(item["path"]))):
            lines.append(f"- [{page['title']}]({page['path']})")
    lines.append("")
    return "\n".join(lines)


def render_index(pages: list[dict[str, object]]) -> str:
    package_count, target_count = workspace_facts()
    counts = Counter(str(page["status"]) for page in pages)
    lines = [
        "<!-- Generated from docs/DOCS.toml by scripts/docs_manifest.py. Do not edit. -->",
        "# Documentation Authority and Lifecycle",
        "",
        "Source: [`docs/DOCS.toml`](DOCS.toml).",
        "",
        "## Authority",
        "",
        "- Architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md).",
        "- Crate boundaries: [`CRATE_OWNERSHIP.toml`](CRATE_OWNERSHIP.toml).",
        "- Optimization control: [`optimization/README.md`](optimization/README.md).",
        "- Documentation lifecycle: [`DOCS.toml`](DOCS.toml).",
        "- Navigation: [`SUMMARY.md`](SUMMARY.md), generated from `DOCS.toml`.",
        "",
        "`current` pages are normative. `generated` pages are projections of the named source.",
        "`superseded` and `archived` pages are historical and do not appear in navigation.",
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
        "| Status | Class | Page | Source |",
        "| --- | --- | --- | --- |",
    ]
    for page in sorted(pages, key=lambda item: str(item["path"])):
        path = str(page["path"])
        source = str(page["source"])
        source_link = f"[{source}]({source})" if source else ""
        lines.append(
            f"| `{page['status']}` | `{page['class']}` | `{path}` | {source_link} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_outputs(check: bool) -> int:
    _, pages = load_pages()
    failures = validate(pages)
    if failures:
        print("\n".join(f"docs manifest: {failure}" for failure in failures), file=sys.stderr)
        return 1
    outputs = {SUMMARY: render_summary(pages), INDEX: render_index(pages)}
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


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bootstrap", action="store_true")
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.bootstrap:
        bootstrap()
        return 0
    return write_outputs(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
