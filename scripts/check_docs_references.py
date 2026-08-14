#!/usr/bin/env python3
"""Reject unpublished local paths hidden in code spans and command examples."""

from __future__ import annotations

import argparse
import glob
import re
import shlex
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

INLINE_CODE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
LINE_SELECTOR = re.compile(r":\d+(?:-\d+)?$")
COMMAND_FENCE = re.compile(
    r"```(?:console|bash|sh|shell)\s*\n(.*?)```", re.IGNORECASE | re.DOTALL
)
PATH_SUFFIXES = {
    ".c",
    ".h",
    ".json",
    ".md",
    ".py",
    ".rs",
    ".sh",
    ".toml",
    ".txt",
    ".vir",
    ".wgsl",
}
ROOT_PREFIXES = (
    ".github/",
    "Cargo.toml",
    "CHANGELOG.md",
    "README.md",
    "consumer/",
    "docs/",
    "libs/",
    "release/",
    "scripts/",
    "tools/",
)
CRATE_RELATIVE_PREFIXES = (
    "api/",
    "benches/",
    "examples/",
    "hardware/",
    "pipeline/",
    "rules/",
    "src/",
    "tests/",
)
OUTPUT_FLAGS = {
    "--emit",
    "--out",
    "--out-dir",
    "--output",
    "--output-dir",
    "--write",
    "-o",
}


@dataclass(frozen=True, order=True)
class Reference:
    document: str
    raw: str
    resolved: str
    source: str


def workspace_readmes(root: Path) -> Iterable[Path]:
    manifest = root / "Cargo.toml"
    if not manifest.is_file():
        return []
    try:
        value = tomllib.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        return []
    members = value.get("workspace", {}).get("members", [])
    if not isinstance(members, list):
        return []
    return [
        root / member / "README.md"
        for member in members
        if isinstance(member, str) and (root / member / "README.md").is_file()
    ]


def inactive_manifest_documents(root: Path) -> set[Path]:
    manifest = root / "docs/DOCS.toml"
    try:
        value = tomllib.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError):
        return set()
    inactive: set[Path] = set()
    for page in value.get("page", []):
        if not isinstance(page, dict) or page.get("status") not in {"archived", "superseded"}:
            continue
        path = page.get("path")
        if isinstance(path, str):
            inactive.add((root / "docs" / path).resolve(strict=False))
    return inactive


def public_documents(root: Path) -> list[Path]:
    inactive = inactive_manifest_documents(root)
    candidates = [root / "README.md", *root.glob("docs/**/*.md"), *workspace_readmes(root)]
    documents: list[Path] = []
    for path in sorted(set(candidates)):
        if not path.is_file() or path.resolve() in inactive:
            continue
        relative = path.relative_to(root)
        if relative.parts[:2] in {("docs", "archive"), ("docs", "legacy")} and path.name != "README.md":
            continue
        if gitignored(root, relative.as_posix()):
            continue
        documents.append(path)
    return documents


def gitignored(root: Path, relative: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--", relative],
        cwd=root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def clean_token(raw: str) -> str:
    return raw.strip().strip("'\"").rstrip(".,;:")


def path_token(raw: str) -> str:
    return LINE_SELECTOR.sub("", clean_token(raw).split("#", 1)[0])


def has_existing_root_prefix(root: Path, token: str) -> bool:
    if "/" not in token:
        return False
    first = token.split("/", 1)[0]
    return first not in {".", ".."} and (root / first).exists()


def is_path_candidate(root: Path, raw: str) -> bool:
    token = path_token(raw)
    if not token or any(character.isspace() for character in token):
        return False
    if token.startswith(("http://", "https://", "mailto:")):
        return False
    if (
        "::" in token
        or token.startswith("$")
        or any(character in token for character in "<>{}@")
        or token.startswith("///")
    ):
        return False
    if token in {".", "..", "./", "../"}:
        return False
    if token.startswith(("./", "../", "/")):
        return True
    if token.startswith(ROOT_PREFIXES):
        return True
    if "/" in token and Path(token).suffix in PATH_SUFFIXES:
        return True
    if has_existing_root_prefix(root, token):
        return True
    return False


def resolve(
    root: Path, document: Path, raw: str, *, from_command: bool = False
) -> tuple[str, Path] | None:
    token = path_token(raw)
    if not is_path_candidate(root, token):
        return None
    absolute = token.startswith("/")
    if absolute:
        # An absolute token is not a workspace-relative claim. Stripping the
        # leading slash reported `/dev/null` as a missing `dev/null` in this
        # repository, which names a defect the document does not have. An
        # absolute path into this checkout is still a claim and is checked.
        candidate = Path(token)
    elif (
        document.name == "README.md"
        and document.parent != root
        and token.startswith(CRATE_RELATIVE_PREFIXES)
    ):
        candidate = document.parent / token
    elif token.startswith(ROOT_PREFIXES) or has_existing_root_prefix(root, token):
        candidate = root / token
    elif from_command and token.startswith("./"):
        candidate = root / token[2:]
    else:
        candidate = document.parent / token
    try:
        relative = candidate.resolve(strict=False).relative_to(root.resolve())
    except ValueError:
        if absolute:
            return None
        return ("outside", candidate)
    return (relative.as_posix(), candidate)


def command_lines(text: str) -> Iterable[str]:
    for block in COMMAND_FENCE.findall(text):
        logical = ""
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if line.startswith(("$ ", "> ")):
                line = line[2:].lstrip()
            if not line or line.startswith("#"):
                continue
            logical = f"{logical} {line[:-1]}".strip() if line.endswith("\\") else f"{logical} {line}".strip()
            if raw_line.rstrip().endswith("\\"):
                continue
            yield logical
            logical = ""
        if logical:
            yield logical


def command_path_tokens(root: Path, line: str) -> Iterable[str]:
    try:
        tokens = shlex.split(line, comments=True, posix=True)
    except ValueError:
        return []
    paths: list[str] = []
    skip_next = False
    for index, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        if token in OUTPUT_FLAGS:
            skip_next = True
            continue
        if any(token.startswith(f"{flag}=") for flag in OUTPUT_FLAGS):
            continue
        if token.startswith("-") or "=" in token and not token.startswith(("./", "../")):
            continue
        if index == 0 or is_path_candidate(root, token):
            if is_path_candidate(root, token):
                paths.append(token)
    return paths


def collect_references(root: Path) -> list[Reference]:
    references: set[Reference] = set()
    for document in public_documents(root):
        relative_document = document.relative_to(root).as_posix()
        text = document.read_text(encoding="utf-8")
        if relative_document != "docs/INDEX.md":
            for raw in INLINE_CODE.findall(text):
                resolved = resolve(root, document, raw)
                if resolved is not None:
                    normalized, _ = resolved
                    references.add(Reference(relative_document, raw, normalized, "code span"))
        for line in command_lines(text):
            for raw in command_path_tokens(root, line):
                resolved = resolve(root, document, raw, from_command=True)
                if resolved is not None:
                    normalized, _ = resolved
                    references.add(Reference(relative_document, raw, normalized, "command"))
    return sorted(references)


def validate(root: Path, references: list[Reference]) -> list[str]:
    violations: list[str] = []
    for reference in references:
        if reference.resolved == "outside":
            violations.append(
                f"OUTSIDE-REPO {reference.document} [{reference.source}: {reference.raw}]"
            )
            continue
        candidate = root / reference.resolved
        matches = glob.glob(str(candidate), recursive=True) if glob.has_magic(str(candidate)) else []
        exists = bool(matches) if glob.has_magic(str(candidate)) else candidate.exists()
        if not exists:
            violations.append(
                f"MISSING {reference.document} [{reference.source}: {reference.raw}] -> {reference.resolved}"
            )
            continue
        published_paths = matches if matches else [str(candidate)]
        for published in published_paths:
            path = Path(published)
            try:
                relative = path.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                violations.append(
                    f"OUTSIDE-REPO {reference.document} [{reference.source}: {reference.raw}]"
                )
                break
            if gitignored(root, relative):
                violations.append(
                    f"GITIGNORED {reference.document} [{reference.source}: {reference.raw}] -> {relative}"
                )
                break
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    args = parser.parse_args()
    root = args.root.resolve()
    references = collect_references(root)
    violations = validate(root, references)
    if violations:
        print("documentation reference contract failed.", file=sys.stderr)
        for violation in violations:
            print(f"  {violation}", file=sys.stderr)
        print(
            "Fix: publish the referenced input, correct the path, or remove the claim. "
            "Output destinations may use --output or --write without pre-existing.",
            file=sys.stderr,
        )
        return 1
    print(
        f"documentation reference contract: {len(references)} path-like code spans and command inputs resolve to published paths."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
