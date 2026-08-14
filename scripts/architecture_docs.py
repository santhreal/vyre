#!/usr/bin/env python3
"""Validate current architecture documentation against live repository authorities."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

MAX_INPUT_BYTES = 16_777_216
CURRENT_DOCS = [
    Path("docs/ARCHITECTURE.md"),
    Path("docs/OPTIMIZATION_ARCHITECTURE.md"),
    Path("docs/RUNTIME_PIPELINE.md"),
    Path("docs/megakernel-wiring.md"),
]
RFC = Path("docs/rfcs/0005-persistent-megakernel.md")
MANIFEST = Path("docs/DOCS.toml")


# Wire version of docs/generated/OP_SCHEMA.json. xtask owns the generator
# (xtask/src/operation_schema.rs SCHEMA_VERSION) and pins the same number;
# its test the_python_contract_pins_the_same_operation_schema_version fails
# when this line drifts from it.
OPERATION_SCHEMA_VERSION = 4


class ContractError(Exception):
    """Architecture prose disagrees with a live authority."""


def read_text(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError as error:
        raise ContractError(f"could not inspect `{path}`: {error}") from error
    if size > MAX_INPUT_BYTES:
        raise ContractError(f"`{path}` exceeds the {MAX_INPUT_BYTES}-byte input limit")
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ContractError(f"could not read `{path}`: {error}") from error


def read_toml(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(read_text(path))
    except tomllib.TOMLDecodeError as error:
        raise ContractError(f"could not parse TOML `{path}`: {error}") from error


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(read_text(path))
    except json.JSONDecodeError as error:
        raise ContractError(f"could not parse JSON `{path}`: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"`{path}` must contain an object")
    return value


def normalize(text: str) -> str:
    return " ".join(text.split())


def require_tokens(path: Path, text: str, tokens: list[str]) -> None:
    normalized_text = normalize(text)
    for token in tokens:
        if normalize(token) not in normalized_text:
            raise ContractError(f"`{path}` is missing current architecture token `{token}`")


def forbid_patterns(path: Path, text: str, patterns: list[str]) -> None:
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            raise ContractError(f"`{path}` retains stale architecture pattern `{pattern}`")


def verification_date(path: Path, text: str) -> str:
    match = re.search(r"^Last verified: (\d{4}-\d{2}-\d{2})$", text, re.MULTILINE)
    if match is None:
        raise ContractError(f"`{path}` has no Last verified date")
    return match.group(1)


def crate_rows(ownership: dict[str, Any]) -> list[dict[str, Any]]:
    rows = ownership.get("crate")
    if not isinstance(rows, list):
        raise ContractError("ownership registry has no [[crate]] table")
    out: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            out.append(row)
    if not out:
        raise ContractError("ownership registry has no crate rows")
    return out


def validate(root: Path) -> None:
    root = root.resolve()
    workspace = read_toml(root / "Cargo.toml")
    members = workspace.get("workspace", {}).get("members")
    if not isinstance(members, list) or not members:
        raise ContractError("workspace.members must be a non-empty array")
    if any(not isinstance(member, str) or "*" in member for member in members):
        raise ContractError("workspace members must be explicit paths")
    if "vyre-megakernel" not in members:
        raise ContractError("workspace.members must include vyre-megakernel")

    train = read_toml(root / "release/release-train.toml")
    version = train.get("versions", {}).get("vyre")
    if not isinstance(version, str):
        raise ContractError("release train has no versions.vyre")

    operation_schema = read_json(root / "docs/generated/OP_SCHEMA.json")
    operations = operation_schema.get("operations")
    tier_counts = operation_schema.get("tier_counts")
    if (
        operation_schema.get("schema_version") != OPERATION_SCHEMA_VERSION
        or not isinstance(operations, list)
        or not isinstance(tier_counts, dict)
        or operation_schema.get("operation_count") != len(operations)
        or sum(tier_counts.values()) != len(operations)
    ):
        raise ContractError("operation schema is not internally coherent")

    backend = read_json(root / "release/evidence/backends/backend-matrix.json")
    if backend.get("blockers") != []:
        raise ContractError(f"backend evidence has blockers: {backend.get('blockers')}")
    preferred = backend.get("preferred_backend_id")
    backend_rows = backend.get("backends")
    if not isinstance(preferred, str) or not isinstance(backend_rows, list):
        raise ContractError("backend evidence has no preferred backend or probe rows")
    if preferred not in {row.get("id") for row in backend_rows if isinstance(row, dict)}:
        raise ContractError("preferred backend has no executable probe row")

    ownership = read_toml(root / "docs/CRATE_OWNERSHIP.toml")
    megakernel_rows = [
        row
        for row in crate_rows(ownership)
        if row.get("package") == "vyre-megakernel" or row.get("path") == "vyre-megakernel"
    ]
    if not megakernel_rows:
        raise ContractError("ownership registry has no current vyre-megakernel crate row")
    compiler_responsibility = megakernel_rows[0].get("responsibility")
    if not isinstance(compiler_responsibility, str) or "ProgramGraph" not in compiler_responsibility:
        raise ContractError("vyre-megakernel compiler responsibility is incomplete")
    if ownership.get("planned", {}).get("vyre-megakernel") is not None:
        raise ContractError("ownership registry must not keep planned.vyre-megakernel after the crate exists")

    docs_manifest = read_toml(root / MANIFEST)
    page_rows = docs_manifest.get("page")
    if not isinstance(page_rows, list):
        raise ContractError(f"`{MANIFEST}` has no [[page]] rows")
    page_status = {
        Path("docs") / row["path"]: row.get("status")
        for row in page_rows
        if isinstance(row, dict) and isinstance(row.get("path"), str)
    }
    texts: dict[Path, str] = {}
    for path in CURRENT_DOCS:
        text = read_text(root / path)
        texts[path] = text
        verification_date(path, text)
        if version not in text:
            raise ContractError(f"`{path}` is not verified against Vyre {version}")
        if page_status.get(path) != "current":
            raise ContractError(f"`{MANIFEST}` must classify `{path}` as current")

    rfc_text = read_text(root / RFC)
    texts[RFC] = rfc_text
    verification_date(RFC, rfc_text)
    if "Status: **Superseded**" not in rfc_text:
        raise ContractError(f"`{RFC}` must be explicitly superseded")
    if page_status.get(RFC) != "superseded":
        raise ContractError(f"`{MANIFEST}` must classify `{RFC}` as superseded")

    stale_absent = [
        r"planned\s+`?vyre-megakernel`?",
        r"planned compiler crate",
        r"not a current workspace",
        r"not present in the current",
        r"Until that crate exists",
        r"Until that boundary ships",
        r"declared target rather than a shipped package",
    ]
    for path, text in texts.items():
        forbidden = [
            r"\b0\.6(?:\.x)?\b",
            r"\b9[- ]op\b",
            r"\bnine[- ]op\b",
            r"WGPU[^\n]{0,40}primary production path",
            r"## Four CI laws",
            r"codex-[0-9a-z]+",
        ]
        forbid_patterns(path, text, forbidden)
        if path in CURRENT_DOCS or path == RFC:
            forbid_patterns(path, text, stale_absent)

    architecture = texts[Path("docs/ARCHITECTURE.md")]
    require_tokens(
        Path("docs/ARCHITECTURE.md"),
        architecture,
        [
            "generated/OP_SCHEMA.json",
            "vyre-foundation::operation::OperationRegistry",
            "do not own shadow operation identities",
            "Cross-program composition",
            "vyre-megakernel",
            "Artifact",
            "vyre-runtime",
            "bytecode interpreter",
        ],
    )

    require_tokens(
        Path("docs/OPTIMIZATION_ARCHITECTURE.md"),
        texts[Path("docs/OPTIMIZATION_ARCHITECTURE.md")],
        [
            "Layer 1: semantic IR optimization",
            "Layer 2: concrete lowering strategy",
            "vyre-runtime/src/megakernel/",
            "vyre-megakernel",
            "vyre-foundation/src/optimizer/megakernel",
        ],
    )
    require_tokens(
        Path("docs/RUNTIME_PIPELINE.md"),
        texts[Path("docs/RUNTIME_PIPELINE.md")],
        [
            "vyre-runtime/src/pipeline_cache/",
            "vyre-runtime/src/megakernel/",
            "vyre-megakernel",
            "artifact_admission",
            "does not silently rerun",
            "not substitute for raw samples",
        ],
    )
    require_tokens(
        Path("docs/megakernel-wiring.md"),
        texts[Path("docs/megakernel-wiring.md")],
        [
            "starts from the same validated `Program`",
            "vyre-runtime/src/megakernel/",
            "vyre-megakernel",
            "Artifact",
            "vyre-driver/src/megakernel_execution",
            "vyre-foundation/src/optimizer/megakernel",
            "does not consume a general VIR bytecode interpreter",
        ],
    )
    require_tokens(
        RFC,
        rfc_text,
        [
            "Historical motivation",
            "Superseded design",
            "Current resolution",
            "does not support a general",
            "vyre-megakernel",
            "workspace member",
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="workspace root to validate")
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate only; exit non-zero on contract failure",
    )
    args = parser.parse_args()
    try:
        validate(args.root)
    except ContractError as error:
        print(f"architecture docs contract failed: {error}", file=sys.stderr)
        return 1
    if args.check:
        print("architecture docs contract ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
