#!/usr/bin/env python3
"""Print publishable workspace members as ``directory:package`` rows."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path


def fail(message: str) -> "NoReturn":
    print(f"Fix: {message}", file=sys.stderr)
    raise SystemExit(2)


def main() -> None:
    if len(sys.argv) != 2:
        fail("public API inventory expects exactly one workspace-root argument")

    root = Path(sys.argv[1]).resolve()
    workspace_manifest = root / "Cargo.toml"
    try:
        workspace = tomllib.loads(workspace_manifest.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        fail(f"cannot read workspace manifest {workspace_manifest}: {error}")

    members = workspace.get("workspace", {}).get("members")
    if not isinstance(members, list) or not all(isinstance(member, str) for member in members):
        fail("workspace.members must remain an explicit string array")

    rows: list[tuple[str, str]] = []
    for member in members:
        manifest_path = root / member / "Cargo.toml"
        try:
            manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as error:
            fail(f"cannot read workspace member manifest {manifest_path}: {error}")

        package = manifest.get("package")
        if not isinstance(package, dict):
            fail(f"workspace member {member} has no [package] table")
        name = package.get("name")
        if not isinstance(name, str) or not name:
            fail(f"workspace member {member} has no non-empty package.name")

        publish = package.get("publish", True)
        if publish is False or publish == []:
            continue
        if publish is not True and not (
            isinstance(publish, list) and all(isinstance(registry, str) for registry in publish)
        ):
            fail(f"workspace member {member} has unsupported package.publish value {publish!r}")
        rows.append((member, name))

    for member, name in sorted(rows, key=lambda row: row[1]):
        print(f"{member}:{name}")


if __name__ == "__main__":
    main()
