#!/usr/bin/env python3
"""Generate release metadata views and validate the release prose contract."""

from __future__ import annotations

import argparse
import sys
import textwrap
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "release/release-train.toml"
FRAGMENTS_PATH = ROOT / "release/changes/unreleased.toml"
CHANGELOG_PATH = ROOT / "CHANGELOG.md"
LAUNCH_PATH = ROOT / "scripts/final-launch.sh"
CATEGORIES = ("Added", "Changed", "Deprecated", "Removed", "Fixed", "Security")


def load_toml(path: Path) -> dict:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"{path.relative_to(ROOT)}: {error}") from error


def normalize(text: str) -> str:
    return " ".join(text.split())


def validate_train(train: dict) -> list[str]:
    errors: list[str] = []
    versions = train.get("versions", {})
    groups = train.get("release_groups", {})
    actions = train.get("external_actions", [])
    if not isinstance(versions, dict) or not versions:
        errors.append("release/release-train.toml: [versions] must be non-empty")
    if not isinstance(groups, dict) or not groups:
        errors.append("release/release-train.toml: [release_groups] must be non-empty")
    package_owner: dict[str, str] = {}
    for name, group in groups.items() if isinstance(groups, dict) else ():
        version_key = group.get("version") if isinstance(group, dict) else None
        packages = group.get("packages") if isinstance(group, dict) else None
        repository = group.get("repository") if isinstance(group, dict) else None
        if not isinstance(repository, str) or repository.count("/") != 1:
            errors.append(f"release group `{name}` must declare an owner/repository")
        if version_key not in versions:
            errors.append(f"release group `{name}` references unknown version key `{version_key}`")
        if not isinstance(packages, list) or not packages:
            errors.append(f"release group `{name}` must declare at least one package")
            continue
        for package in packages:
            if not isinstance(package, str) or not package:
                errors.append(f"release group `{name}` contains an invalid package name")
                continue
            previous = package_owner.setdefault(package, name)
            if previous != name:
                errors.append(f"package `{package}` belongs to both `{previous}` and `{name}`")
    if not isinstance(actions, list) or len(actions) != 3:
        errors.append("release train must declare exactly three approval-gated external actions")
    else:
        ids = [action.get("id") for action in actions if isinstance(action, dict)]
        if len(ids) != len(set(ids)) or any(not value for value in ids):
            errors.append("external action ids must be non-empty and unique")
    return errors


def load_fragments() -> tuple[dict[str, list[str]], list[str]]:
    data = load_toml(FRAGMENTS_PATH)
    errors: list[str] = []
    if data.get("schema_version") != 1:
        errors.append("release/changes/unreleased.toml: schema_version must be 1")
    grouped = {category: [] for category in CATEGORIES}
    ids: set[str] = set()
    texts: set[str] = set()
    for index, fragment in enumerate(data.get("fragments", []), start=1):
        fragment_id = fragment.get("id")
        category = fragment.get("category")
        text = normalize(fragment.get("text", ""))
        if not fragment_id or fragment_id in ids:
            errors.append(f"fragment {index}: id must be non-empty and unique")
        else:
            ids.add(fragment_id)
        if category not in grouped:
            errors.append(f"fragment `{fragment_id}`: category `{category}` is not supported")
            continue
        if not text or text in texts:
            errors.append(f"fragment `{fragment_id}`: text must be non-empty and unique")
            continue
        texts.add(text)
        grouped[category].append(text)
    if not texts:
        errors.append("release/changes/unreleased.toml: at least one fragment is required")
    return grouped, errors


def render_train_identities(train: dict) -> list[str]:
    """The artifact identities a release has to name, stated from the train.

    `release/release-train.toml` requires these tokens to appear in release
    prose. They used to live in a hand-maintained per-version notes file that
    could disagree with the train; generating them from the train instead means
    the requirement is met by construction rather than checked after the fact.
    """
    versions = train["versions"]
    tags = train["tags"]
    pinned = [
        token for token in train.get("required_release_note_tokens", []) if "@" in token
    ]
    lines = [
        f"Vyre {versions['vyre']} releases from candidate tag `{tags['vyre_rc']}`"
        f" and final tag `{tags['vyre']}`.",
    ]
    if pinned:
        lines.append(
            "Backend crates carried at that version: "
            + ", ".join(f"`{token}`" for token in pinned)
            + "."
        )
    return lines


def render_changelog_section(train: dict, grouped: dict[str, list[str]]) -> str:
    lines = ["## [Unreleased]", "", *render_train_identities(train)]
    for category in CATEGORIES:
        entries = grouped[category]
        if not entries:
            continue
        lines.extend(["", f"### {category}", ""])
        for text in entries:
            wrapped = textwrap.wrap(
                text,
                width=79,
                initial_indent="- ",
                subsequent_indent="  ",
                break_long_words=False,
                break_on_hyphens=False,
            )
            lines.extend(wrapped)
    return "\n".join(lines) + "\n"


def replace_unreleased(changelog: str, section: str) -> str:
    start = changelog.find("## [Unreleased]")
    if start < 0:
        raise ValueError("CHANGELOG.md: missing `## [Unreleased]` section")
    end = changelog.find("\n## [", start + len("## [Unreleased]"))
    if end < 0:
        raise ValueError("CHANGELOG.md: missing released section after `## [Unreleased]`")
    return changelog[:start] + section.rstrip() + "\n" + changelog[end:]


def collect_outputs(train: dict, grouped: dict[str, list[str]]) -> dict[Path, str]:
    changelog = CHANGELOG_PATH.read_text(encoding="utf-8")
    return {
        CHANGELOG_PATH: replace_unreleased(
            changelog, render_changelog_section(train, grouped)
        )
    }


def validate_launch_order() -> list[str]:
    launch = LAUNCH_PATH.read_text(encoding="utf-8")
    steps = (
        'git tag -a "$VYRE_RELEASE_TAG_VYRE_RC"',
        'git push origin "$VYRE_RELEASE_TAG_VYRE_RC"',
        "-- vyre-release-gate --prepublish",
        'VYRE_RELEASE_APPROVED="$VYRE_RELEASE_PUBLISH_APPROVAL_TOKEN" bash scripts/publish-release.sh',
        'git tag -a "$VYRE_RELEASE_TAG_VYRE"',
        'git push origin "$VYRE_RELEASE_TAG_VYRE"',
        'gh release create "$VYRE_RELEASE_TAG_VYRE"',
        "> release/evidence/final/public-launch-completion.json",
        "-- launch-state --output",
        "-- vyre-release-gate\n",
    )
    positions = [launch.find(step) for step in steps]
    errors: list[str] = []
    for step, position in zip(steps, positions):
        if position < 0:
            errors.append(f"scripts/final-launch.sh: missing guarded launch step `{step}`")
    if not errors and positions != sorted(positions):
        errors.append(
            "scripts/final-launch.sh: candidate tags, prepublication gate, publish, "
            "final tags, release record, completion evidence, and final gate are out of order"
        )
    return errors


def validate_prose(train: dict) -> list[str]:
    """Hold the changelog to the tokens the release train requires.

    These tokens were checked against a generated per-version notes file and a
    prose runbook, both of which are gone with the rest of the book. The
    requirement is not: a release still has to state these facts somewhere a
    reader will find them, and the changelog is now the only place there is.
    """
    errors: list[str] = []
    changelog = CHANGELOG_PATH.read_text(encoding="utf-8")
    for token in train.get("required_release_note_tokens", []):
        if token not in changelog:
            errors.append(f"CHANGELOG.md: missing required release token `{token}`")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()

    try:
        train = load_toml(TRAIN_PATH)
        grouped, errors = load_fragments()
        errors.extend(validate_train(train))
        if errors:
            raise ValueError("\n".join(errors))
        outputs = collect_outputs(train, grouped)
        if args.write:
            for path, content in outputs.items():
                path.write_text(content, encoding="utf-8")
            print("release-docs: wrote the changelog")
            return 0
        errors = validate_prose(train)
        errors.extend(validate_launch_order())
        for path, expected in outputs.items():
            actual = path.read_text(encoding="utf-8")
            if actual != expected:
                errors.append(f"{path.relative_to(ROOT)}: generated release content is stale; run `python3 scripts/release_docs.py --write`")
        if errors:
            raise ValueError("\n".join(errors))
        print("release-docs: release train, fragments, and changelog agree")
        return 0
    except (KeyError, OSError, ValueError) as error:
        print(f"release-docs: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
