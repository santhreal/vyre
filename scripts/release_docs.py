#!/usr/bin/env python3
"""Generate release metadata views and validate the release prose contract."""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "release/release-train.toml"
FRAGMENTS_DIR = ROOT / "release/changes/unreleased"
CHANGELOG_PATH = ROOT / "CHANGELOG.md"
NOTES_PATH = ROOT / "release/evidence/docs/release-notes-body.md"
LAUNCH_PATH = ROOT / "scripts/final-launch.sh"
CATEGORIES = ("Added", "Changed", "Deprecated", "Removed", "Fixed", "Security")
FRAGMENT_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


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
    """Read one fragment per file, named for the id it carries.

    Every fragment used to be a `[[fragments]]` table appended to one file, and
    five three-way merges dropped a shared header line because both branches
    appended at the same offset against identical context. The second fragment's
    keys then landed inside the first, the file stopped being valid TOML, and
    every verdict behind the parse was unreachable. A `merge=union` attribute
    did not help, because there was no conflict to resolve. One file per
    fragment removes the shared line: two branches adding two fragments add two
    paths, and a merge that touches neither file cannot fuse them.

    The file name is the id, so the id is unique by construction rather than by
    a check that runs after the file already parsed. Order within a category is
    the sorted id, so the rendered changelog does not depend on the order a
    directory listing happens to arrive in.
    """
    errors: list[str] = []
    grouped = {category: [] for category in CATEGORIES}
    texts: dict[str, str] = {}
    if not FRAGMENTS_DIR.is_dir():
        return grouped, [f"{FRAGMENTS_DIR.relative_to(ROOT)}: fragment directory is missing"]
    paths = sorted(FRAGMENTS_DIR.iterdir())
    for path in paths:
        name = path.relative_to(ROOT)
        if path.suffix != ".toml" or not path.is_file():
            errors.append(f"{name}: fragment files are named `<id>.toml`")
            continue
        fragment_id = path.stem
        if not FRAGMENT_ID.fullmatch(fragment_id):
            errors.append(f"{name}: fragment id must match {FRAGMENT_ID.pattern}")
            continue
        fragment = load_toml(path)
        unknown = sorted(set(fragment) - {"category", "text"})
        if unknown:
            errors.append(f"{name}: unknown key(s) {', '.join(unknown)}")
            continue
        category = fragment.get("category")
        if category not in grouped:
            errors.append(f"{name}: category `{category}` is not supported")
            continue
        text = normalize(fragment.get("text", ""))
        if not text:
            errors.append(f"{name}: text must be non-empty")
            continue
        if text in texts:
            errors.append(f"{name}: text repeats `{texts[text]}`")
            continue
        texts[text] = fragment_id
        grouped[category].append(text)
    if not texts and not errors:
        errors.append(
            f"{FRAGMENTS_DIR.relative_to(ROOT)}: at least one fragment is required"
        )
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


def render_release_notes_body(train: dict, grouped: dict[str, list[str]]) -> str:
    """The body `gh release create --notes-file` attaches to the final tag.

    `scripts/final-launch.sh` pointed at `docs/release/v<version>.md`, a page the
    deleted mdbook carried and nothing regenerates, so the last outward step of a
    release would have failed on a missing file after the crates were already
    published. The notes are the same section the changelog carries, under a
    heading naming the tag instead of `Unreleased`, so there is one authored
    source for what a release says it contains.
    """
    section = render_changelog_section(train, grouped)
    return section.replace("## [Unreleased]", f"# {train['tags']['vyre']}", 1)


def collect_outputs(train: dict, grouped: dict[str, list[str]]) -> dict[Path, str]:
    changelog = CHANGELOG_PATH.read_text(encoding="utf-8")
    return {
        CHANGELOG_PATH: replace_unreleased(
            changelog, render_changelog_section(train, grouped)
        ),
        NOTES_PATH: render_release_notes_body(train, grouped),
    }


def validate_launch_order() -> list[str]:
    launch = LAUNCH_PATH.read_text(encoding="utf-8")
    steps = (
        'git tag -a "$VYRE_RELEASE_TAG_VYRE_RC"',
        'git push origin "$VYRE_RELEASE_TAG_VYRE_RC"',
        "-- vyre-release-gate\n",
        'VYRE_RELEASE_APPROVED="$VYRE_RELEASE_PUBLISH_APPROVAL_TOKEN" bash scripts/publish-release.sh',
        'git tag -a "$VYRE_RELEASE_TAG_VYRE"',
        'git push origin "$VYRE_RELEASE_TAG_VYRE"',
        'gh release create "$VYRE_RELEASE_TAG_VYRE"',
        "> release/evidence/final/public-launch-completion.json",
        "-- launch-state --output",
        "-- vyre-release-gate --launch-complete\n",
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
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            print("release-docs: wrote the changelog and the release notes body")
            return 0
        errors = validate_prose(train)
        errors.extend(validate_launch_order())
        for path, expected in outputs.items():
            actual = path.read_text(encoding="utf-8")
            if actual != expected:
                errors.append(f"{path.relative_to(ROOT)}: generated release content is stale; run `python3 scripts/release_docs.py --write`")
        if errors:
            raise ValueError("\n".join(errors))
        print("release-docs: release train, fragments, changelog, and release notes agree")
        return 0
    except (KeyError, OSError, ValueError) as error:
        print(f"release-docs: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
