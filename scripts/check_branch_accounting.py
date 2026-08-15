#!/usr/bin/env python3
"""Fail when the campaign's own branch and worktree state is inconsistent.

Forgetting a branch is the failure mode that has cost this campaign most: a
branch with no worktree and no owner is nobody's work, and a worktree whose
branch already landed is a source tree that greps, gates and duplication scans
walk for nothing. Both states are derivable from git, so neither needs a
human to notice it.

The three invariants, checked against the repository at run time:

  1. Every local branch other than the integration branch, `main` and the
     branch checked out here is either an ancestor of integration (merged) or
     has a live worktree (being worked).
  2. Every worktree maps to a branch that is not yet an ancestor of
     integration. A worktree whose work has landed is stale.
  3. Nothing is both: a branch that is an ancestor of integration must not
     still have a worktree.

Run it from any checkout of the repository:

    python3 scripts/check_branch_accounting.py

Exit 0 when the state is consistent, 1 with one line per violation otherwise.
"""

from __future__ import annotations

import subprocess
import sys

INTEGRATION = "integration"
# The declared tier between a leaf branch and integration.
SUBSYSTEM_PREFIX = "subsystem-"
EXEMPT = {"main", INTEGRATION}
# A repository with almost no branches cannot exercise these invariants, and a
# derivation that silently yields nothing is the same defect as no check.
MINIMUM_BRANCHES = 2


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise SystemExit(
            f"check-branch-accounting: `git {' '.join(args)}` failed: "
            f"{result.stderr.strip()}"
        )
    return result.stdout


def local_branches() -> list[str]:
    return [
        line.strip()
        for line in git("for-each-ref", "--format=%(refname:short)", "refs/heads").splitlines()
        if line.strip()
    ]


def worktree_branches() -> dict[str, str]:
    """Branch name to worktree path, for every worktree with a branch."""
    mapping: dict[str, str] = {}
    path = ""
    for line in git("worktree", "list", "--porcelain").splitlines():
        if line.startswith("worktree "):
            path = line[len("worktree ") :]
        elif line.startswith("branch refs/heads/"):
            mapping[line[len("branch refs/heads/") :]] = path
    return mapping


def contained_in(target: str) -> set[str]:
    merged = git("branch", "--format=%(refname:short)", "--merged", target)
    return {line.strip() for line in merged.splitlines() if line.strip()}


def landed(branches: list[str], worktrees: dict[str, str]) -> set[str]:
    """Branches whose commits an owner branch already holds.

    The topology is leaf branches into subsystem branches into integration, so
    a leaf merged into a subsystem branch is accounted for even though it has
    not reached integration yet. Anchoring only on integration would make the
    check demand that every merged leaf keep its worktree, which is the
    opposite of what it is for. A branch always contains itself, so an owner
    never accounts for itself.
    """
    if not any(branch == INTEGRATION for branch in branches):
        raise SystemExit(
            f"check-branch-accounting: no local `{INTEGRATION}` branch; this "
            "check derives merged state from it and cannot run without it."
        )
    del worktrees
    owners = [INTEGRATION] + [b for b in branches if b.startswith(SUBSYSTEM_PREFIX)]
    held: set[str] = set()
    for owner in owners:
        held |= contained_in(owner) - {owner}
    return held


def main() -> int:
    branches = local_branches()
    if len(branches) < MINIMUM_BRANCHES:
        raise SystemExit(
            f"check-branch-accounting: found {len(branches)} local branch(es), "
            f"expected at least {MINIMUM_BRANCHES}; the branch derivation is "
            "broken, not the repository."
        )

    worktrees = worktree_branches()
    merged = landed(branches, worktrees)
    here = git("rev-parse", "--abbrev-ref", "HEAD").strip()

    violations: list[str] = []

    for branch in sorted(branches):
        if branch in EXEMPT or branch == here:
            continue
        if branch in merged and branch in worktrees:
            violations.append(
                f"`{branch}` is already held by an owner branch and still has "
                f"a worktree at {worktrees[branch]}. Fix: `git worktree remove "
                f"{worktrees[branch]}`; a merged branch kept alive by a stale "
                "tree is walked by every scan for nothing."
            )
        elif branch not in merged and branch not in worktrees:
            violations.append(
                f"`{branch}` is in no owner branch and has no worktree, so "
                "nobody is working it and nobody is merging it. Fix: give it "
                "a worktree and an owner, or merge it."
            )

    for branch, path in sorted(worktrees.items()):
        if branch in EXEMPT or branch == here:
            continue
        if branch in merged:
            continue
        if branch not in branches:
            violations.append(
                f"worktree {path} is on `{branch}`, which is not a local "
                "branch. Fix: the worktree outlived its branch."
            )

    if violations:
        print(
            "check-branch-accounting: the campaign's branch and worktree state "
            "is inconsistent:",
            file=sys.stderr,
        )
        for violation in violations:
            print(f"  {violation}", file=sys.stderr)
        return 1

    print(
        f"check-branch-accounting: {len(branches)} branch(es), "
        f"{len(worktrees)} worktree(s), all accounted for"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
