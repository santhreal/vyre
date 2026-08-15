#!/usr/bin/env python3
"""Resolve the workspace cargo runner for Python release tooling.

This is the Python half of `scripts/lib/cargo_runner.sh` and follows the same
order: an explicit `VYRE_CARGO_RUNNER`, then the checked-in `./cargo_full`
wrapper, then bare `cargo`. A gate that spells `cargo` itself bypasses the
wrapper's build configuration, which is the one place this workspace declares
job count and compiler environment.
"""

from __future__ import annotations

import os
from pathlib import Path


def cargo_runner(root: Path) -> str:
    """The cargo executable release tooling under `root` must invoke."""
    override = os.environ.get("VYRE_CARGO_RUNNER")
    if override:
        return override
    wrapper = root / "cargo_full"
    if os.access(wrapper, os.X_OK):
        return str(wrapper)
    return "cargo"
