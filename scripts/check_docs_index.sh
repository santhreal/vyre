#!/usr/bin/env bash
# Verify documentation lifecycle coverage and generated navigation.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "$ROOT/scripts/docs_manifest.py" --check
