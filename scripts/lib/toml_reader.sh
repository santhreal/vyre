#!/usr/bin/env bash
# Shared TOML value reader for release shell helpers.

vyre_read_toml_values() {
    if [[ "$#" -lt 4 ]]; then
        printf 'Fix: vyre_read_toml_values requires MANIFEST, LABEL, EXPECTED_COUNT, and at least one key.\n' >&2
        return 2
    fi
    local manifest="$1"
    local label="$2"
    local expected_count="$3"
    shift 3
    if [[ "$#" -ne "$expected_count" ]]; then
        printf 'Fix: %s requested %s TOML key(s), expected %s.\n' "$label" "$#" "$expected_count" >&2
        return 2
    fi
    local output
    if ! command -v python3 >/dev/null 2>&1; then
        printf 'Fix: python3 with tomllib is required to read %s.\n' "$manifest" >&2
        return 2
    fi
    local reader
    reader="$(dirname "${BASH_SOURCE[0]}")/read_toml_values.py"
    if [[ ! -f "$reader" ]]; then
        printf 'Fix: %s is missing; restore the TOML reader helper.\n' "$reader" >&2
        return 2
    fi
    if ! output="$(python3 "$reader" "$manifest" "$label" "$@")"; then
        return 2
    fi
    VYRE_TOML_VALUES=()
    while IFS= read -r value; do
        VYRE_TOML_VALUES+=("$value")
    done <<< "$output"
    if [[ "${#VYRE_TOML_VALUES[@]}" -ne "$expected_count" ]]; then
        printf 'Fix: %s produced %s %s value(s), expected %s.\n' "$manifest" "${#VYRE_TOML_VALUES[@]}" "$label" "$expected_count" >&2
        return 2
    fi
}
