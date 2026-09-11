#!/usr/bin/env bash
# Shared TOML value reader for release shell helpers.
#
# The reader is bash rather than python: macOS ships Python 3.9, `tomllib`
# arrived in 3.11, and the release helpers ran on a host where the import failed
# and the loader returned an unset tag list. What these helpers read is one
# scalar per dotted key out of a manifest this repository writes, so the subset
# below is the whole requirement: table headers, dotted key paths, quoted
# strings, and bare scalars. Anything outside it is refused by name instead of
# guessed at, because a guessed release tag is worse than a failed read.

# Trim leading and trailing whitespace from $1 into VYRE_TOML_TRIMMED.
vyre_toml_trim() {
    local text="$1"
    text="${text#"${text%%[![:space:]]*}"}"
    text="${text%"${text##*[![:space:]]}"}"
    VYRE_TOML_TRIMMED="$text"
}

# Print the scalar at dotted key $2 in manifest $1.
#
# Returns 1 when no such key exists and 3 when the key exists but holds
# something this reader does not read: an array, an inline table, or a string
# carrying an escape.
vyre_toml_scalar() {
    local manifest="$1"
    local wanted="$2"
    local table=""
    local line name value full
    while IFS= read -r line || [[ -n "$line" ]]; do
        line="${line%$'\r'}"
        vyre_toml_trim "$line"
        line="$VYRE_TOML_TRIMMED"
        case "$line" in
            '' | '#'*)
                continue
                ;;
            '[['*)
                # An array-of-tables entry is not addressable by a dotted key,
                # and the brackets keep its keys from matching one.
                table="$line"
                continue
                ;;
            '['*']')
                table="${line#\[}"
                table="${table%\]}"
                vyre_toml_trim "$table"
                table="$VYRE_TOML_TRIMMED"
                continue
                ;;
        esac
        name="${line%%=*}"
        if [[ "$name" == "$line" ]]; then
            continue
        fi
        value="${line#*=}"
        vyre_toml_trim "$name"
        name="$VYRE_TOML_TRIMMED"
        name="${name%\"}"
        name="${name#\"}"
        vyre_toml_trim "$value"
        value="$VYRE_TOML_TRIMMED"
        full="$name"
        if [[ -n "$table" ]]; then
            full="$table.$name"
        fi
        if [[ "$full" != "$wanted" ]]; then
            continue
        fi
        case "$value" in
            '"'*)
                value="${value#\"}"
                value="${value%%\"*}"
                ;;
            "'"*)
                value="${value#\'}"
                value="${value%%\'*}"
                ;;
            '[' | '['* | '{'*)
                return 3
                ;;
            *)
                value="${value%%#*}"
                vyre_toml_trim "$value"
                value="$VYRE_TOML_TRIMMED"
                ;;
        esac
        if [[ "$value" == *\\* ]]; then
            return 3
        fi
        printf '%s\n' "$value"
        return 0
    done <"$manifest"
    return 1
}

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
    if [[ ! -f "$manifest" ]]; then
        printf 'Fix: %s is missing; restore the manifest the %s reader names.\n' "$manifest" "$label" >&2
        return 2
    fi
    VYRE_TOML_VALUES=()
    local key value status
    for key in "$@"; do
        value="$(vyre_toml_scalar "$manifest" "$key")"
        status="$?"
        if [[ "$status" -eq 3 ]]; then
            printf 'Fix: %s %s key %s must be a scalar value this reader reads.\n' "$manifest" "$label" "$key" >&2
            return 2
        fi
        if [[ "$status" -ne 0 ]]; then
            printf 'Fix: %s is missing required %s key %s.\n' "$manifest" "$label" "$key" >&2
            return 2
        fi
        VYRE_TOML_VALUES+=("$value")
    done
    if [[ "${#VYRE_TOML_VALUES[@]}" -ne "$expected_count" ]]; then
        printf 'Fix: %s produced %s %s value(s), expected %s.\n' "$manifest" "${#VYRE_TOML_VALUES[@]}" "$label" "$expected_count" >&2
        return 2
    fi
}
