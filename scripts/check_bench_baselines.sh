#!/usr/bin/env bash
# Gap #11  -  bench baselines published.
#
# Every criterion bench gets a baseline committed to benches/RESULTS.md
# with machine spec + commit hash + numbers. Competitors (wgpu, naga,
# rust-gpu) publish these; without them, "vyre is fast" is a claim.

set -euo pipefail
cd "$(dirname "$0")/.."

RESULTS="benches/RESULTS.md"

if [ ! -f "$RESULTS" ]; then
    echo "gap #11: $RESULTS does not exist" >&2
    echo "  fix: run the baseline capture script and commit the output" >&2
    exit 1
fi

# Required header fields
required_fields=("machine:" "gpu:" "cpu:" "rustc:" "commit:")
for field in "${required_fields[@]}"; do
    if ! grep -q "$field" "$RESULTS"; then
        echo "gap #11: $RESULTS missing required field '$field'" >&2
        exit 1
    fi
done

# Every crate that declares a bench target must have a section in RESULTS.md.
# A crate qualifies by owning at least one bench source file, not by owning a
# directory called benches. A benches directory that holds only documentation
# owns no target, and a directory-name search demanded a measured section for a
# crate `cargo bench` cannot run.
while IFS= read -r bench_source; do
    name=$(basename "$(dirname "$(dirname "$bench_source")")")
    if ! grep -q "^### $name\b" "$RESULTS"; then
        echo "gap #11: $RESULTS missing section for $name" >&2
        echo "  fix: run 'cargo bench -p $name' and record the numbers under a '### $name' heading" >&2
        exit 1
    fi
done < <(find . -path '*/benches/*.rs' -not -path '*/target/*' -not -path './benches/*' | sort)

echo "gap #11: baseline file present and covers every crate with a bench target"
