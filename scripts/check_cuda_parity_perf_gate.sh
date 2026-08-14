#!/usr/bin/env bash
# check_cuda_parity_perf_gate.sh
# Runs the vyre-driver-cuda test suite on a live NVIDIA device.
#
# The roster is derived from tracked test targets, not listed here. Two
# hardcoded lists used to decide what ran, and both rotted: the contract list
# still named cuda_device_contract after 8a66e4b65b deleted it, so the gate
# exited at its first target and measured nothing at all, and the other list
# matched only *gpu_parity*, which left the crate's remaining targets ungated on
# the one runner that has a device.
#
# Deriving it also means a parity target added later is covered by being a test
# target, which is how the packed INT4 extension ops arrive through
# int4_quantized_gpu_parity rather than through an entry in this file.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! nvidia-smi >/dev/null 2>&1; then
    echo "CUDA gate requires a live NVIDIA GPU, but nvidia-smi failed." >&2
    echo "Fix: repair CUDA/NVIDIA driver visibility; do not skip CUDA parity on this GPU fleet." >&2
    exit 1
fi

source scripts/lib/cargo_runner.sh
vyre_select_cargo_runner

# Tracked files only, and only the crate's own test directory: a stray .rs left
# in the working tree is not a target, and a nested support module is not one
# either.
mapfile -t TARGETS < <(
    git ls-files -- 'vyre-driver-cuda/tests/*.rs' \
        | awk -F/ 'NF == 3 { sub(/\.rs$/, "", $3); print $3 }' \
        | sort -u
)

if [ "${#TARGETS[@]}" -eq 0 ]; then
    echo "CUDA gate found no tracked test target under vyre-driver-cuda/tests." >&2
    echo "Fix: this gate must run something. Reporting a clean device with nothing to run is the defect it guards." >&2
    exit 1
fi

PARITY_COUNT=0
for target in "${TARGETS[@]}"; do
    case "$target" in
        *gpu_parity*) PARITY_COUNT=$((PARITY_COUNT + 1)) ;;
    esac
done

if [ "$PARITY_COUNT" -eq 0 ]; then
    echo "CUDA gate found no *gpu_parity* target among ${#TARGETS[@]} test targets." >&2
    echo "Fix: reference parity against the live device is the evidence this gate exists to produce." >&2
    exit 1
fi

echo "CUDA gate: ${#TARGETS[@]} tracked test targets, ${PARITY_COUNT} of them gpu_parity, on:"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

# The crate's documented test command, so the gate and docs/testing cannot
# disagree about what proving this backend means.
"$CARGO_RUNNER" test -p vyre-driver-cuda -- --nocapture

echo "CUDA gate: all ${#TARGETS[@]} vyre-driver-cuda test targets passed on the live device."
