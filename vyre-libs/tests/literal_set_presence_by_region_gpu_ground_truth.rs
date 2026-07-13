//! REAL-GPU INDEPENDENT ground-truth gate for `GpuLiteralSet::scan_presence_by_region`
//! (the exact consumer path keyhog dispatches for its coalesced region batches).
//!
//! The CPU-reference twin (`literal_set_presence_by_region_ground_truth`) proves
//! the region-presence PROGRAM's semantics, but it evaluates the IR on the
//! reference backend, which has no subgroups and no wgpu lowering. keyhog's open
//! W1-1 under-fire ("GPU region-presence under-fired ... recovered by CPU recall
//! floor, fix the vyre literal-set path before treating GPU-only as
//! parity-safe") is exactly the class a reference-only gate cannot see: a
//! divergence that manifests only on device (subgroup divergence, wgpu WGSL
//! lowering, staging/readback).
//!
//! This gate dispatches the REAL `scan_presence_by_region` method on the wgpu
//! backend and compares the per-region bitmap against the SAME independent
//! plain-Rust DFA oracle the reference twin uses (`presence_oracle`). Any
//! divergence is an on-device under-fire (oracle set, GPU clear) or over-fire at
//! the source. Skips cleanly when no GPU is available.
//!
//! Run:
//!   cargo test -p vyre-libs --test literal_set_presence_by_region_gpu_ground_truth \
//!     --release -- --nocapture

mod presence_oracle;

use presence_oracle::{
    assert_presence_matches, edge_cases, gpu_only_large_scale_cases, random_haystack,
    random_literals, random_region_starts, scale_cases, Lcg,
};
use vyre_driver_wgpu::WgpuBackend;
use vyre_libs::scan::GpuLiteralSet;

/// Dispatch the real GPU region-presence scan and assert it matches the oracle.
fn check_gpu(
    backend: &WgpuBackend,
    literals: &[Vec<u8>],
    haystack: &[u8],
    region_starts: &[u32],
    label: &str,
) {
    let pattern_refs: Vec<&[u8]> = literals.iter().map(Vec::as_slice).collect();
    let matcher = GpuLiteralSet::compile(&pattern_refs);
    let produced = matcher
        .scan_presence_by_region(backend, haystack, region_starts)
        .unwrap_or_else(|e| panic!("[{label}] scan_presence_by_region dispatch failed: {e}"));
    assert_presence_matches(literals, haystack, region_starts, &produced, label);
}

#[test]
fn gpu_region_presence_matches_independent_dfa_oracle() {
    let backend = match WgpuBackend::shared() {
        Ok(backend) => backend,
        Err(e) => {
            eprintln!("no wgpu backend ({e}); skipping GPU region-presence ground-truth gate");
            return;
        }
    };

    // Every W1-1 edge class, on device.
    for (label, literals, haystack, region_starts) in edge_cases() {
        check_gpu(
            backend.as_ref(),
            &literals,
            &haystack,
            &region_starts,
            &label,
        );
    }

    // keyhog-shaped scale: multi-word presence rows, many small regions, full
    // byte-range patterns (the on-device shape closest to keyhog's real batch).
    for (label, literals, haystack, region_starts) in scale_cases() {
        check_gpu(
            backend.as_ref(),
            &literals,
            &haystack,
            &region_starts,
            &label,
        );
    }

    // GPU-ONLY full-magnitude class: ~6,000 patterns / ~1,000 regions, beyond the
    // CPU reference interpreter's tractable range, but a single AC-walk oracle on
    // device. This is the "full literal-set size on the GPU gate" W6-1 names.
    for (label, literals, haystack, region_starts) in gpu_only_large_scale_cases() {
        check_gpu(
            backend.as_ref(),
            &literals,
            &haystack,
            &region_starts,
            &label,
        );
    }

    // Randomized volume: the GPU dispatch is ~ms/case, so default to a modest
    // count and let the env scale it for nightly/thorough runs.
    let cases: usize = std::env::var("VYRE_PRESENCE_GPU_GROUND_TRUTH_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);
    let mut rng = Lcg::new(0x6770_755f_6774u64);
    for case in 0..cases {
        let literals = random_literals(&mut rng);
        let haystack = random_haystack(&mut rng);
        let region_starts = random_region_starts(&mut rng, haystack.len());
        check_gpu(
            backend.as_ref(),
            &literals,
            &haystack,
            &region_starts,
            &format!("gpu case {case}"),
        );
    }
}
