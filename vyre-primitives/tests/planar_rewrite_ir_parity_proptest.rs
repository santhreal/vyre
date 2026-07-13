//! GPU-IR vs CPU-ref parity for `parsing::planar_rewrite_schedule`, the
//! non-overlapping 2D-grammar rewrite scheduler.
//!
//! Lane 0 walks the `h x w` candidate mask in row-major order; each candidate is
//! claimed into `chosen` unless a previously-chosen match lies within its
//! `k x k` exclusion zone (the cells `(r-di, c-dj)` for `di,dj in 0..k`). The
//! scan reads AND writes `chosen` in place, so it is a single serial lane-0
//! kernel: only lane 0 runs (the rest are `t==0`-gated no-ops), making the result
//! deterministic and idempotent under `reference_eval`'s cell-count grid
//! over-fire. Every shipped test drives `reference_planar_rewrite_schedule` or
//! checks Program shape; the actual in-place greedy IR (the exclusion-zone scan
//! reading back just-written `chosen` cells, the `r>=di && c>=dj` bound, the
//! per-cell clear-then-maybe-set) was never executed. A wrong exclusion radius,
//! an off-by-one bound, or a stale read all diverge here.
#![forbid(unsafe_code)]
#![cfg(all(feature = "parsing", feature = "cpu-parity"))]

use proptest::prelude::*;
use vyre_primitives::parsing::planar_rewrite::{
    planar_rewrite_schedule, reference_planar_rewrite_schedule,
};
use vyre_primitives::wire::{decode_u32_le_bytes_all as unpack, pack_u32_slice as pack};
use vyre_reference::value::Value;

/// Drive the real IR. Buffer binding order: candidates(0), chosen(1, RW).
fn gpu_schedule(candidates: &[u32], h: u32, w: u32, k: u32) -> Vec<u32> {
    let program = planar_rewrite_schedule("candidates", "chosen", h, w, k);
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack(candidates)),
            Value::from(pack(&vec![0u32; candidates.len()])),
        ],
    )
    .expect("planar_rewrite_schedule reference evaluation must succeed");
    unpack(&outputs[0].to_bytes())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1500))]

    #[test]
    fn ir_matches_reference_over_random_candidate_maps(
        h in 1u32..12,
        w in 1u32..12,
        k in 1u32..5,
        seed in any::<u64>(),
    ) {
        let mut rng = seed;
        let mut next = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng >> 32) as u32
        };
        // Dense candidate maps (~50% set) so exclusion conflicts are frequent.
        let candidates: Vec<u32> = (0..h * w).map(|_| (next() & 1)).collect();
        let expected = reference_planar_rewrite_schedule(&candidates, h, w, k);
        let got = gpu_schedule(&candidates, h, w, k);
        prop_assert_eq!(got, expected, "planar_rewrite IR diverged (h={}, w={}, k={})", h, w, k);
    }
}

/// Deterministic anchors: the greedy row-major first-wins choice, a full-dense
/// map producing a k-strided lattice, and the k=1 identity (every candidate
/// chosen, no exclusion).
#[test]
fn ir_matches_reference_on_boundary_maps() {
    // 3x3 all-candidate map, k=2: greedy row-major picks (0,0), which excludes
    // (0,1),(1,0),(1,1); next free candidate is (0,2), then (2,0),(2,2). The IR
    // and oracle must agree on this exact greedy set.
    let h = 3u32;
    let w = 3u32;
    let candidates = vec![1u32; (h * w) as usize];
    let expected = reference_planar_rewrite_schedule(&candidates, h, w, 2);
    let got = gpu_schedule(&candidates, h, w, 2);
    assert_eq!(got, expected, "dense 3x3 k=2 greedy schedule must match");
    // Sanity: (0,0) is always chosen first, and (1,1) is excluded by it.
    assert_eq!(expected[0], 1, "top-left is greedily chosen");
    assert_eq!(
        expected[(1 * w + 1) as usize],
        0,
        "diagonal neighbor excluded by k=2 zone"
    );

    // k=1: exclusion zone is the single cell itself, so every candidate is chosen
    // (no conflict) -> chosen == candidates.
    let candidates = vec![1u32, 0, 1, 1, 1, 0, 0, 1, 1];
    let identity = reference_planar_rewrite_schedule(&candidates, 3, 3, 1);
    assert_eq!(identity, candidates, "k=1 chooses every candidate");
    assert_eq!(
        gpu_schedule(&candidates, 3, 3, 1),
        identity,
        "k=1 IR identity must match"
    );

    // Single row, k=3: candidates at every column; greedy picks col 0, excludes
    // cols within k-1=2 to the right's zone as the scan advances.
    let candidates = vec![1u32; 8];
    let expected = reference_planar_rewrite_schedule(&candidates, 1, 8, 3);
    assert_eq!(
        gpu_schedule(&candidates, 1, 8, 3),
        expected,
        "1x8 k=3 strided schedule must match"
    );
}
