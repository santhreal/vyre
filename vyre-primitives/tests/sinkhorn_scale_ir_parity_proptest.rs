//! GPU-IR vs CPU-ref parity for `math::sinkhorn_scale`, the Sinkhorn-Knopp
//! scaling-step combiner and (via the shared `u32_binary_map_program` builder)
//! a proxy for that builder's correctness.
//!
//! The op emits `out[i] = target[i] / max(divisor[i], FLOOR)` in u32, with the
//! divide-by-zero guard lowered as `select(divisor == 0, 1, divisor)`. The file's
//! only oracle is `sinkhorn_iter_cpu`, a full f64 Sinkhorn iteration, which does
//! NOT isolate this integer scaling kernel; so this test carries an INDEPENDENT
//! u32 reference (a second implementation of the exact contract, not a copy of
//! the IR). It also exercises the shared `u32_binary_map_program` grid-strided
//! per-lane map that `sparse_recovery`, `spectral_shape`, and `dp_accountant`
//! also build on. A missing zero-guard (u32 `x / 0` traps / is UB), a swapped
//! target/divisor operand, or a dropped `i < count` bound all diverge here.
#![forbid(unsafe_code)]
#![cfg(all(feature = "math", feature = "cpu-parity"))]

use proptest::prelude::*;
use vyre_primitives::math::sinkhorn::{sinkhorn_scale, DIVISOR_FLOOR};
use vyre_primitives::wire::{decode_u32_le_bytes_all as unpack, pack_u32_slice as pack};
use vyre_reference::value::Value;

/// Independent u32 reference for one scaling step: `out[i] = target[i] /
/// max(divisor[i], FLOOR)`, guarding only literal zero as the IR does.
fn oracle(target: &[u32], divisor: &[u32]) -> Vec<u32> {
    target
        .iter()
        .zip(divisor)
        .map(|(&t, &d)| {
            let d_safe = if d == 0 { DIVISOR_FLOOR } else { d };
            t / d_safe
        })
        .collect()
}

/// Drive the real IR through `reference_eval`. Buffer binding order: target(0),
/// divisor(1), out(2, the only ReadWrite buffer).
fn gpu_scale(target: &[u32], divisor: &[u32]) -> Vec<u32> {
    let count = target.len() as u32;
    let program = sinkhorn_scale("target", "divisor", "out", count);
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack(target)),
            Value::from(pack(divisor)),
            Value::from(pack(&vec![0u32; target.len()])),
        ],
    )
    .expect("sinkhorn_scale reference evaluation must succeed");
    unpack(&outputs[0].to_bytes())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn ir_matches_reference_over_random_vectors(
        // Length straddles the 256-lane workgroup boundary into multiple blocks.
        pairs in proptest::collection::vec((any::<u32>(), any::<u32>()), 1..600)
    ) {
        let target: Vec<u32> = pairs.iter().map(|&(t, _)| t).collect();
        let divisor: Vec<u32> = pairs.iter().map(|&(_, d)| d).collect();
        let expected = oracle(&target, &divisor);
        let got = gpu_scale(&target, &divisor);
        prop_assert_eq!(got, expected, "sinkhorn_scale IR diverged from the u32 reference");
    }
}

/// Deterministic anchors: the zero-divisor floor, divisor==1 identity, exact and
/// truncating integer division, and a length crossing the 256-lane block seam.
#[test]
fn ir_matches_reference_on_boundary_vectors() {
    // Zero divisor -> floor to 1 -> out == target (the divide-by-zero guard).
    let target = vec![7u32, 0, u32::MAX, 65_536];
    let zeros = vec![0u32; 4];
    assert_eq!(
        oracle(&target, &zeros),
        target,
        "reference: zero divisor floors to identity"
    );
    assert_eq!(
        gpu_scale(&target, &zeros),
        target,
        "IR must floor a zero divisor to 1 (no u32 divide-by-zero)"
    );

    // Mixed exact + truncating division.
    let t = vec![100u32, 100, 100, 1, 0];
    let d = vec![1u32, 3, 100, 2, 5];
    let expected = vec![100u32, 33, 1, 0, 0];
    assert_eq!(
        oracle(&t, &d),
        expected,
        "reference: integer division truncates"
    );
    assert_eq!(
        gpu_scale(&t, &d),
        expected,
        "IR integer division must match"
    );

    // 513 lanes: crosses the 256-lane workgroup seam twice; lane i divides
    // (i*2) by (i%7 + 1) so every block boundary carries a distinct quotient.
    let n = 513usize;
    let target: Vec<u32> = (0..n).map(|i| (i as u32) * 2).collect();
    let divisor: Vec<u32> = (0..n).map(|i| (i as u32 % 7) + 1).collect();
    assert_eq!(
        gpu_scale(&target, &divisor),
        oracle(&target, &divisor),
        "block-seam-spanning divide must match the reference"
    );
}
