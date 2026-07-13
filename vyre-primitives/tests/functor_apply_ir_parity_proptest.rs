//! GPU-IR vs CPU-ref parity for `graph::functorial::functor_apply` /
//! `functor_apply_sized`, the per-cell categorical data-migration primitive.
//!
//! Lane `t` (one per target column) scans every source column and takes the LAST
//! source whose `mapping[src] == t` (deterministic last-wins), storing it into
//! `target_row[t]`; unmapped target columns stay 0. This target-centric gather
//! avoids the write races a source-centric scatter would hit when several source
//! columns alias one target, while preserving the CPU reference's last-wins
//! contract. It is a single-round per-lane op, faithfully modelled by
//! `reference_eval`. Every shipped test drives `functor_apply_cpu` or checks
//! Program shape; the actual gather IR (the inner `mapping[src] == t` scan, the
//! last-wins assign, the target-bound gate) was never executed. A first-wins vs
//! last-wins error, a dropped unmapped-column zero, or mishandling an
//! out-of-range mapping (no target lane matches it, so it must vanish) all
//! diverge here.
#![forbid(unsafe_code)]
#![cfg(all(feature = "graph", feature = "cpu-parity"))]

use proptest::prelude::*;
use vyre_primitives::graph::functorial::{functor_apply_cpu, functor_apply_sized};
use vyre_primitives::wire::{decode_u32_le_bytes_all as unpack, pack_u32_slice as pack};
use vyre_reference::value::Value;

/// Drive the real IR. Buffer binding order: source_row(0), mapping(1),
/// target_row(2, RW). reference_eval returns target_row.
fn gpu_apply(source_row: &[u32], mapping: &[u32], target_size: u32) -> Vec<u32> {
    let program = functor_apply_sized(
        "source_row",
        "mapping",
        "target_row",
        source_row.len() as u32,
        target_size,
    );
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack(source_row)),
            Value::from(pack(mapping)),
            Value::from(pack(&vec![0u32; target_size as usize])),
        ],
    )
    .expect("functor_apply reference evaluation must succeed");
    unpack(&outputs[0].to_bytes())
}

proptest! {
    // Bounded sizes: the target-centric gather is O(n_cols * target_size) per case
    // through the interpreter (each of target_size lanes scans all n_cols sources),
    // so the random sizes stay small; the deterministic block-seam anchor covers
    // the >256-lane width.
    #![proptest_config(ProptestConfig::with_cases(800))]

    #[test]
    fn ir_matches_cpu_ref_over_random_migrations(
        n_cols in 1u32..64,
        seed in any::<u64>(),
    ) {
        let mut rng = seed;
        let mut next = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng >> 32) as u32
        };
        let target_size = 1 + next() % 64;
        let source_row: Vec<u32> = (0..n_cols).map(|_| next()).collect();
        // Mappings mostly in-range, occasionally out-of-range (must be dropped),
        // with deliberate aliasing (several sources -> one target) to exercise
        // the last-wins rule.
        let mapping: Vec<u32> = (0..n_cols)
            .map(|_| {
                if next() % 8 == 0 {
                    target_size + next() % 4 // out of range
                } else {
                    next() % target_size // in range, aliasing likely
                }
            })
            .collect();
        let expected = functor_apply_cpu(&source_row, &mapping, target_size);
        let got = gpu_apply(&source_row, &mapping, target_size);
        prop_assert_eq!(got, expected, "functor_apply IR diverged (n_cols={}, target={})", n_cols, target_size);
    }
}

/// Deterministic anchors: last-wins on aliased targets, unmapped-column zeros,
/// an out-of-range mapping dropped, and a target width crossing the 256-lane seam.
#[test]
fn ir_matches_cpu_ref_on_boundary_migrations() {
    // Aliasing: sources 0 and 2 both map to target 1 -> last (source 2's value)
    // wins. Source 1 maps to target 0. Target 2 unmapped -> 0.
    let source_row = vec![10u32, 20, 30];
    let mapping = vec![1u32, 0, 1];
    let target_size = 3u32;
    let expected = functor_apply_cpu(&source_row, &mapping, target_size);
    assert_eq!(
        expected,
        vec![20u32, 30, 0],
        "cpu_ref: last-wins on target 1, target 2 unmapped"
    );
    assert_eq!(
        gpu_apply(&source_row, &mapping, target_size),
        expected,
        "last-wins + unmapped-zero must match"
    );

    // Out-of-range mapping (target index == target_size) is dropped: no lane
    // matches it, so target stays all zero.
    let source_row = vec![99u32];
    let mapping = vec![3u32]; // target_size is 3 -> index 3 is out of range
    let dropped = functor_apply_cpu(&source_row, &mapping, 3);
    assert_eq!(
        dropped,
        vec![0u32, 0, 0],
        "cpu_ref: out-of-range mapping dropped"
    );
    assert_eq!(
        gpu_apply(&source_row, &mapping, 3),
        dropped,
        "OOB mapping must vanish in IR too"
    );

    // Target width 300 (> 256-lane workgroup) with an identity mapping over 300
    // source columns: every target lane past the block seam must gather its cell.
    let n = 300u32;
    let source_row: Vec<u32> = (0..n).map(|i| i * 3 + 1).collect();
    let mapping: Vec<u32> = (0..n).collect();
    let expected = functor_apply_cpu(&source_row, &mapping, n);
    assert_eq!(expected, source_row, "identity mapping copies the row");
    assert_eq!(
        gpu_apply(&source_row, &mapping, n),
        expected,
        "block-seam identity gather must match"
    );
}
