//! GPU-IR vs CPU-ref parity for `math::sparse_recovery::iht_threshold`, the
//! Iterative Hard Thresholding step, and (via `u32_vector_scalar_map_program`) a
//! proxy for that shared vector-op-scalar builder.
//!
//! The op emits `out[i] = (|z[i]| >= threshold) ? z[i] : 0`, where `|z|` is the
//! low 31 bits (`z & 0x7FFF_FFFF`, the fixed-point magnitude). The file's oracle
//! `iht_top_k_cpu` works in f64 and computes the top-k THRESHOLD by sorting, so
//! it does not isolate this per-element compare; this test carries an INDEPENDENT
//! u32 reference of the exact contract. It also covers the shared
//! `u32_vector_scalar_map_program` builder that `spectral_shape` also uses (the
//! sibling `u32_binary_map_program` variant is covered by sinkhorn_scale). A
//! wrong magnitude mask (e.g. comparing the signed value, so a large-magnitude
//! NEGATIVE number below threshold survives), a `>` vs `>=` boundary error, or a
//! scalar-broadcast mistake all diverge here.
#![forbid(unsafe_code)]
#![cfg(all(feature = "math", feature = "cpu-parity"))]

use proptest::prelude::*;
use vyre_primitives::math::sparse_recovery::iht_threshold;
use vyre_primitives::wire::{decode_u32_le_bytes_all as unpack, pack_u32_slice as pack};
use vyre_reference::value::Value;

/// Independent u32 reference: keep `z[i]` iff its low-31-bit magnitude meets the
/// threshold, else zero.
fn oracle(z: &[u32], threshold: u32) -> Vec<u32> {
    z.iter()
        .map(|&v| if v & 0x7FFF_FFFF >= threshold { v } else { 0 })
        .collect()
}

/// Drive the real IR. Buffer binding order: z(0), threshold(1, count 1), out(2, RW).
fn gpu_threshold(z: &[u32], threshold: u32) -> Vec<u32> {
    let program = iht_threshold("z", "threshold", "out", z.len() as u32);
    let outputs = vyre_reference::reference_eval(
        &program,
        &[
            Value::from(pack(z)),
            Value::from(pack(&[threshold])),
            Value::from(pack(&vec![0u32; z.len()])),
        ],
    )
    .expect("iht_threshold reference evaluation must succeed");
    unpack(&outputs[0].to_bytes())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    #[test]
    fn ir_matches_reference_over_random_vectors(
        z in proptest::collection::vec(any::<u32>(), 1..600),
        threshold in any::<u32>(),
    ) {
        // Bias the threshold into the magnitude range so both branches fire often.
        let threshold = threshold & 0x7FFF_FFFF;
        let expected = oracle(&z, threshold);
        let got = gpu_threshold(&z, threshold);
        prop_assert_eq!(got, expected, "iht_threshold IR diverged from the u32 reference");
    }
}

/// Deterministic anchors: the `>=` boundary, the sign-bit magnitude mask (a
/// large-magnitude negative below threshold must be zeroed), threshold 0 (keep
/// all), and a length crossing the 256-lane block seam.
#[test]
fn ir_matches_reference_on_boundary_vectors() {
    // Magnitude mask: 0x8000_0005 has magnitude 5 (sign bit stripped), so with
    // threshold 10 it is BELOW and must be zeroed - even though as an unsigned
    // value it is huge. Comparing the raw value would wrongly keep it.
    let z = vec![0x8000_0005u32, 10, 9, 0x7FFF_FFFF, 0x8000_000A];
    let threshold = 10u32;
    let expected = vec![0u32, 10, 0, 0x7FFF_FFFF, 0x8000_000A];
    assert_eq!(
        oracle(&z, threshold),
        expected,
        "reference magnitude compare"
    );
    assert_eq!(
        gpu_threshold(&z, threshold),
        expected,
        "IR must compare the low-31-bit magnitude, not the raw value"
    );

    // threshold 0 -> every value has magnitude >= 0 -> keep all (identity).
    let z = vec![0u32, 1, 0x8000_0000, u32::MAX, 42];
    assert_eq!(gpu_threshold(&z, 0), z, "threshold 0 keeps everything");

    // 513 lanes crossing the 256-lane block seam; alternating above/below.
    let n = 513usize;
    let z: Vec<u32> = (0..n).map(|i| if i % 2 == 0 { 100 } else { 5 }).collect();
    let threshold = 50u32;
    assert_eq!(
        gpu_threshold(&z, threshold),
        oracle(&z, threshold),
        "block-seam threshold must match the reference"
    );
}
