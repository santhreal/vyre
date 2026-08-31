//! Generated CPU-reference matrix for public f32 hardware intrinsic builders.
//!
//! The f32 Cat-C surface has precision-sensitive contracts: FMA must use
//! single-round `mul_add`, and inverse sqrt must clamp hostile inputs before
//! lowering. This matrix covers edge values and generated lanes beyond one
//! workgroup.

mod gate_fixtures;

use gate_fixtures::{generated_f32_with_edges, inverse_sqrt_f32_ref, run_eval_single};
use vyre_primitives::wire::pack_f32_slice as pack;

const FINITE_EDGES: [f32; 12] = [
    -8.0,
    -1.0,
    -0.0,
    0.0,
    f32::MIN_POSITIVE,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    16.0,
    f32::MAX,
];

fn generated_finite(len: usize, seed: u32) -> Vec<f32> {
    generated_f32_with_edges(len, seed, &FINITE_EDGES)
}

fn generated_inverse_sqrt_inputs(len: usize, seed: u32) -> Vec<f32> {
    let hostile = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -1.0,
        -0.0,
        0.0,
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        0.25,
        1.0,
        4.0,
        16.0,
    ];
    let mut values = generated_finite(len, seed)
        .into_iter()
        .map(|value| value.abs() + 0.01)
        .collect::<Vec<_>>();
    for (idx, value) in hostile.into_iter().enumerate().take(values.len()) {
        values[idx] = value;
    }
    values
}

#[test]
fn generated_fma_f32_matrix_matches_mul_add_bits() {
    let lengths = [1usize, 2, 3, 4, 31, 32, 63, 64, 65, 257, 1024, 2048];
    let mut checked_lanes = 0usize;

    for &len in &lengths {
        let a = generated_finite(len, 0x0f1a_a011 ^ len as u32);
        let b = generated_finite(len, 0x0f1a_a012 ^ len as u32);
        let c = generated_finite(len, 0x0f1a_a013 ^ len as u32);
        let program = vyre_primitives::hardware::fma_f32::fma_f32("a", "b", "c", "out", len as u32);
        let got = run_eval_single(&program, vec![pack(&a), pack(&b), pack(&c)]);
        let expected = a
            .iter()
            .zip(b.iter())
            .zip(c.iter())
            .map(|((&x, &y), &z)| x.mul_add(y, z))
            .collect::<Vec<_>>();
        assert_eq!(got, pack(&expected), "fma_f32 failed for len {len}");
        checked_lanes += len;
    }

    assert_eq!(checked_lanes, lengths.iter().sum::<usize>());
}

#[test]
fn generated_inverse_sqrt_f32_matrix_matches_clamped_host_semantics() {
    let lengths = [1usize, 2, 3, 4, 31, 32, 63, 64, 65, 257, 1024, 2048];
    let mut checked_lanes = 0usize;

    for &len in &lengths {
        let input = generated_inverse_sqrt_inputs(len, 0x0f1a_b005 ^ len as u32);
        let program = vyre_primitives::hardware::inverse_sqrt_f32::inverse_sqrt_f32(
            "input", "out", len as u32,
        );
        let got = run_eval_single(&program, vec![pack(&input)]);
        let expected = input
            .iter()
            .copied()
            .map(inverse_sqrt_f32_ref)
            .collect::<Vec<_>>();
        assert_eq!(
            got,
            pack(&expected),
            "inverse_sqrt_f32 failed for len {len}"
        );
        checked_lanes += len;
    }

    assert_eq!(checked_lanes, lengths.iter().sum::<usize>());
}
