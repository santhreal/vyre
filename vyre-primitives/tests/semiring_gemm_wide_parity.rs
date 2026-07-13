//! GPU-IR parity for `math::scallop_join_wide::semiring_gemm_wide`.
//!
//! A W-word boolean-semiring GEMM for Datalog fixpoint lineage (had no parity
//! test (the final genuine orphan from the registry-coverage closure gate)).
//! Output cell `t = i*n + j` holds W u32 words:
//!   C[i,j][word] = seed[i,j][word]
//!     OR  ( OR over kk of:  (A[i,kk] != 0-cell AND B[kk,j] != 0-cell)
//!                             ? (A[i,kk][word] | B[kk,j][word]) : 0 )
//! The "product" gates on whether BOTH operand CELLS are entirely zero (across
//! all W words); when both are nonzero it unions their bit payloads, and the
//! "sum" over kk is OR. Row-major flat layout: A is m*k cells, B is k*n cells,
//! C/seed are m*n cells, each cell W consecutive words. Pins this against a
//! hand-computed reference via `reference_eval` (Testing-Contract: real values).
#![cfg(feature = "math")]

use vyre_primitives::math::scallop_join_wide::semiring_gemm_wide;
use vyre_primitives::wire::{decode_u32_le_bytes_all as unpack, pack_u32_slice as pack};
use vyre_reference::value::Value;

fn eval(a: &[u32], b: &[u32], seed: Option<&[u32]>, m: u32, n: u32, k: u32, w: u32) -> Vec<u32> {
    let cells = (m * n) as usize;
    let program = semiring_gemm_wide("a", "b", "c", seed.map(|_| "seed"), m, n, k, w);
    let mut inputs = vec![
        Value::from(pack(a)),
        Value::from(pack(b)),
        Value::from(pack(&vec![0u32; cells * w as usize])),
    ];
    if let Some(seed) = seed {
        inputs.push(Value::from(pack(seed)));
    }
    let outputs = vyre_reference::reference_eval(&program, &inputs)
        .expect("semiring_gemm_wide reference evaluation must succeed");
    unpack(&outputs[0].to_bytes()) // c is the sole ReadWrite buffer
}

#[test]
fn w1_boolean_semiring_matmul_matches_hand_reference() {
    // m=n=k=2, w=1. A/B row-major, one word per cell.
    let a = [0b0001u32, 0b0000, 0b0010, 0b0100]; // A[0,0]=1,A[0,1]=0,A[1,0]=2,A[1,1]=4
    let b = [0b1000u32, 0b0000, 0b0001, 0b0010]; // B[0,0]=8,B[0,1]=0,B[1,0]=1,B[1,1]=2
                                                 // C00: kk0 (A=1,B=8 both nz) 1|8=0b1001; kk1 (A[0,1]=0) 0 -> 0b1001
                                                 // C01: kk0 (B[0,1]=0) 0; kk1 (A[0,1]=0) 0 -> 0
                                                 // C10: kk0 (A=2,B=8) 2|8=0b1010; kk1 (A=4,B[1,0]=1) 4|1=0b0101 -> OR = 0b1111
                                                 // C11: kk0 (B[0,1]=0) 0; kk1 (A=4,B=2) 4|2=0b0110 -> 0b0110
    let expected = vec![0b1001, 0, 0b1111, 0b0110];
    assert_eq!(eval(&a, &b, None, 2, 2, 2, 1), expected);
}

#[test]
fn w1_seed_is_unioned_into_output() {
    let a = [0b0001u32, 0b0000, 0b0010, 0b0100];
    let b = [0b1000u32, 0b0000, 0b0001, 0b0010];
    let seed = [0b1_0000u32, 0b0000, 0b0000, 0b10_0000];
    // Same as the no-seed case, then OR the seed per cell.
    let expected = vec![0b1001 | 0b1_0000, 0, 0b1111, 0b0110 | 0b10_0000];
    assert_eq!(eval(&a, &b, Some(&seed), 2, 2, 2, 1), expected);
}

#[test]
fn w2_multi_word_cell_indexing_matches_hand_reference() {
    // m=n=k=1, w=2: a single cell of two words. Both operand cells nonzero, so
    // each word is a|b. Locks the (cell*w + word) flat indexing across words.
    let a = [0b01u32, 0b10]; // A[0,0] = {word0:0b01, word1:0b10}
    let b = [0b100u32, 0b1000]; // B[0,0] = {word0:0b100, word1:0b1000}
    let expected = vec![0b01 | 0b100, 0b10 | 0b1000]; // [0b101, 0b1010]
    assert_eq!(eval(&a, &b, None, 1, 1, 1, 2), expected);
}

#[test]
fn w1_zero_operand_cell_contributes_nothing() {
    // If A's cell is all-zero, the product is 0 regardless of B (and vice versa),
    // so C is exactly the seed.
    let a = [0u32, 0, 0, 0];
    let b = [0b1111u32, 0b1111, 0b1111, 0b1111];
    let seed = [0b1u32, 0b10, 0b100, 0b1000];
    assert_eq!(
        eval(&a, &b, Some(&seed), 2, 2, 2, 1),
        vec![0b1, 0b10, 0b100, 0b1000],
        "all-zero A cells yield only the seed"
    );
}
