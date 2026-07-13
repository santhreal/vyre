//! Differential + ABI-behavioral parity for `c11_compute_alignments_for_abi`.
//!
//! `c11_compute_alignments` (the tested + registered base) is exactly
//! `c11_compute_alignments_for_abi(.., 8, 8, 8)` (LP64). The explicit-ABI builder was
//! an orphan (registry-coverage closure gate `adversarial_registry_closure.rs`). This
//! pins it two ways with real output bytes (Testing-Contract, never `!is_empty`):
//!   1. LP64 `for_abi(8,8,8)` must be byte-identical to the default base, and
//!   2. ILP32 `for_abi(4,4,4)` must actually shrink pointer + long SIZES and the
//!      double ALIGNMENT, proving the ABI parameters are wired into the emitted
//!      size/alignment evidence rather than silently ignored (the module doc claims
//!      `-m64`->LP64 / `-m32`->ILP32; this is the executable proof of that claim).
#![cfg(feature = "c-parser")]
mod common;

use common::decode_u32_words as words;
use common::u32_bytes as bytes;
use vyre::ir::Expr;
use vyre_libs::compiler::types_layout::{
    c11_compute_alignments, c11_compute_alignments_for_abi, C_ABI_CHAR, C_ABI_DOUBLE, C_ABI_LONG,
    C_ABI_POINTER,
};
use vyre_reference::value::Value;

/// CHAR, POINTER, LONG, DOUBLE, and an unknown(0) tag (one of every ABI size class).
const TYPES: [u32; 5] = [C_ABI_CHAR, C_ABI_POINTER, C_ABI_LONG, C_ABI_DOUBLE, 0];

fn eval_sizes_aligns(program: &vyre::ir::Program, types: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let n = types.len();
    let outputs = vyre_reference::reference_eval(
        program,
        &[
            Value::from(bytes(types)),
            Value::from(vec![0u8; n * 4]), // out_sizes (binding 1)
            Value::from(vec![0u8; n * 4]), // out_alignments (binding 2)
        ],
    )
    .expect("compute_alignments program must execute under reference_eval");
    (words(&outputs[0].to_bytes()), words(&outputs[1].to_bytes()))
}

#[test]
fn lp64_for_abi_matches_default_base() {
    let n = TYPES.len() as u32;
    let base = c11_compute_alignments("types", "sizes", "aligns", Expr::u32(n));
    let for_abi = c11_compute_alignments_for_abi("types", "sizes", "aligns", Expr::u32(n), 8, 8, 8);

    let (base_sizes, base_aligns) = eval_sizes_aligns(&base, &TYPES);
    let (abi_sizes, abi_aligns) = eval_sizes_aligns(&for_abi, &TYPES);

    // char=1, pointer=long=8, double=8, unknown->4.
    assert_eq!(base_sizes, vec![1, 8, 8, 8, 4], "LP64 sizes");
    // char=1, pointer=long=8, double aligns to 8, unknown->4.
    assert_eq!(base_aligns, vec![1, 8, 8, 8, 4], "LP64 alignments");
    assert_eq!(
        abi_sizes, base_sizes,
        "for_abi(8,8,8) sizes must equal the LP64 default base"
    );
    assert_eq!(
        abi_aligns, base_aligns,
        "for_abi(8,8,8) alignments must equal the LP64 default base"
    );
}

#[test]
fn ilp32_for_abi_shrinks_pointer_long_and_double_align() {
    let n = TYPES.len() as u32;
    let ilp32 = c11_compute_alignments_for_abi("types", "sizes", "aligns", Expr::u32(n), 4, 4, 4);
    let (sizes, aligns) = eval_sizes_aligns(&ilp32, &TYPES);

    // pointer + long shrink to 4; double SIZE stays 8 (only its alignment is ABI-tunable); unknown->4.
    assert_eq!(
        sizes,
        vec![1, 4, 4, 8, 4],
        "ILP32 pointer+long must shrink to 4; double size stays 8"
    );
    // double ALIGNMENT shrinks to 4 under -m32; everything else follows its size.
    assert_eq!(
        aligns,
        vec![1, 4, 4, 4, 4],
        "ILP32 double alignment must shrink to 4"
    );
}
