//! 64-bit widening-cast (`i32`/`u32` -> `i64`/`u64`) parity against Rust `as` on
//! the live GPU.
//!
//! WGSL has no native 64-bit integer; `U64`/`I64` are backed by `vec2<u32>` (low
//! word `.x`, high word `.y`). A widening cast must fill the HIGH word per the
//! SOURCE's signedness: a signed (`i32`) source SIGN-extends (high =
//! `0xFFFF_FFFF` when the value is negative, matching `i32 as i64`), an unsigned
//! (`u32`) source ZERO-extends. The emitter synthesizes the signed high word
//! componentwise as `(low >> 31) * 0xFFFF_FFFF`: a logical `ShiftRight` on a
//! `u32` then a multiply. That is exactly the class of computed-value lowering
//! the naga signed-`Modulo` bug proved can be silently wrong on real hardware
//! (a source-read alone is NOT proof): if naga ever lowered that `ShiftRight` as
//! arithmetic, or the multiply mis-typed, every negative `i32 -> i64/u64` would
//! read back as a large positive value (the Law 10 miscompile the high-word fix
//! closed).
//!
//! These tests dispatch the cast on the 5090, read back BOTH 32-bit words of the
//! resulting 64-bit value, and assert the full `u64` byte-for-byte against Rust.
//! The inventory carried 64-bit sign-extension as unit/source-proven with a
//! live-GPU check as the "remaining gold standard" (this is that check).

mod harness;
use harness::u32_bytes;

use vyre_driver::{DispatchConfig, VyreBackend};
use vyre_driver_wgpu::WgpuBackend;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_test_support::cast_parity::{
    I32_TO_I64_EXPECTED, SIGNED_WIDENING_INPUTS, UNSIGNED_WIDENING_INPUTS,
};

/// `out[i] = cast(target64, load(src, i))`: `src` is a 32-bit buffer (signed or
/// unsigned per `src_ty`), `out` is the 64-bit `target64` buffer (`array<vec2<u32>>`).
fn widen_program(src_ty: DataType, target64: DataType, n: u32) -> Program {
    let mut body = Vec::new();
    for i in 0..n {
        body.push(Node::store(
            "out",
            Expr::u32(i),
            Expr::cast(target64.clone(), Expr::load("src", Expr::u32(i))),
        ));
    }
    Program::wrapped(
        vec![
            BufferDecl::storage("out", 0, BufferAccess::ReadWrite, target64).with_count(n),
            BufferDecl::storage("src", 1, BufferAccess::ReadOnly, src_ty).with_count(n),
        ],
        [1, 1, 1],
        body,
    )
}

/// Dispatch and reconstruct each 64-bit element from its two little-endian words.
fn run(backend: &WgpuBackend, src_ty: DataType, target64: DataType, words: &[u32]) -> Vec<u64> {
    let n = words.len() as u32;
    let program = widen_program(src_ty, target64, n);
    let src_bytes = u32_bytes(words);
    // 8 bytes per 64-bit output element => 2 zero u32 words each.
    let out_init = u32_bytes(&vec![0u32; words.len() * 2]);
    let outputs = backend
        .dispatch_borrowed(
            &program,
            &[out_init.as_slice(), src_bytes.as_slice()],
            &DispatchConfig::default(),
        )
        .expect("Fix: WGPU must dispatch the 64-bit widening-cast contract.");
    outputs[0]
        .chunks_exact(8)
        .map(|c| {
            let low = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            let high = u32::from_le_bytes([c[4], c[5], c[6], c[7]]);
            (u64::from(high) << 32) | u64::from(low)
        })
        .collect()
}
fn assert_widening_contract(
    backend: &WgpuBackend,
    src_ty: DataType,
    target64: DataType,
    words: &[u32],
    expected: &[u64],
    label: &str,
) {
    let gpu = run(backend, src_ty, target64, words);
    assert_eq!(
        gpu, expected,
        "GPU {label} widening diverged from Rust reference.\n  expected: {expected:#018x?}\n  gpu:      {gpu:#018x?}"
    );
}

#[test]
fn i32_to_i64_sign_extends_high_word_on_gpu() {
    let backend =
        WgpuBackend::acquire().expect("Fix: 64-bit widening parity requires a live GPU backend.");
    let ins = SIGNED_WIDENING_INPUTS;
    let words: Vec<u32> = ins.iter().map(|&v| v as u32).collect();
    let expected: Vec<u64> = ins.iter().map(|&v| (v as i64) as u64).collect();
    assert_eq!(
        expected, I32_TO_I64_EXPECTED,
        "reference i32->i64 sign-extension drifted"
    );
    assert_widening_contract(
        &backend,
        DataType::I32,
        DataType::I64,
        &words,
        &expected,
        "i32->i64",
    );
}

#[test]
fn i32_to_u64_sign_extends_high_word_on_gpu() {
    let backend =
        WgpuBackend::acquire().expect("Fix: 64-bit widening parity requires a live GPU backend.");
    let ins = SIGNED_WIDENING_INPUTS;
    let words: Vec<u32> = ins.iter().map(|&v| v as u32).collect();
    let expected: Vec<u64> = ins.iter().map(|&v| v as u64).collect();
    assert_eq!(
        expected[0], 0xFFFF_FFFF_FFFF_FFF9,
        "reference i32->u64 sign-extension drifted (-7 must carry the high word)"
    );
    assert_widening_contract(
        &backend,
        DataType::I32,
        DataType::U64,
        &words,
        &expected,
        "i32->u64",
    );
}

#[test]
fn u32_to_u64_zero_extends_high_word_on_gpu() {
    let backend =
        WgpuBackend::acquire().expect("Fix: 64-bit widening parity requires a live GPU backend.");
    let words = UNSIGNED_WIDENING_INPUTS;
    let expected: Vec<u64> = words.iter().map(|&w| u64::from(w)).collect();
    assert_eq!(
        expected[0], 0x0000_0000_FFFF_FFFF,
        "reference u32->u64 zero-extension drifted"
    );
    assert_widening_contract(
        &backend,
        DataType::U32,
        DataType::U64,
        &words,
        &expected,
        "u32->u64",
    );
}
