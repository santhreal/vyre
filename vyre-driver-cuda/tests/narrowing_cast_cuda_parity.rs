//! Integer-narrowing cast (`u32` -> `u8`/`u16`/`i8`/`i16`) parity against Rust
//! `as` / the reference oracle on the live CUDA device, the PTX/CUDA twin of the
//! wgpu `narrowing_cast_parity` gate.
//!
//! A narrowing cast is validate-LEGAL (V035 only WARNS), so it reaches the GPU.
//! The PTX backend's `from_dtype` COLLAPSES U8/U16->U32 and I8/I16->I32, so a bare
//! convert is a no-op for a same-width source; the narrowing fix made `emit_cast`
//! emit the canonical `cvt.u32.u8` (zero-extend) / `cvt.s32.s8` (sign-extend)
//! BEFORE the identity early-return. That PTX path was unit-asserted but NEVER
//! dispatched on a live CUDA device, the same source-read-vs-hardware gap the
//! naga signed-`Modulo` miscompile punished. If the `cvt` were skipped, `300u32 as
//! u8` would read back 300 instead of 44 (a silent non-narrowing divergence).
//!
//! To isolate the CAST's narrowing from the byte-element STORE (which masks to a
//! byte regardless), the narrowed value is widened back out: `cast(WIDE,
//! cast(NARROW, x))`, and stored into a 32-bit buffer, so the word read back
//! reflects exactly what the narrowing cast produced.

mod common;
use common::live_backend;

use vyre_driver::parity_harness::{
    dispatch_single_output, elementwise_program, u32_words, ParityInput,
};
use vyre_driver::DispatchConfig;
use vyre_driver_cuda::CudaBackend;
use vyre_foundation::ir::{DataType, Expr};
use vyre_test_support::cast_parity::{
    NARROWING_INPUTS, U32_TO_I16_EXPECTED, U32_TO_I8_EXPECTED, U32_TO_U16_EXPECTED,
    U32_TO_U8_EXPECTED,
};

/// Dispatch `out = cast(wide, cast(narrow, input))` for every probe input.
///
/// `wide` is the non-narrowing integer that round-trips the narrowed value into
/// a 32-bit store slot (U32 for U8/U16, I32 for I8/I16), so the word read back
/// reflects exactly what the narrowing cast produced rather than what a
/// byte-element store would have masked it to.
fn run(backend: &CudaBackend, narrow: DataType, wide: DataType) -> Vec<u32> {
    let ins = NARROWING_INPUTS;
    let buffers = vec![ParityInput::u32_words("input", &ins)];
    let count = ins.len() as u32;
    let program = elementwise_program(wide.clone(), &buffers, count, &|loads| {
        Expr::cast(wide.clone(), Expr::cast(narrow.clone(), loads[0].clone()))
    });
    let bytes = dispatch_single_output(
        &|program, inputs| backend.dispatch_borrowed(program, inputs, &DispatchConfig::default()),
        &program,
        &buffers,
        ins.len() * 4,
        "narrowing-cast parity",
    );
    u32_words(&bytes)
}

#[test]
fn u32_to_u8_narrowing_truncates_low_byte_on_cuda() {
    let backend = live_backend();
    let gpu = run(&backend, DataType::U8, DataType::U32);
    let expected: Vec<u32> = NARROWING_INPUTS
        .iter()
        .map(|&v| u32::from(v as u8))
        .collect();
    assert_eq!(
        expected, U32_TO_U8_EXPECTED,
        "reference u32->u8 truncation drifted"
    );
    assert_eq!(
        gpu, expected,
        "CUDA u32->u8 narrowing diverged from `as u8` (cvt.u32.u8 skipped?).\n  inputs:   {:?}\n  expected: {:?}\n  gpu:      {:?}",
        NARROWING_INPUTS, expected, gpu
    );
}

#[test]
fn u32_to_u16_narrowing_truncates_low_half_on_cuda() {
    let backend = live_backend();
    let gpu = run(&backend, DataType::U16, DataType::U32);
    let expected: Vec<u32> = NARROWING_INPUTS
        .iter()
        .map(|&v| u32::from(v as u16))
        .collect();
    assert_eq!(
        expected, U32_TO_U16_EXPECTED,
        "reference u32->u16 truncation drifted"
    );
    assert_eq!(
        gpu, expected,
        "CUDA u32->u16 narrowing diverged from `as u16`.\n  expected: {expected:?}\n  gpu: {gpu:?}"
    );
}

#[test]
fn u32_to_i8_narrowing_sign_extends_on_cuda() {
    let backend = live_backend();
    let gpu: Vec<i32> = run(&backend, DataType::I8, DataType::I32)
        .into_iter()
        .map(|w| w as i32)
        .collect();
    let expected: Vec<i32> = NARROWING_INPUTS
        .iter()
        .map(|&v| i32::from(v as u8 as i8))
        .collect();
    assert_eq!(
        expected, U32_TO_I8_EXPECTED,
        "reference u32->i8 sign-extension drifted"
    );
    assert_eq!(
        gpu, expected,
        "CUDA u32->i8 narrowing diverged from `as i8` (cvt.s32.s8 sign-extend?).\n  expected: {expected:?}\n  gpu: {gpu:?}"
    );
}

#[test]
fn u32_to_i16_narrowing_sign_extends_on_cuda() {
    let backend = live_backend();
    let gpu: Vec<i32> = run(&backend, DataType::I16, DataType::I32)
        .into_iter()
        .map(|w| w as i32)
        .collect();
    let expected: Vec<i32> = NARROWING_INPUTS
        .iter()
        .map(|&v| i32::from(v as u16 as i16))
        .collect();
    assert_eq!(
        expected, U32_TO_I16_EXPECTED,
        "reference u32->i16 sign-extension drifted"
    );
    assert_eq!(
        gpu, expected,
        "CUDA u32->i16 narrowing diverged from `as i16`.\n  expected: {expected:?}\n  gpu: {gpu:?}"
    );
}
