//! Integer-narrowing cast (`u32` -> `u8`/`u16`/`i8`/`i16`) parity against Rust
//! `as` on the live CUDA device.
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
//! The scenario, the probe corpus, the Rust `as` oracle and the pinned results
//! are backend-independent and live in `vyre_test_support::cast_parity`, which
//! the wgpu twin of this gate reads too. Only the dispatch below is CUDA.

#![cfg(feature = "device-tests")]

mod harness;
use harness::live_backend;

use vyre_driver::parity_harness::{
    dispatch_single_output, elementwise_program, u32_words, ParityInput,
};
use vyre_driver::DispatchConfig;
use vyre_driver_cuda::CudaBackend;
use vyre_foundation::ir::{DataType, Expr};
use vyre_test_support::cast_parity::{NarrowingCase, NARROWING_CASES, NARROWING_INPUTS};

/// Dispatch `out = cast(wide, cast(narrow, input))` for every probe input.
///
/// `wide` is the non-narrowing integer that round-trips the narrowed value into
/// a 32-bit store slot, so the word read back reflects exactly what the
/// narrowing cast produced rather than what a byte-element store would have
/// masked it to.
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

/// Every narrowing cast the matrix declares must truncate and re-extend on the
/// live device exactly as Rust `as` does on the host.
///
/// Walking the shared matrix rather than repeating a body per cast means a case
/// added there is dispatched here without touching this file, so a new narrowing
/// target cannot land proven on one backend only.
#[test]
fn every_narrowing_cast_matches_rust_as_on_cuda() {
    let backend = live_backend();
    assert!(
        !NARROWING_CASES.is_empty(),
        "Fix: the narrowing-cast matrix is empty, so this gate proves nothing on CUDA"
    );
    for case in NARROWING_CASES {
        let NarrowingCase { narrow, wide, .. } = case;
        let device = run(&backend, narrow.clone(), wide.clone());
        case.assert_target_words("CUDA", &device);
    }
}
