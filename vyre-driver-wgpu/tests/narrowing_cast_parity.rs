//! Integer-narrowing cast parity against Rust `as` on the live wgpu device.
//!
//! A narrowing cast (u32 -> u8/u16/i8/i16) is validate-LEGAL (V035 only WARNS
//! "narrowing cast may truncate high bits"), so it reaches the GPU. WGSL has no
//! 8/16-bit scalar register, so `scalar_cast_target` backs U8/U16 with a u32 and
//! I8/I16 with an i32; the bare `As` that produces that register is a no-op for a
//! same-width source. Before the narrowing fix, `300u32 as u8` therefore STAYED
//! 300 on the GPU instead of truncating to 44, a silent divergence from Rust
//! `as`, the V035 contract, and the reference oracle (the div-by-zero /
//! shift-mask silent-divergence class, Law 10).
//!
//! The scenario, the probe corpus, the Rust `as` oracle and the pinned results
//! are backend-independent and live in `vyre_test_support::cast_parity`, which
//! the CUDA twin of this gate reads too. Only the dispatch below is wgpu.

#![cfg(feature = "device-tests")]

use vyre_driver::parity_harness::{
    dispatch_single_output, elementwise_program, u32_words, ParityInput,
};
use vyre_driver::{DispatchConfig, VyreBackend};
use vyre_driver_wgpu::WgpuBackend;
use vyre_foundation::ir::{DataType, Expr};
use vyre_test_support::cast_parity::{NarrowingCase, NARROWING_CASES, NARROWING_INPUTS};

/// Dispatch `out = cast(wide, cast(narrow, input))` for every probe input.
///
/// `wide` is the non-narrowing integer that round-trips the narrowed value into
/// a 32-bit store slot, so the word read back reflects exactly what the
/// narrowing cast produced rather than what a byte-element store would have
/// masked it to.
fn run(backend: &WgpuBackend, narrow: DataType, wide: DataType) -> Vec<u32> {
    let inputs = NARROWING_INPUTS;
    let buffers = vec![ParityInput::u32_words("input", &inputs)];
    let program = elementwise_program(wide.clone(), &buffers, inputs.len() as u32, &|loads| {
        Expr::cast(wide.clone(), Expr::cast(narrow.clone(), loads[0].clone()))
    });
    let raw = dispatch_single_output(
        &|prog, in_bufs| backend.dispatch_borrowed(prog, in_bufs, &DispatchConfig::default()),
        &program,
        &buffers,
        inputs.len() * 4,
        "wgpu narrowing-cast parity",
    );
    u32_words(&raw)
}

/// Every narrowing cast the matrix declares must truncate and re-extend on the
/// live device exactly as Rust `as` does on the host.
///
/// Walking the shared matrix rather than repeating a body per cast means a case
/// added there is dispatched here without touching this file, so a new narrowing
/// target cannot land proven on one backend only.
#[test]
fn every_narrowing_cast_matches_rust_as_on_gpu() {
    let backend =
        WgpuBackend::acquire().expect("Fix: narrowing-cast parity requires a live GPU backend.");
    assert!(
        !NARROWING_CASES.is_empty(),
        "Fix: the narrowing-cast matrix is empty, so this gate proves nothing on wgpu"
    );
    for case in NARROWING_CASES {
        let NarrowingCase { narrow, wide, .. } = case;
        let device = run(&backend, narrow.clone(), wide.clone());
        case.assert_target_words("wgpu", &device);
    }
}
