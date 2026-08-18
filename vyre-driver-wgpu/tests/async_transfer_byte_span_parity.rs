//! Parity for the byte span an async transfer names, on the live GPU.
//!
//! An async transfer carries its offset and length in bytes. The naga emitter
//! used to turn the byte offset into a word index by dividing it by four, which
//! drops the low two bits: every offset that was not a multiple of four copied
//! from the wrong byte, and a store shortened the copy to the source length
//! instead of padding with zeros. What the emitter lowers a transfer to now is a
//! loop that assembles each word from the two source words the span straddles
//! and merges a partial end word under a byte mask.
//!
//! The matrix, the fixture buffers and the reference arm are
//! `vyre_test_support::async_span_parity`, shared with the CUDA twin so both
//! backends answer the same question. What is not shared is this file: the naga
//! word assembly and the live dispatch.

use vyre_driver::{DispatchConfig, VyreBackend};
use vyre_driver_wgpu::WgpuBackend;
use vyre_foundation::ir::Program;
use vyre_test_support::async_span_parity::{assert_matrix_covers_every_alignment, cases, SpanCase};

/// What the wgpu arm lowered the transfer to, for the failure message.
const LOWERING: &str = "the naga word-assembly copy loop";

fn dispatch(backend: &WgpuBackend, program: &Program, inputs: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let lowered = vyre_foundation::optimizer::optimize(program.clone())
        .expect("registered optimizer must converge");
    backend
        .dispatch(&lowered, inputs, &DispatchConfig::default())
        .expect("Fix: an async transfer program must dispatch on the live GPU")
}

fn check(backend: &WgpuBackend, case: &SpanCase) {
    let program = case.program();
    let gpu = dispatch(backend, &program, &case.inputs());
    case.assert_matches_reference("GPU", LOWERING, &gpu);
}

/// Every span in the shared matrix copies the bytes it names, in both
/// directions, whether or not the emitter can see the offset's alignment.
///
/// One test over the matrix rather than one per direction: the coverage
/// assertion runs first, so a matrix trimmed back to the aligned offsets fails
/// before the GPU is acquired.
#[test]
fn every_async_transfer_span_matches_the_reference_on_gpu() {
    assert_matrix_covers_every_alignment();
    let backend = WgpuBackend::acquire().expect("Fix: async span parity needs a live GPU.");
    for case in cases() {
        check(&backend, &case);
    }
}
