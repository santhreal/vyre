//! CUDA parity for the byte span an async transfer names, on the live GPU.
//!
//! An async transfer carries its offset and length in bytes. The PTX emitter used
//! to shift the byte offset down to a word index, which drops the low two bits:
//! every offset that was not a multiple of four copied from the wrong byte, and a
//! store shortened the copy to the source length instead of padding with zeros.
//! Native `cp.async` moves four naturally-aligned bytes per instruction and
//! cannot express a byte-granular span, so it is now reserved for a statically
//! word-aligned offset and the assembled copy loop carries the rest.
//!
//! The matrix, the fixture buffers and the reference arm are
//! `vyre_test_support::async_span_parity`, shared with the wgpu twin so both
//! backends answer the same question. What is not shared is this file: the PTX
//! word assembly and the live dispatch.

#![cfg(feature = "device-tests")]

mod harness;

use harness::live_backend;
use vyre_driver::DispatchConfig;
use vyre_driver_cuda::CudaBackend;
use vyre_foundation::ir::Program;
use vyre_test_support::async_span_parity::{assert_matrix_covers_every_alignment, cases, SpanCase};

/// What the CUDA arm lowered the transfer to, for the failure message.
const LOWERING: &str = "the PTX word-assembly copy loop";

fn dispatch(backend: &CudaBackend, program: &Program, inputs: &[Vec<u8>]) -> Vec<Vec<u8>> {
    let lowered = vyre_foundation::optimizer::optimize(program.clone())
        .expect("registered optimizer must converge");
    backend
        .dispatch(&lowered, inputs, &DispatchConfig::default())
        .expect("Fix: an async transfer program must dispatch on the live CUDA device")
}

fn check(backend: &CudaBackend, case: &SpanCase) {
    let program = case.program();
    let gpu = dispatch(backend, &program, &case.inputs());
    case.assert_matches_reference("CUDA", LOWERING, &gpu);
}

/// Every span in the shared matrix copies the bytes it names, in both
/// directions, whether or not the emitter can see the offset's alignment.
#[test]
fn every_async_transfer_span_matches_the_reference_on_cuda() {
    assert_matrix_covers_every_alignment();
    let backend = live_backend();
    for case in cases() {
        check(&backend, &case);
    }
}
