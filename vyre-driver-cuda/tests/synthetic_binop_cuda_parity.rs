//! Parity for the u32 mulhi / abs_diff / saturating / rotate binops against
//! Rust/oracle on the live CUDA device, the PTX/CUDA twin of the wgpu
//! `synthetic_binop_parity` gate.
//!
//! These ops are the class the naga signed-`Modulo` miscompile proved can be
//! silently wrong on real silicon, and the PTX backend lowers them DIFFERENTLY
//! from naga's multi-step `vec2`/`select` synthesis (PTX has native `mul.hi.u32`,
//! `vabsdiff`, funnel-shift `shf`, and `*.sat` forms), so naga's GPU-locked
//! result transfers NOTHING, the PTX route needs its own live-GPU proof. No
//! CUDA test exercised mulhi / abs_diff / saturating_{add,sub,mul} /
//! rotate_{left,right} directly with overflow/edge operands; a wrong native
//! instruction, an unmasked rotate amount, or a missing saturate clamp would be
//! invisible.
//!
//! Each op is dispatched on the live device over overflow/identity-boundary
//! operands from `vyre_test_support::binop_parity`, the table the wgpu twin reads
//! too, and asserted byte-for-byte against the shared Rust std reference, which
//! IS the oracle contract. What stays here is the dispatch: the PTX lowering owes
//! its own proof against the reference, never a comparison against what naga
//! produced.

#![cfg(feature = "device-tests")]

mod harness;
use harness::live_backend;

use vyre_driver::parity_harness::u32_binop_parity;
use vyre_driver::DispatchConfig;
use vyre_driver_cuda::CudaBackend;
use vyre_test_support::binop_parity::{
    assert_covers_every_synthetic_op, assert_matches_reference, synthetic_u32_reference,
    synthetic_u32_reference_ops, U32BinopCase, SYNTHETIC_U32_BINOPS,
};

/// What the CUDA arm lowered these ops to, for the failure message.
const LOWERING: &str = "the PTX lowering";

/// Dispatch `case` on the live device and assert byte-for-byte against the
/// shared CPU reference.
fn check(backend: &CudaBackend, case: &U32BinopCase) {
    let pairs = case.pairs();
    let gpu = u32_binop_parity(
        &|program, inputs| backend.dispatch_borrowed(program, inputs, &DispatchConfig::default()),
        case.build,
        &pairs,
        case.op,
    );
    assert_matches_reference(
        "CUDA",
        LOWERING,
        case.op,
        &pairs,
        &gpu,
        synthetic_u32_reference(case.op),
    );
}

/// Every synthetic op in the shared table lowers correctly through PTX.
///
/// One test over the table rather than one test per op: the seven bodies this
/// replaces differed only in the op name and the reference, and the wgpu twin
/// carried the same seven, so an op added to the table was proven by neither
/// until somebody wrote an eighth body twice. The coverage assertion runs
/// first, so a missing reference fails before the device is acquired.
#[test]
fn every_synthetic_binop_matches_its_reference_on_cuda() {
    assert_covers_every_synthetic_op("cuda", &synthetic_u32_reference_ops());
    let backend = live_backend();
    for case in SYNTHETIC_U32_BINOPS {
        check(&backend, case);
    }
}
