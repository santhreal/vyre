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
//! Each op is dispatched on the 5090 over overflow/identity-boundary operands
//! from `vyre_test_support::binop_parity`, the table the wgpu twin reads too, and
//! asserted byte-for-byte against the Rust std reference, which IS the oracle
//! contract. The reference closure and the dispatch below stay here: the PTX
//! lowering owes its own proof against the reference, never a comparison against
//! what naga produced.

mod common;
use common::live_backend;

use vyre_driver::parity_harness::u32_binop_parity;
use vyre_driver::DispatchConfig;
use vyre_driver_cuda::CudaBackend;
use vyre_test_support::binop_parity::{assert_matches_reference, synthetic_u32_case, U32BinopCase};

/// What the CUDA arm lowered these ops to, for the failure message.
const LOWERING: &str = "the PTX lowering";

/// Dispatch `case` on the live device and assert byte-for-byte against `reference`.
fn check(backend: &CudaBackend, case: &U32BinopCase, reference: impl Fn(u32, u32) -> u32) {
    let pairs = case.pairs();
    let gpu = u32_binop_parity(
        &|program, inputs| backend.dispatch_borrowed(program, inputs, &DispatchConfig::default()),
        case.build,
        &pairs,
        case.op,
    );
    assert_matches_reference("CUDA", LOWERING, case.op, &pairs, &gpu, reference);
}

#[test]
fn mulhi_matches_widening_high_word_on_cuda() {
    let reference = |a: u32, b: u32| ((u64::from(a) * u64::from(b)) >> 32) as u32;
    check(&live_backend(), synthetic_u32_case("mulhi"), reference);
    assert_eq!(reference(u32::MAX, u32::MAX), 0xFFFF_FFFE);
    assert_eq!(reference(0x1_0000, 0x1_0000), 1);
}

#[test]
fn abs_diff_matches_unsigned_absolute_difference_on_cuda() {
    check(
        &live_backend(),
        synthetic_u32_case("abs_diff"),
        u32::abs_diff,
    );
    assert_eq!(u32::abs_diff(0, u32::MAX), u32::MAX);
    assert_eq!(u32::abs_diff(100, 50), 50);
}

#[test]
fn saturating_add_clamps_to_max_on_cuda() {
    check(
        &live_backend(),
        synthetic_u32_case("saturating_add"),
        u32::saturating_add,
    );
    assert_eq!(u32::saturating_add(u32::MAX, 1), u32::MAX);
    assert_eq!(u32::saturating_add(0x8000_0000, 0x8000_0000), u32::MAX);
}

#[test]
fn saturating_sub_clamps_to_zero_on_cuda() {
    check(
        &live_backend(),
        synthetic_u32_case("saturating_sub"),
        u32::saturating_sub,
    );
    assert_eq!(u32::saturating_sub(1, u32::MAX), 0);
    assert_eq!(u32::saturating_sub(100, 50), 50);
}

#[test]
fn saturating_mul_clamps_to_max_on_cuda() {
    check(
        &live_backend(),
        synthetic_u32_case("saturating_mul"),
        u32::saturating_mul,
    );
    assert_eq!(u32::saturating_mul(0x1_0000, 0x1_0000), u32::MAX); // 2^32 overflows
    assert_eq!(u32::saturating_mul(1000, 1000), 1_000_000);
}

#[test]
fn rotate_left_matches_barrel_rotate_on_cuda() {
    let reference = |a: u32, b: u32| a.rotate_left(b & 31);
    check(
        &live_backend(),
        synthetic_u32_case("rotate_left"),
        reference,
    );
    assert_eq!(reference(1, 32), 1); // 1<<32 rotate == identity (mask)
    assert_eq!(reference(0x8000_0000, 1), 1); // wrap
    assert_eq!(reference(0xDEAD_BEEF, 4), 0xEADB_EEFD);
}

#[test]
fn rotate_right_matches_barrel_rotate_on_cuda() {
    let reference = |a: u32, b: u32| a.rotate_right(b & 31);
    check(
        &live_backend(),
        synthetic_u32_case("rotate_right"),
        reference,
    );
    assert_eq!(reference(1, 1), 0x8000_0000);
    assert_eq!(reference(1, 32), 1);
    assert_eq!(reference(0xDEAD_BEEF, 4), 0xFDEA_DBEE);
}
