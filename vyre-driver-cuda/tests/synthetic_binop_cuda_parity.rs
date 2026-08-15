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
//!
//! Which ops exist is shared, so an op added to `SYNTHETIC_U32_BINOPS` turns
//! this suite red until it has a reference here.

mod common;
use common::live_backend;

use vyre_driver::parity_harness::u32_binop_parity;
use vyre_driver::DispatchConfig;
use vyre_driver_cuda::CudaBackend;
use vyre_test_support::binop_parity::{
    assert_covers_every_synthetic_op, assert_every_driver_crate_has_a_recorded_parity_position,
    assert_matches_reference, U32BinopCase, SYNTHETIC_U32_BINOPS,
};

/// What the CUDA arm lowered these ops to, for the failure message.
const LOWERING: &str = "the PTX lowering";

/// This backend's independent answer for each op in the shared table.
///
/// Written here, never read from the table, because the table also builds the
/// IR the dispatch runs: a reference taken from the same row would compare the
/// lowering against itself. It is the same arithmetic the naga twin writes, and
/// that is the point: two unrelated lowerings answering to one contract.
const REFERENCES: &[(&str, fn(u32, u32) -> u32)] = &[
    ("mulhi", |a, b| ((u64::from(a) * u64::from(b)) >> 32) as u32),
    ("abs_diff", u32::abs_diff),
    ("saturating_add", u32::saturating_add),
    ("saturating_sub", u32::saturating_sub),
    ("saturating_mul", u32::saturating_mul),
    ("rotate_left", |a, b| a.rotate_left(b & 31)),
    ("rotate_right", |a, b| a.rotate_right(b & 31)),
];

fn reference_for(op: &str) -> fn(u32, u32) -> u32 {
    REFERENCES
        .iter()
        .find(|(name, _)| *name == op)
        .map(|(_, reference)| *reference)
        .unwrap_or_else(|| panic!("Fix: no CUDA reference arm for `{op}`; add one to REFERENCES"))
}

fn covered_ops() -> Vec<&'static str> {
    REFERENCES.iter().map(|(op, _)| *op).collect()
}

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

/// Every synthetic op in the shared table lowers correctly through PTX.
///
/// One test over the table rather than one test per op: the seven bodies this
/// replaces differed only in the op name and the reference, and the wgpu twin
/// carried the same seven, so an op added to the table was proven by neither
/// until somebody wrote an eighth body twice. The coverage assertion runs
/// first, so a missing reference fails before the device is acquired.
#[test]
fn every_synthetic_binop_matches_its_reference_on_cuda() {
    assert_covers_every_synthetic_op("cuda", &covered_ops());
    let backend = live_backend();
    for case in SYNTHETIC_U32_BINOPS {
        check(&backend, case, reference_for(case.op));
    }
}

/// The reference arms above answer the boundary values they are here for.
///
/// A reference that drifts makes the dispatch comparison agree about a wrong
/// answer, and the operand table cannot catch that: it supplies operands, not
/// expectations. Stated on both arms because each owns its own reference, so a
/// mistyped copy on one side must fail on that side.
#[test]
fn the_reference_arms_answer_their_boundary_values() {
    let mulhi = reference_for("mulhi");
    assert_eq!(mulhi(u32::MAX, u32::MAX), 0xFFFF_FFFE);
    assert_eq!(mulhi(0x1_0000, 0x1_0000), 1);

    assert_eq!(u32::abs_diff(0, u32::MAX), u32::MAX);
    assert_eq!(u32::abs_diff(100, 50), 50);
    assert_eq!(u32::saturating_add(u32::MAX, 1), u32::MAX);
    assert_eq!(u32::saturating_add(0x8000_0000, 0x8000_0000), u32::MAX);
    assert_eq!(u32::saturating_sub(1, u32::MAX), 0);
    assert_eq!(u32::saturating_sub(100, 50), 50);
    // 2^16 * 2^16 overflows u32 exactly.
    assert_eq!(u32::saturating_mul(0x1_0000, 0x1_0000), u32::MAX);
    assert_eq!(u32::saturating_mul(1000, 1000), 1_000_000);

    let rotl = reference_for("rotate_left");
    // A rotate of 32 masks to identity; the sign bit wraps to bit 0.
    assert_eq!(rotl(1, 32), 1);
    assert_eq!(rotl(0x8000_0000, 1), 1);
    assert_eq!(rotl(0xDEAD_BEEF, 4), 0xEADB_EEFD);

    let rotr = reference_for("rotate_right");
    assert_eq!(rotr(1, 1), 0x8000_0000);
    assert_eq!(rotr(1, 32), 1);
    assert_eq!(rotr(0xDEAD_BEEF, 4), 0xFDEA_DBEE);
}

/// Every driver crate in the workspace has a recorded position on this gate.
///
/// Asserted from both twins because either one is a complete run, and a
/// backend added while only one suite runs is still a backend proving nothing.
#[test]
fn every_driver_crate_has_a_recorded_parity_position() {
    assert_every_driver_crate_has_a_recorded_parity_position();
}
