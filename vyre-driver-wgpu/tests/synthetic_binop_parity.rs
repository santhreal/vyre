//! Parity for the SYNTHETIC u32 binop lowerings against Rust/oracle on the
//! live GPU.
//!
//! naga has no native instruction for these ops; op_dispatch synthesizes each
//! from a multi-step expression, the exact class of computed lowering the naga
//! signed-`Modulo` bug proved can be silently wrong on real silicon:
//!   * `mulhi`         -> 16-bit decomposition (al*bl + cross terms + ah*bh)
//!   * `abs_diff`      -> `select(a < b, b - a, a - b)`
//!   * `saturating_add`-> `select(a + b < a, MAX, a + b)`
//!   * `saturating_sub`-> `select(a < b, 0, a - b)`
//!   * `saturating_mul`-> `select(b != 0 && a > MAX/b, MAX, a * b)`
//!   * `rotate_left/right` -> `(x << (s&31)) | (x >> ((32-(s&31))&31))`
//!
//! Rotate is exercised inside the real BLAKE3 workload by
//! `blake3_compress_gpu_parity`; this isolates every synthetic op directly with
//! overflow/edge operands and asserts byte-for-byte against the Rust std
//! reference, which IS the oracle contract (`saturating_add`, `abs_diff`,
//! `rotate_left`, widening `mulhi`).
//!
//! The operand table, the comparison, and the coverage gate are
//! `vyre_test_support::binop_parity`, shared with the CUDA twin of this gate so
//! both backends prove the same boundary. The reference closures below are NOT
//! shared: naga's multi-step synthesis and PTX's native instructions are
//! unrelated lowerings of one contract, and each owes its own live proof
//! against the CPU reference. What is shared is which ops exist, so an op added
//! to the table turns this suite red until it has a reference here.

mod binop_parity_support;
mod common;

use binop_parity_support::program;
use vyre_driver_wgpu::WgpuBackend;
use vyre_foundation::ir::Program;
use vyre_test_support::binop_parity::{
    assert_covers_every_synthetic_op, assert_every_driver_crate_has_a_recorded_parity_position,
    assert_matches_reference, synthetic_u32_case, U32BinopCase, SYNTHETIC_U32_BINOPS,
};

/// What the wgpu arm lowered these ops to, for the failure message.
const LOWERING: &str = "the multi-step naga synthesis";

/// This backend's independent answer for each op in the shared table.
///
/// Written here, never read from the table, because the table also builds the
/// IR the dispatch runs: a reference taken from the same row would compare the
/// lowering against itself. Every entry is Rust std or one expression over it,
/// which is the oracle contract for these ops.
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
        .unwrap_or_else(|| panic!("Fix: no wgpu reference arm for `{op}`; add one to REFERENCES"))
}

fn covered_ops() -> Vec<&'static str> {
    REFERENCES.iter().map(|(op, _)| *op).collect()
}

fn dispatch(backend: &WgpuBackend, program: &Program, pairs: &[(u32, u32)]) -> Vec<u32> {
    binop_parity_support::dispatch(backend, program, pairs, "synthetic-binop parity contract")
}

/// Dispatch `case` on the live GPU and assert byte-for-byte against `reference`.
fn check(backend: &WgpuBackend, case: &U32BinopCase, reference: impl Fn(u32, u32) -> u32) {
    let pairs = case.pairs();
    let gpu = dispatch(backend, &program(pairs.len() as u32, case.build), &pairs);
    assert_matches_reference("GPU synthetic", LOWERING, case.op, &pairs, &gpu, reference);
}

fn live_gpu() -> WgpuBackend {
    WgpuBackend::acquire().expect("Fix: synthetic-binop parity needs a live GPU.")
}

/// Every synthetic op in the shared table lowers correctly through naga.
///
/// One test over the table rather than one test per op: the seven bodies this
/// replaces differed only in the op name and the reference, and the CUDA twin
/// carried the same seven, so an op added to the table was proven by neither
/// until somebody wrote an eighth body twice. The coverage assertion runs
/// first, so a missing reference fails before the GPU is acquired.
#[test]
fn every_synthetic_binop_matches_its_reference_on_gpu() {
    assert_covers_every_synthetic_op("wgpu", &covered_ops());
    let backend = live_gpu();
    for case in SYNTHETIC_U32_BINOPS {
        check(&backend, case, reference_for(case.op));
    }
}

/// The reference arms above answer the boundary values they are here for.
///
/// A reference that drifts makes the dispatch comparison agree about a wrong
/// answer, and the operand table cannot catch that: it supplies operands, not
/// expectations. These are the load-bearing values written as literals, so a
/// mistyped reference fails here rather than passing everywhere.
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
/// Read from the tree, so a fifth backend turns this RED before it can ship
/// unproven.
#[test]
fn every_driver_crate_has_a_recorded_parity_position() {
    assert_every_driver_crate_has_a_recorded_parity_position();
}

/// The shared lookup refuses an op the table does not declare.
///
/// The adversarial case at the boundary of `synthetic_u32_case`: a renamed op
/// must fail at the lookup rather than silently dispatch nothing.
#[test]
#[should_panic(expected = "no synthetic u32 binop case")]
fn an_undeclared_op_name_is_refused_by_the_shared_lookup() {
    let _ = synthetic_u32_case("mulhi_but_renamed");
}
