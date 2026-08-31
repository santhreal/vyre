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
//!   * `saturating_mul`-> `select(mulhi(a, b) != 0, MAX, a * b)`
//!   * `rotate_left/right` -> `(x << (s&31)) | (x >> ((32-(s&31))&31))`
//!
//! Rotate is exercised inside the real BLAKE3 workload by
//! `blake3_compress_gpu_parity`; this isolates every synthetic op directly with
//! overflow/edge operands and asserts byte-for-byte against the Rust std
//! reference, which IS the oracle contract (`saturating_add`, `abs_diff`,
//! `rotate_left`, widening `mulhi`).
//!
//! The operand table, the CPU reference and the coverage gate are
//! `vyre_test_support::binop_parity`, shared with the CUDA twin so both backends
//! prove the same boundary against the same restatement of the contract. What is
//! NOT shared is this file: the naga synthesis and the live dispatch. An op
//! added to the shared table without a reference fails at the lookup, before the
//! GPU is acquired.

#![cfg(feature = "device-tests")]

mod binop_parity_fixtures;
mod harness;

use binop_parity_fixtures::program;
use vyre_driver_wgpu::WgpuBackend;
use vyre_foundation::ir::Program;
use vyre_test_support::binop_parity::{
    assert_covers_every_synthetic_op, assert_matches_reference, synthetic_u32_reference,
    synthetic_u32_reference_ops, U32BinopCase, SYNTHETIC_U32_BINOPS,
};

/// What the wgpu arm lowered these ops to, for the failure message.
const LOWERING: &str = "the multi-step naga synthesis";

fn dispatch(backend: &WgpuBackend, program: &Program, pairs: &[(u32, u32)]) -> Vec<u32> {
    binop_parity_fixtures::dispatch(backend, program, pairs, "synthetic-binop parity contract")
}

/// Dispatch `case` on the live GPU and assert byte-for-byte against the shared
/// CPU reference.
fn check(backend: &WgpuBackend, case: &U32BinopCase) {
    let pairs = case.pairs();
    let gpu = dispatch(backend, &program(pairs.len() as u32, case.build), &pairs);
    assert_matches_reference(
        "GPU synthetic",
        LOWERING,
        case.op,
        &pairs,
        &gpu,
        synthetic_u32_reference(case.op),
    );
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
    assert_covers_every_synthetic_op("wgpu", &synthetic_u32_reference_ops());
    let backend = live_gpu();
    for case in SYNTHETIC_U32_BINOPS {
        check(&backend, case);
    }
}
