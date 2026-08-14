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
//! The operand table and the comparison are `vyre_test_support::binop_parity`,
//! shared with the CUDA twin of this gate so both backends prove the same
//! boundary. The reference closure and the dispatch below are NOT shared: naga's
//! multi-step synthesis and PTX's native instructions are unrelated lowerings of
//! one contract, and each owes its own live proof against the CPU reference.

mod binop_parity_support;
mod common;

use binop_parity_support::program;
use vyre_driver_wgpu::WgpuBackend;
use vyre_foundation::ir::Program;
use vyre_test_support::binop_parity::{assert_matches_reference, synthetic_u32_case, U32BinopCase};

/// What the wgpu arm lowered these ops to, for the failure message.
const LOWERING: &str = "the multi-step naga synthesis";

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

#[test]
fn mulhi_matches_widening_high_word_on_gpu() {
    let reference = |a: u32, b: u32| ((u64::from(a) * u64::from(b)) >> 32) as u32;
    check(&live_gpu(), synthetic_u32_case("mulhi"), reference);
    // Pin the load-bearing cases literally: MAX*MAX high word, 2^16 squared.
    assert_eq!(reference(u32::MAX, u32::MAX), 0xFFFF_FFFE);
    assert_eq!(reference(0x1_0000, 0x1_0000), 1);
}

#[test]
fn abs_diff_matches_unsigned_absolute_difference_on_gpu() {
    check(&live_gpu(), synthetic_u32_case("abs_diff"), u32::abs_diff);
    assert_eq!(u32::abs_diff(0, u32::MAX), u32::MAX);
    assert_eq!(u32::abs_diff(100, 50), 50);
}

#[test]
fn saturating_add_clamps_to_max_on_gpu() {
    check(
        &live_gpu(),
        synthetic_u32_case("saturating_add"),
        u32::saturating_add,
    );
    assert_eq!(u32::saturating_add(u32::MAX, 1), u32::MAX);
    assert_eq!(u32::saturating_add(0x8000_0000, 0x8000_0000), u32::MAX);
}

#[test]
fn saturating_sub_clamps_to_zero_on_gpu() {
    check(
        &live_gpu(),
        synthetic_u32_case("saturating_sub"),
        u32::saturating_sub,
    );
    assert_eq!(u32::saturating_sub(1, u32::MAX), 0);
    assert_eq!(u32::saturating_sub(100, 50), 50);
}

#[test]
fn saturating_mul_clamps_to_max_on_gpu() {
    check(
        &live_gpu(),
        synthetic_u32_case("saturating_mul"),
        u32::saturating_mul,
    );
    assert_eq!(u32::saturating_mul(0x1_0000, 0x1_0000), u32::MAX); // 2^32 overflows
    assert_eq!(u32::saturating_mul(1000, 1000), 1_000_000);
}

#[test]
fn rotate_left_matches_barrel_rotate_on_gpu() {
    let reference = |a: u32, b: u32| a.rotate_left(b & 31);
    check(&live_gpu(), synthetic_u32_case("rotate_left"), reference);
    // 1<<32 rotate == identity (mask), 0x80000000 rotl 1 == 1 (wrap).
    assert_eq!(reference(1, 32), 1);
    assert_eq!(reference(0x8000_0000, 1), 1);
    assert_eq!(reference(0xDEAD_BEEF, 4), 0xEADB_EEFD);
}

#[test]
fn rotate_right_matches_barrel_rotate_on_gpu() {
    let reference = |a: u32, b: u32| a.rotate_right(b & 31);
    check(&live_gpu(), synthetic_u32_case("rotate_right"), reference);
    assert_eq!(reference(1, 1), 0x8000_0000);
    assert_eq!(reference(1, 32), 1);
    assert_eq!(reference(0xDEAD_BEEF, 4), 0xFDEA_DBEE);
}
