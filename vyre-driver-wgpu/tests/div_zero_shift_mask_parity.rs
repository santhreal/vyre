//! Div/mod-by-zero and oversized-shift parity against the reference oracle on
//! the live GPU (the "undefined on hardware, TOTAL on the oracle" class (Law 10)).
//!
//! Three operations have hardware-undefined behavior that the vyre-reference
//! oracle nonetheless defines with a single total contract, so the wgpu backend
//! must force that contract or silently disagree with its own oracle:
//!
//!   * `u32 x / 0`  -> `u32::MAX`   (oracle `div_u32`; naga would yield `x`,
//!                                   PTX leaves it to unspecified hardware)
//!   * `u32 x % 0`  -> `0`          (oracle `rem_u32`)
//!   * `u32 x << s` / `x >> s` with `s >= 32` -> shift amount taken `& 31`
//!                                   (oracle `shift_u32`; SPIR-V/WGSL mask the
//!                                   amount to the bit width, but that is never
//!                                   verified against the oracle on real silicon)
//!
//! op_dispatch forces the div/mod sentinels with a `Select(divisor == 0, ...)`.
//! That Select-forced value is a COMPUTED lowering exactly like the naga
//! signed-`Modulo` bug, a source read ("we emit a Select to u32::MAX") is NOT
//! proof the 5090 returns `u32::MAX`. These tests dispatch all three on real
//! hardware and assert byte-for-byte against the oracle contract.
//!
//! The operands and the oracle's pinned answer for them are
//! `vyre_test_support::binop_parity::TOTAL_U32_CASES`, shared with the CUDA twin
//! so both backends prove the same boundary. The reference closure below is not
//! shared: it recomputes the total contract independently, and comparing it
//! against the pin is what catches a drifting reference before the GPU
//! comparison runs.
//!
//! (Signed `i32 / 0` and `i32::MIN / -1` are rejected upstream as undefined
//! `div_i32`/`rem_i32` return an error, so they are not emittable and not
//! tested here; only the unsigned, total cases reach the GPU.)

mod binop_parity_support;
mod common;

use binop_parity_support::program;
use vyre_driver_wgpu::WgpuBackend;
use vyre_test_support::binop_parity::{total_u32_case, TotalU32Case};

fn live_gpu() -> WgpuBackend {
    WgpuBackend::acquire().expect("Fix: div-zero / shift-mask parity requires a live GPU backend.")
}

/// `out[i] = build(a[i], b[i])` over u32 buffers, dispatched on the case operands.
fn dispatch(backend: &WgpuBackend, case: &TotalU32Case) -> Vec<u32> {
    binop_parity_support::dispatch(
        backend,
        &program(case.pairs.len() as u32, case.build),
        case.pairs,
        "div-zero / shift-mask parity contract",
    )
}

#[test]
fn u32_div_by_zero_yields_max_on_gpu() {
    let case = total_u32_case("div");
    let gpu = dispatch(&live_gpu(), case);
    let reference: Vec<u32> = case
        .pairs
        .iter()
        .map(|&(a, b)| if b == 0 { u32::MAX } else { a / b })
        .collect();
    assert_eq!(
        reference, case.oracle,
        "reference u32 div-by-zero contract drifted from the pinned oracle"
    );
    assert_eq!(
        gpu, reference,
        "GPU u32 `/ 0` diverged from the oracle (`x / 0 == u32::MAX`). A bare naga \
         Divide would yield `x` here, silently disagreeing with the oracle.\n  \
         cases: {:?}\n  expected: {reference:?}\n  gpu: {gpu:?}",
        case.pairs
    );
}

#[test]
fn u32_mod_by_zero_yields_zero_on_gpu() {
    let case = total_u32_case("rem");
    let gpu = dispatch(&live_gpu(), case);
    let reference: Vec<u32> = case
        .pairs
        .iter()
        .map(|&(a, b)| if b == 0 { 0 } else { a % b })
        .collect();
    assert_eq!(
        reference, case.oracle,
        "reference u32 mod-by-zero contract drifted from the pinned oracle"
    );
    assert_eq!(
        gpu, reference,
        "GPU u32 `% 0` diverged from the oracle (`x % 0 == 0`).\n  cases: {:?}\n  \
         expected: {reference:?}\n  gpu: {gpu:?}",
        case.pairs
    );
}

#[test]
fn u32_oversized_shift_left_masks_amount_on_gpu() {
    let case = total_u32_case("shl");
    let gpu = dispatch(&live_gpu(), case);
    // Oracle `shift_u32`: left << (right & 31). wrapping_shl masks identically.
    let reference: Vec<u32> = case.pairs.iter().map(|&(v, s)| v.wrapping_shl(s)).collect();
    assert_eq!(
        reference, case.oracle,
        "reference u32 oversized shift-left contract drifted from the pinned oracle"
    );
    assert_eq!(
        gpu, reference,
        "GPU u32 `<<` with amount >= 32 diverged from the oracle (`<< (s & 31)`). A \
         non-masking shift would zero `1 << 32`.\n  cases: {:?}\n  expected: {reference:?}\n  \
         gpu: {gpu:?}",
        case.pairs
    );
}

#[test]
fn u32_oversized_shift_right_masks_amount_on_gpu() {
    let case = total_u32_case("shr");
    let gpu = dispatch(&live_gpu(), case);
    let reference: Vec<u32> = case.pairs.iter().map(|&(v, s)| v.wrapping_shr(s)).collect();
    assert_eq!(
        reference, case.oracle,
        "reference u32 oversized shift-right contract drifted from the pinned oracle"
    );
    assert_eq!(
        gpu, reference,
        "GPU u32 `>>` with amount >= 32 diverged from the oracle (`>> (s & 31)`).\n  \
         cases: {:?}\n  expected: {reference:?}\n  gpu: {gpu:?}",
        case.pairs
    );
}
