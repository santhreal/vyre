//! Div/mod-by-zero and oversized-shift parity against the reference oracle on the
//! live CUDA device, the "undefined on hardware, TOTAL on the oracle" class
//! (Law 10), the PTX/CUDA twin of the wgpu `div_zero_shift_mask_parity` gate.
//!
//! The oracle defines three hardware-undefined ops with a single total contract:
//!   * `u32 x / 0`  -> `u32::MAX`   (oracle `div_u32`; PTX forces it with
//!                                   `emit_total_u32_div`: default 0xffffffff +
//!                                   `@pred bra` over the `div`)
//!   * `u32 x % 0`  -> `0`          (oracle `rem_u32`; PTX `emit_total_u32_mod`)
//!   * `u32 x << s` / `x >> s`, `s >= 32` -> amount taken `& 31` (oracle
//!                                   `shift_u32`; PTX masks with `and.b32 ...,31`)
//!
//! The generated scalar matrix already exercises the zero divisor (`lane % 13 ==
//! 0 => 0`), but it PRE-MASKS shift amounts (`RhsKind::Shift => value & 31`), so
//! it NEVER sends `s >= 32` and the PTX `and.b32 ...,31` mask is unverified on
//! hardware, exactly the source-read-vs-silicon gap the naga signed-`Modulo`
//! miscompile punished. These tests dispatch all three with oversized amounts and
//! literal-pinned zero-divisor sentinels, byte-for-byte against the oracle.
//!
//! The operands and the oracle's pinned answer for them are
//! `vyre_test_support::binop_parity::TOTAL_U32_CASES`, shared with the wgpu twin.
//! The reference closure below is not shared: it recomputes the total contract
//! independently for the PTX arm, and comparing it against the pin is what
//! catches a drifting reference before the device comparison runs.
//!
//! (Signed `i32 / 0` and `i32::MIN / -1` are rejected upstream as undefined, so
//! they are not emittable and not tested here; only the unsigned total cases.)

mod common;
use common::live_backend;

use vyre_driver::parity_harness::u32_binop_parity;
use vyre_driver::DispatchConfig;
use vyre_driver_cuda::CudaBackend;
use vyre_test_support::binop_parity::{total_u32_case, TotalU32Case};

/// `out[i] = build(a[i], b[i])` over u32 buffers, dispatched on the case operands.
fn dispatch(backend: &CudaBackend, case: &TotalU32Case) -> Vec<u32> {
    u32_binop_parity(
        &|program, inputs| backend.dispatch_borrowed(program, inputs, &DispatchConfig::default()),
        case.build,
        case.pairs,
        case.op,
    )
}

#[test]
fn u32_div_by_zero_yields_max_on_cuda() {
    let case = total_u32_case("div");
    let gpu = dispatch(&live_backend(), case);
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
        "CUDA u32 `/ 0` diverged from the oracle (`x / 0 == u32::MAX`).\n  \
         cases: {:?}\n  expected: {reference:?}\n  gpu: {gpu:?}",
        case.pairs
    );
}

#[test]
fn u32_mod_by_zero_yields_zero_on_cuda() {
    let case = total_u32_case("rem");
    let gpu = dispatch(&live_backend(), case);
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
        "CUDA u32 `% 0` diverged from the oracle (`x % 0 == 0`).\n  cases: {:?}\n  \
         expected: {reference:?}\n  gpu: {gpu:?}",
        case.pairs
    );
}

#[test]
fn u32_oversized_shift_left_masks_amount_on_cuda() {
    let case = total_u32_case("shl");
    let gpu = dispatch(&live_backend(), case);
    // Oracle `shift_u32`: left << (right & 31). wrapping_shl masks identically.
    let reference: Vec<u32> = case.pairs.iter().map(|&(v, s)| v.wrapping_shl(s)).collect();
    assert_eq!(
        reference, case.oracle,
        "reference u32 oversized shift-left contract drifted from the pinned oracle"
    );
    assert_eq!(
        gpu, reference,
        "CUDA u32 `<<` with amount >= 32 diverged from the oracle (`<< (s & 31)`). A \
         non-masking shift would zero `1 << 32`.\n  cases: {:?}\n  expected: {reference:?}\n  \
         gpu: {gpu:?}",
        case.pairs
    );
}

#[test]
fn u32_oversized_shift_right_masks_amount_on_cuda() {
    let case = total_u32_case("shr");
    let gpu = dispatch(&live_backend(), case);
    let reference: Vec<u32> = case.pairs.iter().map(|&(v, s)| v.wrapping_shr(s)).collect();
    assert_eq!(
        reference, case.oracle,
        "reference u32 oversized shift-right contract drifted from the pinned oracle"
    );
    assert_eq!(
        gpu, reference,
        "CUDA u32 `>>` with amount >= 32 diverged from the oracle (`>> (s & 31)`).\n  \
         cases: {:?}\n  expected: {reference:?}\n  gpu: {gpu:?}",
        case.pairs
    );
}
