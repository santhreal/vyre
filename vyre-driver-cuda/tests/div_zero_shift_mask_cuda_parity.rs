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
//! The operands, the oracle's pinned answer for them, the Rust restatement of the
//! total contract and the coverage gate are `vyre_test_support::binop_parity`,
//! shared with the wgpu twin so both backends prove the same boundary against the
//! same restatement. What is NOT shared is this file: the PTX lowering, the live
//! dispatch, and the note below saying what PTX specifically gets wrong when a
//! row fails.
//!
//! (Signed `i32 / 0` and `i32::MIN / -1` are rejected upstream as undefined, so
//! they are not emittable and not tested here; only the unsigned total cases.)

#![cfg(feature = "device-tests")]

mod harness;
use harness::live_backend;

use vyre_driver::parity_harness::u32_binop_parity;
use vyre_driver::DispatchConfig;
use vyre_driver_cuda::CudaBackend;
use vyre_test_support::binop_parity::{
    assert_covers_every_total_op, total_u32_reference_ops, total_u32_reference_values,
    TotalU32Case, TOTAL_U32_CASES,
};

/// Why a divergence here is a PTX miscompile and not a hardware liberty.
///
/// Per backend because the wrong answer is per backend: an unforced PTX `div.u32`
/// leaves `x / 0` to unspecified hardware, which is not naga's `x`.
const WHY: &[(&str, &str)] = &[
    ("div", "`x / 0 == u32::MAX`."),
    ("rem", "`x % 0 == 0`."),
    (
        "shl",
        "`<< (s & 31)`. A non-masking shift would zero `1 << 32`.",
    ),
    ("shr", "`>> (s & 31)`."),
];

fn why(op: &str) -> &'static str {
    WHY.iter()
        .find(|(name, _)| *name == op)
        .map(|(_, text)| *text)
        .unwrap_or_else(|| panic!("Fix: no CUDA divergence note for `{op}`; add one to WHY"))
}

/// `out[i] = build(a[i], b[i])` over u32 buffers, dispatched on the case operands.
fn dispatch(backend: &CudaBackend, case: &TotalU32Case) -> Vec<u32> {
    u32_binop_parity(
        &|program, inputs| backend.dispatch_borrowed(program, inputs, &DispatchConfig::default()),
        case.build,
        case.pairs,
        case.op,
    )
}

/// Every total op forces its contract on the live device.
///
/// One test over the table rather than one test per op: the four bodies this
/// replaces differed only in the op name, the reference and the note, and the
/// wgpu twin carried the same four, so an op added to the table was proven by
/// neither. The coverage assertion runs first, so a missing reference fails
/// before the device is acquired.
#[test]
fn every_total_u32_op_forces_its_contract_on_cuda() {
    assert_covers_every_total_op("cuda", &total_u32_reference_ops());
    let backend = live_backend();
    for case in TOTAL_U32_CASES {
        let reference = total_u32_reference_values(case);
        let gpu = dispatch(&backend, case);
        assert_eq!(
            gpu,
            reference,
            "CUDA u32 `{}` diverged from the oracle ({}).\n  cases: {:?}\n  \
             expected: {reference:?}\n  gpu: {gpu:?}",
            case.op,
            why(case.op),
            case.pairs
        );
    }
}
