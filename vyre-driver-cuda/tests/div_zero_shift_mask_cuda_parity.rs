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
//! The operands, the oracle's pinned answer for them, and the coverage gate are
//! `vyre_test_support::binop_parity`, shared with the wgpu twin. The reference
//! arms below are NOT shared: they recompute the total contract independently for
//! the PTX arm, and comparing them against the pin is what catches a drifting
//! reference before the device comparison runs. What is shared is which ops
//! exist, so an op added to the table turns this suite red until it has a
//! reference here.
//!
//! (Signed `i32 / 0` and `i32::MIN / -1` are rejected upstream as undefined, so
//! they are not emittable and not tested here; only the unsigned total cases.)

mod common;
use common::live_backend;

use vyre_driver::parity_harness::u32_binop_parity;
use vyre_driver::DispatchConfig;
use vyre_driver_cuda::CudaBackend;
use vyre_test_support::binop_parity::{
    assert_covers_every_total_op, total_u32_case, TotalU32Case, TOTAL_U32_CASES,
};

/// This arm's independent answer for each total op in the shared table.
///
/// Written here, never read from the row, because the row also builds the IR the
/// dispatch runs: a reference taken from it would compare the lowering against
/// itself. Each entry restates the total contract in Rust.
const REFERENCES: &[(&str, fn(u32, u32) -> u32)] = &[
    ("div", |a, b| if b == 0 { u32::MAX } else { a / b }),
    ("rem", |a, b| if b == 0 { 0 } else { a % b }),
    // Oracle `shift_u32`: left << (right & 31). wrapping_shl masks identically.
    ("shl", u32::wrapping_shl),
    ("shr", u32::wrapping_shr),
];

/// Why a divergence here is a miscompile and not a hardware liberty.
const WHY: &[(&str, &str)] = &[
    ("div", "`x / 0 == u32::MAX`."),
    ("rem", "`x % 0 == 0`."),
    (
        "shl",
        "`<< (s & 31)`. A non-masking shift would zero `1 << 32`.",
    ),
    ("shr", "`>> (s & 31)`."),
];

fn reference_for(op: &str) -> fn(u32, u32) -> u32 {
    REFERENCES
        .iter()
        .find(|(name, _)| *name == op)
        .map(|(_, reference)| *reference)
        .unwrap_or_else(|| panic!("Fix: no CUDA reference arm for `{op}`; add one to REFERENCES"))
}

fn why(op: &str) -> &'static str {
    WHY.iter()
        .find(|(name, _)| *name == op)
        .map(|(_, text)| *text)
        .unwrap_or_else(|| panic!("Fix: no CUDA divergence note for `{op}`; add one to WHY"))
}

fn covered_ops() -> Vec<&'static str> {
    REFERENCES.iter().map(|(op, _)| *op).collect()
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
    assert_covers_every_total_op("cuda", &covered_ops());
    let backend = live_backend();
    for case in TOTAL_U32_CASES {
        let reference: Vec<u32> = case
            .pairs
            .iter()
            .map(|&(a, b)| reference_for(case.op)(a, b))
            .collect();
        assert_eq!(
            reference, case.oracle,
            "reference u32 `{}` total contract drifted from the pinned oracle",
            case.op
        );
        let gpu = dispatch(&backend, case);
        assert_eq!(
            gpu, reference,
            "CUDA u32 `{}` diverged from the oracle ({}).\n  cases: {:?}\n  \
             expected: {reference:?}\n  gpu: {gpu:?}",
            case.op,
            why(case.op),
            case.pairs
        );
    }
}

/// The reference arms answer the boundary values they are here for.
///
/// A reference that drifts row-for-row with the pin makes both agree about a
/// wrong answer. These are the load-bearing values as literals, so a mistyped
/// reference fails here rather than passing everywhere.
#[test]
fn the_reference_arms_answer_their_boundary_values() {
    let div = reference_for("div");
    assert_eq!(div(1, 0), u32::MAX);
    assert_eq!(div(u32::MAX, 0), u32::MAX);
    assert_eq!(div(100, 7), 14);

    let rem = reference_for("rem");
    assert_eq!(rem(1, 0), 0);
    assert_eq!(rem(100, 7), 2);

    let shl = reference_for("shl");
    // A shift of 32 masks to zero, so `1 << 32 == 1`, never 0.
    assert_eq!(shl(1, 32), 1);
    assert_eq!(shl(1, 63), 0x8000_0000);

    let shr = reference_for("shr");
    assert_eq!(shr(1, 32), 1);
    assert_eq!(shr(0xFF, 36), 0xF);
}

/// The shared lookup refuses an op the table does not declare.
///
/// The adversarial case at the boundary of `total_u32_case`: a renamed op must
/// fail at the lookup rather than silently dispatch nothing.
#[test]
#[should_panic(expected = "no total u32 case")]
fn an_undeclared_op_name_is_refused_by_the_shared_lookup() {
    let _ = total_u32_case("div_but_renamed");
}
