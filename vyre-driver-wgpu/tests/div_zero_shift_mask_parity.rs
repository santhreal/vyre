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
//! proof the device returns `u32::MAX`. These tests dispatch all three on real
//! hardware and assert byte-for-byte against the oracle contract.
//!
//! The operands, the oracle's pinned answer for them, and the coverage gate are
//! `vyre_test_support::binop_parity`, shared with the CUDA twin so both backends
//! prove the same boundary. The reference arms below are NOT shared: they
//! recompute the total contract independently for this lowering, and comparing
//! them against the pin is what catches a drifting reference before the GPU
//! comparison runs. What is shared is which ops exist, so an op added to the
//! table turns this suite red until it has a reference here.
//!
//! (Signed `i32 / 0` and `i32::MIN / -1` are rejected upstream as undefined
//! `div_i32`/`rem_i32` return an error, so they are not emittable and not
//! tested here; only the unsigned, total cases reach the GPU.)

mod binop_parity_support;
mod common;

use binop_parity_support::program;
use vyre_driver_wgpu::WgpuBackend;
use vyre_test_support::binop_parity::{
    assert_covers_every_total_op, total_u32_case, TotalU32Case, TOTAL_U32_CASES,
};

/// This arm's independent answer for each total op in the shared table.
///
/// Written here, never read from the row, because the row also builds the IR
/// the dispatch runs: a reference taken from it would compare the lowering
/// against itself. Each entry restates the total contract in Rust.
const REFERENCES: &[(&str, fn(u32, u32) -> u32)] = &[
    ("div", |a, b| if b == 0 { u32::MAX } else { a / b }),
    ("rem", |a, b| if b == 0 { 0 } else { a % b }),
    // Oracle `shift_u32`: left << (right & 31). wrapping_shl masks identically.
    ("shl", u32::wrapping_shl),
    ("shr", u32::wrapping_shr),
];

/// Why a divergence here is a miscompile and not a hardware liberty.
const WHY: &[(&str, &str)] = &[
    (
        "div",
        "`x / 0 == u32::MAX`. A bare naga Divide would yield `x` here, silently \
         disagreeing with the oracle.",
    ),
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
        .unwrap_or_else(|| panic!("Fix: no wgpu reference arm for `{op}`; add one to REFERENCES"))
}

fn why(op: &str) -> &'static str {
    WHY.iter()
        .find(|(name, _)| *name == op)
        .map(|(_, text)| *text)
        .unwrap_or_else(|| panic!("Fix: no wgpu divergence note for `{op}`; add one to WHY"))
}

fn covered_ops() -> Vec<&'static str> {
    REFERENCES.iter().map(|(op, _)| *op).collect()
}

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

/// Every total op forces its contract on real hardware.
///
/// One test over the table rather than one test per op: the four bodies this
/// replaces differed only in the op name, the reference and the note, and the
/// CUDA twin carried the same four, so an op added to the table was proven by
/// neither. The coverage assertion runs first, so a missing reference fails
/// before the GPU is acquired.
#[test]
fn every_total_u32_op_forces_its_contract_on_gpu() {
    assert_covers_every_total_op("wgpu", &covered_ops());
    let backend = live_gpu();
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
            "GPU u32 `{}` diverged from the oracle ({}).\n  cases: {:?}\n  \
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
