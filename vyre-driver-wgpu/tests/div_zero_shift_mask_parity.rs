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
//! The operands, the oracle's pinned answer for them, the Rust restatement of
//! the total contract and the coverage gate are
//! `vyre_test_support::binop_parity`, shared with the CUDA twin so both backends
//! prove the same boundary against the same restatement. What is NOT shared is
//! this file: the naga lowering, the live dispatch, and the note below saying
//! what naga specifically gets wrong when a row fails.
//!
//! (Signed `i32 / 0` and `i32::MIN / -1` are rejected upstream as undefined
//! `div_i32`/`rem_i32` return an error, so they are not emittable and not
//! tested here; only the unsigned, total cases reach the GPU.)

mod binop_parity_support;
mod common;

use binop_parity_support::program;
use vyre_driver_wgpu::WgpuBackend;
use vyre_test_support::binop_parity::{
    assert_covers_every_total_op, total_u32_reference_ops, total_u32_reference_values,
    TotalU32Case, TOTAL_U32_CASES,
};

/// Why a divergence here is a naga miscompile and not a hardware liberty.
///
/// Per backend because the wrong answer is per backend: naga's bare `Divide`
/// yields `x` for `x / 0`, which is not what an unmasked PTX shift would do.
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

fn why(op: &str) -> &'static str {
    WHY.iter()
        .find(|(name, _)| *name == op)
        .map(|(_, text)| *text)
        .unwrap_or_else(|| panic!("Fix: no wgpu divergence note for `{op}`; add one to WHY"))
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
    assert_covers_every_total_op("wgpu", &total_u32_reference_ops());
    let backend = live_gpu();
    for case in TOTAL_U32_CASES {
        let reference = total_u32_reference_values(case);
        let gpu = dispatch(&backend, case);
        assert_eq!(
            gpu,
            reference,
            "GPU u32 `{}` diverged from the oracle ({}).\n  cases: {:?}\n  \
             expected: {reference:?}\n  gpu: {gpu:?}",
            case.op,
            why(case.op),
            case.pairs
        );
    }
}
