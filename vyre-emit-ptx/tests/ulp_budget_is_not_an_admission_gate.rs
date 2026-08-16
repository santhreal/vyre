//! WHY: closes the class "the emitter refuses a program for want of a
//! `ulp_budget` it cannot usefully choose".
//!
//! `PtxEmitOptions::ulp_budget` is a preference between two lowerings, and PTX
//! offers two only for `InverseSqrt` and `Reciprocal`. For `tanh`, `ex2`, `lg2`,
//! `sin` and `cos` the approximate instruction is the only one there is, so a
//! budget selects nothing; refusing without it made the program unemittable for
//! want of a value no caller could usefully pick. Every route then had to pass a
//! positive budget, and the route that did not, the production artifact route in
//! the CUDA driver, refused 21 registered ops and took them out of the
//! conformance certificate.
//!
//! The invariant asserted here is over admission, never over instruction
//! choice: a budget may change which instruction an op with two forms uses, and
//! may never decide whether the emitter accepts the descriptor at all.
//!
//! The roster is `tests/support/spec_variant_tables.rs`, held to the frozen
//! `vyre_spec` public surface at run time by
//! `vyre-spec/tests/spec_variant_tables_cover_the_frozen_surface.rs`. A `UnOp`
//! added to the enum and gated on the budget turns this red without anyone
//! editing a list here.
//!
//! Does not catch: an op the emitter refuses under both settings, which is a
//! missing lowering rather than a budget gate; or a budget that moves the
//! emitted numbers outside the parity window, which `vyre_foundation::fp_parity`
//! owns and the conformance comparator enforces.

#![forbid(unsafe_code)]

#[path = "../../tests/support/spec_variant_tables.rs"]
mod spec_variant_tables;

use spec_variant_tables::builtin_un_ops;
use vyre_emit_ptx::{emit_with_options, ComputeCapability, PtxEmitOptions};
use vyre_foundation::ir::{DataType, UnOp};
use vyre_lower::descriptor_builder::{
    body, descriptor, effect, global_ro, global_wo, lit, op, SlotCount,
};
use vyre_lower::{KernelDescriptor, KernelOpKind, LiteralValue};

fn options(ulp_budget: Option<u32>) -> PtxEmitOptions {
    PtxEmitOptions {
        target: ComputeCapability { major: 8, minor: 0 },
        subgroup_size: 32,
        ulp_budget,
        cooperative_grid_sync: false,
    }
}

/// One f32 load, one unary op, one f32 store, so the emitter sees exactly the
/// op under test. Not held to numeric sense: several `UnOp` variants are
/// integer-only, and the emitter is expected to refuse those under both
/// settings, which this suite reads as "not a budget gate" rather than a pass.
fn single_un_op_f32_descriptor(unary: UnOp) -> KernelDescriptor {
    descriptor("single_un_op")
        .slot(global_ro(0, DataType::F32, "input").with_count(1))
        .slot(global_wo(1, DataType::F32, "out").with_count(1))
        .body(
            body()
                .ops([
                    lit(0, 0),
                    op(KernelOpKind::LoadGlobal, [0, 0], 1),
                    op(KernelOpKind::UnOpKind(unary), [1], 2),
                    effect(KernelOpKind::StoreGlobal, [1, 0, 2]),
                ])
                .literal(LiteralValue::U32(0)),
        )
        .build()
}

#[test]
fn no_un_op_needs_a_ulp_budget_to_be_admitted() {
    let mut budget_gated = Vec::new();
    for unary in builtin_un_ops() {
        let kernel = single_un_op_f32_descriptor(unary.clone());
        if emit_with_options(&kernel, options(Some(64))).is_err() {
            continue;
        }
        if let Err(error) = emit_with_options(&kernel, options(None)) {
            budget_gated.push(format!("{unary:?}: {error}"));
        }
    }
    assert!(
        budget_gated.is_empty(),
        "Fix: these UnOp lowerings admit the descriptor only when a ULP budget is set, so a route \
         that passes none refuses a program this emitter can emit. Emit the one instruction the \
         dialect has instead of refusing, and keep the budget for the ops where the dialect \
         offers a choice.\n{}",
        budget_gated.join("\n")
    );
}

#[test]
fn a_budget_still_chooses_between_the_two_forms_ptx_offers() {
    let kernel = single_un_op_f32_descriptor(UnOp::InverseSqrt);
    let approximate = emit_with_options(&kernel, options(Some(64)))
        .expect("Fix: a budgeted inverse-sqrt must emit.");
    let strict =
        emit_with_options(&kernel, options(None)).expect("Fix: a strict inverse-sqrt must emit.");
    assert!(
        approximate.contains("rsqrt.approx.f32"),
        "Fix: a positive budget must select the approximate reciprocal-sqrt; got:\n{approximate}"
    );
    assert!(
        !strict.contains("rsqrt.approx.f32") && strict.contains("rcp.rn.f32"),
        "Fix: no budget must keep the exact reciprocal-sqrt sequence; got:\n{strict}"
    );
}

#[test]
fn an_approximate_only_op_emits_the_same_instruction_under_both_settings() {
    for unary in [UnOp::Tanh, UnOp::Exp2, UnOp::Log2, UnOp::Sin, UnOp::Cos] {
        let kernel = single_un_op_f32_descriptor(unary.clone());
        let strict = emit_with_options(&kernel, options(None))
            .unwrap_or_else(|error| panic!("Fix: {unary:?} must emit without a budget: {error}"));
        let budgeted = emit_with_options(&kernel, options(Some(64)))
            .unwrap_or_else(|error| panic!("Fix: {unary:?} must emit with a budget: {error}"));
        assert_eq!(
            strict, budgeted,
            "Fix: {unary:?} has one PTX instruction, so the budget must not change its emission"
        );
    }
}
