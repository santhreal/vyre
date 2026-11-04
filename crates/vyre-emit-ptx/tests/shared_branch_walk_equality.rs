//! Step 0 equality guard for the structured-control-flow walk clone family.
//!
//! Three analyses walk the same thing: every structured control-flow op in a
//! descriptor body tree, resolving child bodies and assigning each site a
//! flattened op index.
//!
//! - `vyre_lower::analyses::analyze_workgroup_uniform`
//! - `vyre_emit_ptx::patterns::predicated_execution::analyze`
//! - `vyre_emit_ptx::patterns::ldmatrix_cp_async::analyze`
//!
//! Collapsing them onto one walker is only a rehome if every entry point
//! reports exactly what it reported before. This file pins each entry point's
//! output over one shared fixture set, so a change to the shared walker that
//! moves any of them turns this red.
//!
//! Two fixtures pin deliberate disagreements.
//!
//! `nested_if_in_arm` has a branch inside an if-arm. The uniformity walk
//! descends into if-arms and sees it; the predication walk does not descend
//! into if-arms and does not. That is a property of each consumer, not of the
//! walk, and the shared walker keeps it as an explicit descent choice.
//!
//! `copy_in_then_arm` pins a resolved drift. The async-copy walk used to take
//! only the LAST child-body operand of a structured op, which on an
//! if-then-else is the else arm, so a transfer in the then arm was invisible.
//! Nothing documented that as a policy and the other two walks reached both
//! arms; the shared walker resolves every child-body operand, so the transfer
//! is now reported.

use vyre_emit_ptx::patterns::{ldmatrix_cp_async, predicated_execution};
use vyre_emit_ptx::ComputeCapability;
use vyre_foundation::ir::DataType;
use vyre_lower::analyses::{analyze_workgroup_uniform, BranchUniformity};
use vyre_lower::descriptor_builder::{
    body, descriptor, effect, global_ro, lit, op, shared_rw, KernelBodyBuilder,
};
use vyre_lower::{KernelDescriptor, KernelOpKind, LiteralValue};

/// Global load into shared store, no control flow.
fn flat_copy() -> KernelDescriptor {
    descriptor("flat_copy")
        .slot(global_ro(0, DataType::F32, "g"))
        .slot(shared_rw(1, DataType::F32, 64, "s"))
        .dispatch(64, 1, 1)
        .body(
            body()
                .literal(LiteralValue::U32(0))
                .op(lit(0, 0))
                .op(op(KernelOpKind::LoadGlobal, [0, 0], 1))
                .op(effect(KernelOpKind::StoreShared, [1, 0, 1])),
        )
        .build()
}

/// An if-then whose arm itself contains an if-then.
fn nested_if_in_arm() -> KernelDescriptor {
    let inner_arm: KernelBodyBuilder = body().literal(LiteralValue::U32(0)).op(lit(0, 20));
    let arm = body()
        .literal(LiteralValue::Bool(true))
        .op(lit(0, 10))
        .op(effect(KernelOpKind::StructuredIfThen, [10, 0]))
        .child(inner_arm);
    descriptor("nested_if_in_arm")
        .dispatch(64, 1, 1)
        .body(
            body()
                .literal(LiteralValue::Bool(true))
                .op(lit(0, 0))
                .op(effect(KernelOpKind::StructuredIfThen, [0, 0]))
                .child(arm),
        )
        .build()
}

/// A for-loop whose body holds a copy chain and an if-then-else.
fn loop_with_copy_and_branch() -> KernelDescriptor {
    let loop_body = body()
        .literals([LiteralValue::U32(0), LiteralValue::Bool(true)])
        .op(lit(0, 20))
        .op(op(KernelOpKind::LoadGlobal, [0, 20], 21))
        .op(effect(KernelOpKind::StoreShared, [1, 20, 21]))
        .op(lit(1, 22))
        .op(effect(KernelOpKind::StructuredIfThenElse, [22, 0, 1]))
        .child(body().literal(LiteralValue::U32(0)).op(lit(0, 30)))
        .child(body());
    descriptor("loop_with_copy_and_branch")
        .slot(global_ro(0, DataType::F32, "g"))
        .slot(shared_rw(1, DataType::F32, 64, "s"))
        .dispatch(64, 1, 1)
        .body(
            body()
                .literals([LiteralValue::U32(0), LiteralValue::Bool(true)])
                .op(lit(0, 0))
                .op(lit(0, 1))
                .op(effect(
                    KernelOpKind::StructuredForLoop {
                        loop_var: "i".into(),
                    },
                    [0, 1, 0],
                ))
                .child(loop_body),
        )
        .build()
}

/// A block wrapping a region wrapping an if-then.
fn block_region_if() -> KernelDescriptor {
    let region_body = body()
        .literal(LiteralValue::Bool(true))
        .op(lit(0, 30))
        .op(effect(KernelOpKind::StructuredIfThen, [30, 0]))
        .child(body());
    let block_body = body()
        .literal(LiteralValue::Bool(true))
        .op(lit(0, 0))
        .op(effect(
            KernelOpKind::Region {
                generator: "g".into(),
            },
            [0],
        ))
        .child(region_body);
    descriptor("block_region_if")
        .dispatch(64, 1, 1)
        .body(
            body()
                .literal(LiteralValue::Bool(true))
                .op(effect(KernelOpKind::StructuredBlock, [0]))
                .child(block_body),
        )
        .build()
}

/// An if-then-else whose THEN arm holds a global-to-shared transfer.
fn copy_in_then_arm() -> KernelDescriptor {
    let then_arm = body()
        .literal(LiteralValue::U32(0))
        .op(lit(0, 20))
        .op(op(KernelOpKind::LoadGlobal, [0, 20], 21))
        .op(effect(KernelOpKind::StoreShared, [1, 20, 21]));
    descriptor("copy_in_then_arm")
        .slot(global_ro(0, DataType::F32, "g"))
        .slot(shared_rw(1, DataType::F32, 64, "s"))
        .dispatch(64, 1, 1)
        .body(
            body()
                .literal(LiteralValue::Bool(true))
                .op(lit(0, 0))
                .op(effect(KernelOpKind::StructuredIfThenElse, [0, 0, 1]))
                .child(then_arm)
                .child(body()),
        )
        .build()
}

fn fixtures() -> Vec<KernelDescriptor> {
    vec![
        flat_copy(),
        nested_if_in_arm(),
        loop_with_copy_and_branch(),
        block_region_if(),
        copy_in_then_arm(),
    ]
}

fn uniform_sites(desc: &KernelDescriptor) -> Vec<(usize, BranchUniformity)> {
    analyze_workgroup_uniform(desc)
        .branches
        .iter()
        .map(|b| (b.op_index, b.uniformity))
        .collect()
}

fn predication_sites(desc: &KernelDescriptor) -> Vec<(usize, u32, u32, bool)> {
    predicated_execution::analyze(desc)
        .candidates
        .iter()
        .map(|c| {
            (
                c.if_op_index,
                c.then_body_op_count,
                c.else_body_op_count,
                c.has_unsafe_effect,
            )
        })
        .collect()
}

fn async_copy_sites(desc: &KernelDescriptor) -> Vec<(usize, usize, u32, u32)> {
    ldmatrix_cp_async::analyze(desc, ComputeCapability::SM_80)
        .candidates
        .iter()
        .map(|c| {
            (
                c.load_op_index,
                c.store_op_index,
                c.global_binding_slot,
                c.shared_binding_slot,
            )
        })
        .collect()
}

/// Every fixture must carry an expectation; a short table would let a new
/// fixture pass unchecked.
fn check<T: std::fmt::Debug + PartialEq>(
    observed: fn(&KernelDescriptor) -> Vec<T>,
    expected: Vec<Vec<T>>,
) {
    let fixtures = fixtures();
    assert_eq!(
        fixtures.len(),
        expected.len(),
        "every fixture needs a pinned expectation"
    );
    for (desc, want) in fixtures.iter().zip(expected) {
        assert_eq!(observed(desc), want, "kernel {}", desc.id);
    }
}

#[test]
fn uniformity_walk_reports_the_pinned_branch_sites() {
    let expected: Vec<Vec<(usize, BranchUniformity)>> = vec![
        vec![],
        vec![
            (1, BranchUniformity::Uniform),
            (3, BranchUniformity::Uniform),
        ],
        vec![(7, BranchUniformity::Uniform)],
        vec![(4, BranchUniformity::Uniform)],
        vec![(1, BranchUniformity::Uniform)],
    ];
    check(uniform_sites, expected);
}

#[test]
fn predication_walk_reports_the_pinned_branch_sites() {
    let expected: Vec<Vec<(usize, u32, u32, bool)>> = vec![
        vec![],
        // The nested branch inside the arm is NOT visited: predication does
        // not descend into if-arms, it only records that the arm holds an
        // effect it cannot guard with an instruction predicate.
        vec![(1, 2, 0, true)],
        vec![(7, 1, 0, false)],
        vec![(4, 0, 0, false)],
        vec![(1, 3, 0, false)],
    ];
    check(predication_sites, expected);
}

#[test]
fn async_copy_walk_reports_the_pinned_transfer_sites() {
    let expected: Vec<Vec<(usize, usize, u32, u32)>> = vec![
        vec![(1, 2, 0, 1)],
        vec![],
        vec![(4, 5, 0, 1)],
        vec![],
        // Resolved drift: the transfer sits in the THEN arm, which the
        // last-child-operand walk never reached.
        vec![(3, 4, 0, 1)],
    ];
    check(async_copy_sites, expected);
}

/// The three walkers must agree on the flattened index of every structured
/// branch they BOTH choose to visit. Disagreement means one of them derives
/// child-body offsets differently, which is the drift this family is being
/// merged to remove.
#[test]
fn walkers_agree_on_shared_branch_indices() {
    for desc in &fixtures() {
        let uniform: Vec<usize> = uniform_sites(desc).into_iter().map(|(i, _)| i).collect();
        let predication: Vec<usize> = predication_sites(desc)
            .into_iter()
            .map(|(i, _, _, _)| i)
            .collect();
        for index in &predication {
            assert!(
                uniform.contains(index),
                "kernel {}: predication reported branch op {index} that the uniformity walk \
                 never visited; the two walks derive different child-body offsets",
                desc.id
            );
        }
    }
}
