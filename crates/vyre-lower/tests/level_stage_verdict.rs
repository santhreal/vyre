//! The physical-kernel level accepts a verified descriptor and rejects a broken one.
//!
//! WHY this suite exists: the level stage this crate registers is what the level
//! registry calls to verify a physical kernel, and a stage that answers
//! `Verified` for every descriptor certifies what it never checked while reading
//! exactly like a stage that works. Each case pairs a descriptor the level must
//! accept with one it must reject, so a verifier replaced by a constant answer
//! turns this suite red.
//!
//! The stage's canonical predicate is held the same way: a descriptor the
//! emitter-ready canonicalization rewrites is not canonical, and one it leaves
//! alone is.

use vyre_foundation::ir::{BinOp, DataType};
use vyre_foundation::optimizer::level_contract::{stage_for_level, LevelVerdict};
use vyre_lower::descriptor_builder::{
    body, descriptor, global_rw, lit, op, store_global, SlotCount,
};
use vyre_lower::{KernelDescriptor, KernelOp, KernelOpKind, LiteralValue};
use vyre_spec::IrLevel;

fn invocation_id(result: u32) -> KernelOp {
    op(KernelOpKind::GlobalInvocationId, [0], result)
}

/// One 64-element read-write output over a 64-invocation dispatch, carrying
/// `ops` over a one-entry literal pool.
fn kernel(ops: [KernelOp; 3]) -> KernelDescriptor {
    descriptor("level-stage-verdict")
        .slot(global_rw(0, DataType::U32, "out").with_count(64))
        .dispatch(64, 1, 1)
        .body(body().literals([LiteralValue::U32(1)]).ops(ops))
        .build()
}

fn verified_descriptor() -> KernelDescriptor {
    kernel([lit(0, 0), invocation_id(1), store_global(0, 1, 0)])
}

/// A verified descriptor passes; one assigning a result identifier twice does not.
#[test]
fn physical_kernel_stage_rejects_a_descriptor_the_verifier_rejects() {
    let stage = stage_for_level(IrLevel::PhysicalKernel)
        .expect("Fix: the physical-kernel stage must exist");

    let good = verified_descriptor();
    assert_eq!(
        stage.verify(&good),
        LevelVerdict::Verified,
        "Fix: a descriptor the verifier accepts must verify at the physical-kernel level"
    );

    let bad = kernel([lit(0, 0), invocation_id(0), store_global(0, 0, 0)]);
    let verdict = stage.verify(&bad);
    assert!(
        matches!(verdict, LevelVerdict::Rejected(_)),
        "Fix: a descriptor assigning one result identifier twice must be rejected, got {verdict:?}"
    );
}

/// The canonical predicate answers for the descriptor it is given, not always.
#[test]
fn physical_kernel_stage_reports_a_non_canonical_descriptor() {
    let stage = stage_for_level(IrLevel::PhysicalKernel)
        .expect("Fix: the physical-kernel stage must exist");

    let canonical = vyre_lower::verify_descriptor(&verified_descriptor())
        .expect("Fix: the fixture descriptor must verify and canonicalize");
    assert_eq!(
        stage.is_canonical(&canonical),
        LevelVerdict::Verified,
        "Fix: the canonicalized descriptor must be reported canonical"
    );

    let out_of_order = descriptor("out-of-order")
        .dispatch(1, 1, 1)
        .body(
            body()
                .literals([LiteralValue::U32(4), LiteralValue::U32(7)])
                .ops([
                    op(KernelOpKind::BinOpKind(BinOp::Add), [1, 2], 3),
                    lit(1, 2),
                ]),
        )
        .build();
    let verdict = stage.is_canonical(&out_of_order);
    assert!(
        matches!(verdict, LevelVerdict::Rejected(_)),
        "Fix: a descriptor the canonicalization rewrites is not canonical, got {verdict:?}"
    );
}

/// The stage refuses a subject of another level rather than verifying it.
#[test]
fn physical_kernel_stage_refuses_another_levels_subject() {
    let stage = stage_for_level(IrLevel::PhysicalKernel)
        .expect("Fix: the physical-kernel stage must exist");
    assert_eq!(
        stage.verify(&global_rw(0, DataType::U32, "out")),
        LevelVerdict::WrongSubject {
            expected: "KernelDescriptor"
        }
    );
}
