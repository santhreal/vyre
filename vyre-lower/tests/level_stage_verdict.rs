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

use vyre_foundation::ir::DataType;
use vyre_foundation::optimizer::level_contract::{stage_for_level, LevelVerdict};
use vyre_lower::{
    BindingLayout, BindingSlot, BindingVisibility, Dispatch, KernelBody, KernelDescriptor,
    KernelOp, KernelOpKind, LiteralValue, MemoryClass,
};
use vyre_spec::IrLevel;

fn bindings() -> BindingLayout {
    BindingLayout {
        slots: vec![BindingSlot {
            slot: 0,
            element_type: DataType::U32,
            element_count: Some(64),
            memory_class: MemoryClass::Global,
            visibility: BindingVisibility::ReadWrite,
            name: "out".into(),
        }],
    }
}

fn literal(pool_idx: u32, result: u32) -> KernelOp {
    KernelOp {
        kind: KernelOpKind::Literal,
        operands: vec![pool_idx],
        result: Some(result),
    }
}

fn invocation_id(result: u32) -> KernelOp {
    KernelOp {
        kind: KernelOpKind::GlobalInvocationId,
        operands: vec![0],
        result: Some(result),
    }
}

fn store(index: u32, value: u32) -> KernelOp {
    KernelOp {
        kind: KernelOpKind::StoreGlobal,
        operands: vec![0, index, value],
        result: None,
    }
}

fn descriptor(body: KernelBody) -> KernelDescriptor {
    KernelDescriptor {
        id: "level-stage-verdict".into(),
        bindings: bindings(),
        dispatch: Dispatch::new(64, 1, 1),
        body,
    }
}

fn verified_descriptor() -> KernelDescriptor {
    descriptor(KernelBody {
        ops: vec![literal(0, 0), invocation_id(1), store(1, 0)],
        child_bodies: vec![],
        literals: vec![LiteralValue::U32(1)],
    })
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

    let bad = descriptor(KernelBody {
        ops: vec![literal(0, 0), invocation_id(0), store(0, 0)],
        child_bodies: vec![],
        literals: vec![LiteralValue::U32(1)],
    });
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

    let out_of_order = vyre_lower::descriptor_builder::descriptor("emit-order")
        .dispatch(1, 1, 1)
        .body(
            vyre_lower::descriptor_builder::body()
                .literals([LiteralValue::U32(1), LiteralValue::U32(2)])
                .ops([
                    vyre_lower::descriptor_builder::lit(0, 0),
                    vyre_lower::descriptor_builder::op(
                        KernelOpKind::BinOpKind(vyre_foundation::ir::BinOp::Add),
                        [0, 2],
                        3,
                    ),
                    vyre_lower::descriptor_builder::lit(1, 2),
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
        stage.verify(&bindings()),
        LevelVerdict::WrongSubject {
            expected: "KernelDescriptor"
        }
    );
}
