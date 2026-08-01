//! Descriptor-wide result-id uniqueness (verifier invariant 6).
//!
//! `verify_body` collects the produced-id set fresh per body, so it only
//! ever caught two ops in the SAME op list assigning one id. Reuse
//! between two different bodies passed verification silently, and every
//! backend keys its register map on the raw id: the PTX emitter's
//! `u32_literals` map made a `GlobalInvocationId` store index resolve to
//! a sibling body's `Literal(1)`, so the emitted address was the
//! constant `[%rd4+4]` and every thread wrote element 1 instead of its
//! own. These tests pin that the verifier now rejects each shape of
//! cross-body reuse rather than leaving it to a backend to miscompile.

use vyre_foundation::ir::DataType;
use vyre_lower::verify::{verify, VerifyErrorKind};
use vyre_lower::{
    BindingLayout, BindingSlot, BindingVisibility, Dispatch, KernelBody, KernelDescriptor,
    KernelOp, KernelOpKind, LiteralValue, MemoryClass,
};

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

fn if_then(condition: u32, child: u32) -> KernelOp {
    KernelOp {
        kind: KernelOpKind::StructuredIfThen,
        operands: vec![condition, child],
        result: None,
    }
}

fn descriptor(body: KernelBody) -> KernelDescriptor {
    KernelDescriptor {
        id: "verify-id-uniqueness".into(),
        bindings: bindings(),
        dispatch: Dispatch::new(64, 1, 1),
        body,
    }
}

fn reuse_errors(desc: &KernelDescriptor) -> Vec<(u32, Vec<usize>, Vec<usize>)> {
    match verify(desc) {
        Ok(()) => Vec::new(),
        Err(errors) => errors
            .into_iter()
            .filter_map(|error| match error.kind {
                VerifyErrorKind::ResultIdReusedAcrossBodies {
                    result,
                    first_body_path,
                } => Some((result, first_body_path, error.body_path)),
                _ => None,
            })
            .collect(),
    }
}

/// A descriptor whose bodies all use distinct ids stays clean. Without
/// this, a verifier that rejected everything would pass the tests below.
#[test]
fn distinct_ids_across_bodies_verify_clean() {
    let child = KernelBody {
        ops: vec![literal(0, 3), store(3, 3)],
        child_bodies: vec![],
        literals: vec![LiteralValue::U32(1)],
    };
    let desc = descriptor(KernelBody {
        ops: vec![literal(0, 0), invocation_id(1), if_then(0, 0), store(1, 0)],
        child_bodies: vec![child],
        literals: vec![LiteralValue::U32(1)],
    });
    assert_eq!(verify(&desc), Ok(()));
}

/// Parent and child assigning the same id. The child can read the
/// parent's value by that id, so the reference is ambiguous in the IR
/// itself, not merely in a backend's flat map.
#[test]
fn parent_and_child_sharing_an_id_is_rejected() {
    let child = KernelBody {
        ops: vec![literal(0, 1), store(1, 1)],
        child_bodies: vec![],
        literals: vec![LiteralValue::U32(1)],
    };
    let desc = descriptor(KernelBody {
        ops: vec![literal(0, 0), invocation_id(1), if_then(0, 0)],
        child_bodies: vec![child],
        literals: vec![LiteralValue::U32(1)],
    });
    assert_eq!(reuse_errors(&desc), vec![(1, vec![], vec![0])]);
}

/// Two sibling bodies assigning the same id. This is the exact shape
/// `loop_unroll` used to emit: the `GlobalInvocationId` in one branch
/// and the `Literal(1)` in the other both claimed `%1`, and the PTX
/// emitter folded the store address to a constant.
#[test]
fn sibling_bodies_sharing_an_id_is_rejected() {
    let first = KernelBody {
        ops: vec![literal(0, 1), store(1, 1)],
        child_bodies: vec![],
        literals: vec![LiteralValue::U32(1)],
    };
    let second = KernelBody {
        ops: vec![invocation_id(1), store(1, 1)],
        child_bodies: vec![],
        literals: vec![],
    };
    let desc = descriptor(KernelBody {
        ops: vec![literal(0, 0), if_then(0, 0), if_then(0, 1)],
        child_bodies: vec![first, second],
        literals: vec![LiteralValue::U32(1)],
    });
    assert_eq!(reuse_errors(&desc), vec![(1, vec![0], vec![1])]);
}

/// Reuse across a grandchild, not just one level down. The walk has to
/// carry the owner map through the whole tree.
#[test]
fn grandchild_sharing_a_top_level_id_is_rejected() {
    let grandchild = KernelBody {
        ops: vec![literal(0, 1), store(1, 1)],
        child_bodies: vec![],
        literals: vec![LiteralValue::U32(1)],
    };
    let child = KernelBody {
        ops: vec![literal(0, 5), if_then(5, 0)],
        child_bodies: vec![grandchild],
        literals: vec![LiteralValue::U32(1)],
    };
    let desc = descriptor(KernelBody {
        ops: vec![literal(0, 0), invocation_id(1), if_then(0, 0)],
        child_bodies: vec![child],
        literals: vec![LiteralValue::U32(1)],
    });
    assert_eq!(reuse_errors(&desc), vec![(1, vec![], vec![0, 0])]);
}

/// Every reuse is reported, not just the first, so one verify run tells
/// you the full extent of a rewrite's damage.
#[test]
fn every_reused_id_is_reported() {
    let child = KernelBody {
        ops: vec![literal(0, 1), literal(0, 2), store(1, 2)],
        child_bodies: vec![],
        literals: vec![LiteralValue::U32(1)],
    };
    let desc = descriptor(KernelBody {
        ops: vec![
            literal(0, 0),
            invocation_id(1),
            literal(0, 2),
            if_then(0, 0),
            store(1, 2),
        ],
        child_bodies: vec![child],
        literals: vec![LiteralValue::U32(1)],
    });
    assert_eq!(
        reuse_errors(&desc),
        vec![(1, vec![], vec![0]), (2, vec![], vec![0])]
    );
}

/// A duplicate inside ONE body is already `DuplicateResultId`. Invariant
/// 6 must not double-report it, or a single defect reads as two.
#[test]
fn same_body_duplicate_is_not_also_reported_as_cross_body_reuse() {
    let desc = descriptor(KernelBody {
        ops: vec![literal(0, 0), literal(0, 0), store(0, 0)],
        child_bodies: vec![],
        literals: vec![LiteralValue::U32(1)],
    });
    let errors = verify(&desc).expect_err("duplicate id must fail verification");
    assert!(errors
        .iter()
        .any(|error| error.kind == VerifyErrorKind::DuplicateResultId(0)));
    assert_eq!(reuse_errors(&desc), Vec::new());
}
