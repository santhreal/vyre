use super::*;
use crate::descriptor_builder::{effect, lit, op};
use crate::{
    BindingLayout, Dispatch, KernelBody, KernelDescriptor, KernelOp, KernelOpKind, LiteralValue,
};
use vyre_foundation::ir::BinOp;

fn empty_desc(ops: Vec<KernelOp>, literals: Vec<LiteralValue>) -> KernelDescriptor {
    KernelDescriptor {
        id: "k".into(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: Dispatch::new(1, 1, 1),
        body: KernelBody {
            ops,
            child_bodies: vec![],
            literals,
        },
    }
}

#[test]
fn empty_kernel_verifies() {
    assert!(verify(&empty_desc(vec![], vec![])).is_ok());
}

#[test]
fn well_formed_kernel_verifies() {
    let desc = empty_desc(
        vec![
            lit(0, 0),
            lit(1, 1),
            op(KernelOpKind::BinOpKind(BinOp::Add), [0, 1], 2),
        ],
        vec![LiteralValue::U32(3), LiteralValue::U32(4)],
    );
    assert_eq!(verify(&desc), Ok(()));
}

#[test]
fn duplicate_result_id_detected() {
    let desc = empty_desc(
        vec![
            lit(0, 0),
            lit(0, 0), // dup
        ],
        vec![LiteralValue::U32(1)],
    );
    let r = verify(&desc);
    assert!(r.is_err());
    let errs = r.unwrap_err();
    assert!(errs
        .iter()
        .any(|e| matches!(e.kind, VerifyErrorKind::DuplicateResultId(0))));
}

#[test]
fn dangling_result_ref_detected() {
    let desc = empty_desc(
        vec![
            lit(0, 0),
            effect(KernelOpKind::BinOpKind(BinOp::Add), [0, 99]),
        ],
        vec![LiteralValue::U32(1)],
    );
    let r = verify(&desc);
    let errs = r.unwrap_err();
    assert!(errs.iter().any(|e| matches!(
        e.kind,
        VerifyErrorKind::DanglingResultRef { ref_id: 99, .. }
    )));
}

#[test]
fn literal_pool_out_of_range_detected() {
    let desc = empty_desc(
        vec![effect(KernelOpKind::Literal, [5])],
        vec![LiteralValue::U32(1)],
    );
    let r = verify(&desc);
    let errs = r.unwrap_err();
    assert!(errs.iter().any(|e| matches!(
        e.kind,
        VerifyErrorKind::LiteralPoolOutOfRange {
            pool_idx: 5,
            pool_size: 1,
            ..
        }
    )));
}

#[test]
fn child_body_index_out_of_range_detected() {
    let desc = KernelDescriptor {
        id: "k".into(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: Dispatch::new(1, 1, 1),
        body: KernelBody {
            ops: vec![lit(0, 0), effect(KernelOpKind::StructuredIfThen, [0, 7])],
            child_bodies: vec![],
            literals: vec![LiteralValue::U32(1)],
        },
    };
    let r = verify(&desc);
    let errs = r.unwrap_err();
    assert!(errs.iter().any(|e| matches!(
        e.kind,
        VerifyErrorKind::ChildBodyIndexOutOfRange {
            body_idx: 7,
            child_count: 0,
            ..
        }
    )));
}

#[test]
fn literal_op_with_no_operands_detected() {
    let desc = empty_desc(
        vec![op(KernelOpKind::Literal, [], 0)],
        vec![LiteralValue::U32(1)],
    );
    let r = verify(&desc);
    let errs = r.unwrap_err();
    assert!(errs
        .iter()
        .any(|e| matches!(e.kind, VerifyErrorKind::LiteralOpMissingPoolOperand)));
}

#[test]
fn operand_count_too_short_detected() {
    let desc = empty_desc(
        vec![lit(0, 0), effect(KernelOpKind::BinOpKind(BinOp::Add), [0])],
        vec![LiteralValue::U32(1)],
    );
    let r = verify(&desc);
    let errs = r.unwrap_err();
    assert!(errs.iter().any(|e| matches!(
        e.kind,
        VerifyErrorKind::OperandCountTooShort {
            expected_min: 2,
            got: 1
        }
    )));
}

#[test]
fn errors_are_collected_not_short_circuited() {
    // 3 distinct violations in one body.
    let desc = empty_desc(
        vec![
            lit(99, 0), // pool oor
            lit(0, 0),  // dup
            effect(KernelOpKind::BinOpKind(BinOp::Add), [100, 200]),
        ],
        vec![LiteralValue::U32(1)],
    );
    let r = verify(&desc);
    let errs = r.unwrap_err();
    assert!(errs.len() >= 3);
}

#[test]
fn child_body_violations_recurse() {
    let desc = KernelDescriptor {
        id: "k".into(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: Dispatch::new(1, 1, 1),
        body: KernelBody {
            ops: vec![],
            child_bodies: vec![KernelBody {
                ops: vec![lit(99, 0)],
                child_bodies: vec![],
                literals: vec![LiteralValue::U32(1)],
            }],
            literals: vec![],
        },
    };
    let r = verify(&desc);
    let errs = r.unwrap_err();
    assert!(errs.iter().any(|e| e.body_path == vec![0]));
}

#[test]
fn child_body_may_capture_parent_result_available_before_control_op() {
    let child = KernelBody {
        ops: vec![op(KernelOpKind::BinOpKind(BinOp::Add), [0, 0], 1)],
        child_bodies: vec![],
        literals: vec![],
    };
    let desc = KernelDescriptor {
        id: "captures".into(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: Dispatch::new(1, 1, 1),
        body: KernelBody {
            ops: vec![lit(0, 0), effect(KernelOpKind::StructuredBlock, [0])],
            child_bodies: vec![child],
            literals: vec![LiteralValue::U32(7)],
        },
    };

    assert_eq!(verify(&desc), Ok(()));
}

#[test]
fn child_body_cannot_capture_parent_result_declared_after_control_op() {
    let child = KernelBody {
        ops: vec![op(KernelOpKind::BinOpKind(BinOp::Add), [1, 1], 2)],
        child_bodies: vec![],
        literals: vec![],
    };
    let desc = KernelDescriptor {
        id: "future_capture".into(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: Dispatch::new(1, 1, 1),
        body: KernelBody {
            ops: vec![
                lit(0, 0),
                effect(KernelOpKind::StructuredBlock, [0]),
                lit(1, 1),
            ],
            child_bodies: vec![child],
            literals: vec![LiteralValue::U32(7), LiteralValue::U32(9)],
        },
    };

    let errors = verify(&desc).expect_err("future child capture must fail");
    assert!(errors.iter().any(|error| {
        error.body_path == vec![0]
            && matches!(
                error.kind,
                VerifyErrorKind::DanglingResultRef { ref_id: 1, .. }
            )
    }));
}

#[test]
fn parent_body_may_read_result_assigned_by_completed_child_body() {
    let child = KernelBody {
        ops: vec![op(KernelOpKind::BinOpKind(BinOp::Add), [0, 0], 1)],
        child_bodies: vec![],
        literals: vec![],
    };
    let desc = KernelDescriptor {
        id: "loop_carried".into(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: Dispatch::new(1, 1, 1),
        body: KernelBody {
            ops: vec![
                lit(0, 0),
                effect(KernelOpKind::StructuredBlock, [0]),
                op(KernelOpKind::BinOpKind(BinOp::Mul), [1, 0], 2),
            ],
            child_bodies: vec![child],
            literals: vec![LiteralValue::U32(7)],
        },
    };

    assert_eq!(verify(&desc), Ok(()));
}

#[test]
fn dispatch_zero_x_dim_detected() {
    let desc = KernelDescriptor {
        id: "k".into(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: Dispatch::new(0, 1, 1),
        body: KernelBody {
            ops: vec![],
            child_bodies: vec![],
            literals: vec![],
        },
    };
    let r = verify(&desc);
    let errs = r.unwrap_err();
    assert!(errs
        .iter()
        .any(|e| matches!(e.kind, VerifyErrorKind::DispatchZeroDim { axis: 0 })));
}

#[test]
fn dispatch_zero_z_dim_detected() {
    let desc = KernelDescriptor {
        id: "k".into(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: Dispatch::new(64, 1, 0),
        body: KernelBody {
            ops: vec![],
            child_bodies: vec![],
            literals: vec![],
        },
    };
    let r = verify(&desc);
    let errs = r.unwrap_err();
    assert!(errs
        .iter()
        .any(|e| matches!(e.kind, VerifyErrorKind::DispatchZeroDim { axis: 2 })));
}

#[test]
fn duplicate_binding_slot_detected() {
    use crate::{BindingSlot, BindingVisibility, MemoryClass};
    use vyre_foundation::ir::DataType;
    let dup = BindingSlot {
        slot: 5,
        element_type: DataType::U32,
        element_count: None,
        memory_class: MemoryClass::Global,
        visibility: BindingVisibility::ReadWrite,
        name: "a".into(),
    };
    let mut second = dup.clone();
    second.name = "b".into();
    let desc = KernelDescriptor {
        id: "k".into(),
        bindings: BindingLayout {
            slots: vec![dup, second],
        },
        dispatch: Dispatch::new(64, 1, 1),
        body: KernelBody {
            ops: vec![],
            child_bodies: vec![],
            literals: vec![],
        },
    };
    let r = verify(&desc);
    let errs = r.unwrap_err();
    assert!(errs
        .iter()
        .any(|e| matches!(e.kind, VerifyErrorKind::DuplicateBindingSlotId { slot: 5 })));
}

#[test]
fn dispatch_normal_no_error() {
    let desc = KernelDescriptor {
        id: "k".into(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: Dispatch::new(64, 1, 1),
        body: KernelBody {
            ops: vec![],
            child_bodies: vec![],
            literals: vec![],
        },
    };
    assert_eq!(verify(&desc), Ok(()));
}

#[test]
fn host_binding_in_workgroup_range_is_rejected() {
    use crate::{BindingSlot, BindingVisibility, MemoryClass};
    use vyre_foundation::ir::DataType;
    let bad = BindingSlot {
        slot: crate::lower::WORKGROUP_SLOT_BASE + 7,
        element_type: DataType::U32,
        element_count: None,
        memory_class: MemoryClass::Global,
        visibility: BindingVisibility::ReadWrite,
        name: "host_in_high_range".into(),
    };
    let desc = KernelDescriptor {
        id: "k".into(),
        bindings: BindingLayout { slots: vec![bad] },
        dispatch: Dispatch::new(64, 1, 1),
        body: KernelBody {
            ops: vec![],
            child_bodies: vec![],
            literals: vec![],
        },
    };
    let errs = verify(&desc).unwrap_err();
    assert!(errs
        .iter()
        .any(|e| matches!(e.kind, VerifyErrorKind::HostBindingInWorkgroupRange { .. })));
}

#[test]
fn workgroup_binding_in_host_range_is_rejected() {
    use crate::{BindingSlot, BindingVisibility, MemoryClass};
    use vyre_foundation::ir::DataType;
    let bad = BindingSlot {
        slot: 5,
        element_type: DataType::U32,
        element_count: Some(64),
        memory_class: MemoryClass::Shared,
        visibility: BindingVisibility::ReadWrite,
        name: "shared_in_low_range".into(),
    };
    let desc = KernelDescriptor {
        id: "k".into(),
        bindings: BindingLayout { slots: vec![bad] },
        dispatch: Dispatch::new(64, 1, 1),
        body: KernelBody {
            ops: vec![],
            child_bodies: vec![],
            literals: vec![],
        },
    };
    let errs = verify(&desc).unwrap_err();
    assert!(errs
        .iter()
        .any(|e| matches!(e.kind, VerifyErrorKind::WorkgroupBindingInHostRange { .. })));
}
