//! Test: descriptor control.
use super::*;

#[test]
fn descriptor_async_load_emits_bounded_copy_loop() {
    let desc = async_copy_desc(KernelOpKind::AsyncLoad { tag: "load".into() });
    let module = emit(&desc).expect("descriptor AsyncLoad must lower to a bounded copy loop");
    assert!(
        block_has_loop(&module.entry_points[0].function.body),
        "descriptor AsyncLoad must emit a Naga loop for the synchronous copy fallback"
    );
}

#[test]
fn descriptor_async_store_emits_bounded_copy_loop() {
    let desc = async_copy_desc(KernelOpKind::AsyncStore {
        tag: "store".into(),
    });
    let module = emit(&desc).expect("descriptor AsyncStore must lower to a bounded copy loop");
    assert!(
        block_has_loop(&module.entry_points[0].function.body),
        "descriptor AsyncStore must emit a Naga loop for the synchronous copy fallback"
    );
}

#[test]
fn descriptor_trap_emits_sidecar_atomic_path() {
    let desc = KernelDescriptor {
        id: "trap".into(),
        bindings: BindingLayout {
            slots: vec![trap_sidecar_slot(0)],
        },
        dispatch: Dispatch::new(64, 1, 1),
        body: KernelBody {
            literals: vec![LiteralValue::U32(7)],
            child_bodies: vec![],
            ops: vec![
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![0],
                    result: Some(0),
                },
                KernelOp {
                    kind: KernelOpKind::Trap {
                        tag: "page-fault".into(),
                    },
                    operands: vec![0],
                    result: None,
                },
            ],
        },
    };
    let module = emit(&desc).expect("descriptor Trap must emit sidecar atomics");
    let body = &module.entry_points[0].function.body;
    assert!(
        block_has_atomic(body),
        "trap emission must write the sidecar through atomics"
    );
    assert!(
        body.iter()
            .any(|statement| matches!(statement, Statement::Return { .. })),
        "trap emission must terminate the trapped lane"
    );
}

#[test]
fn descriptor_resume_is_runtime_marker_not_unsupported() {
    let desc = KernelDescriptor {
        id: "resume".into(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: Dispatch::new(1, 1, 1),
        body: KernelBody {
            literals: vec![],
            child_bodies: vec![],
            ops: vec![KernelOp {
                kind: KernelOpKind::Resume { tag: "r".into() },
                operands: vec![],
                result: None,
            }],
        },
    };
    emit(&desc).expect("descriptor Resume is a runtime sequencing marker");
}

#[test]
fn descriptor_wide_literal_opaque_emits_from_payload() {
    let desc = KernelDescriptor {
        id: "opaque-lit".into(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: Dispatch::new(1, 1, 1),
        body: KernelBody {
            literals: vec![],
            child_bodies: vec![],
            ops: vec![KernelOp {
                kind: KernelOpKind::OpaqueExpr(Box::new(vyre_lower::OpaqueExprData {
                    extension_id: 1,
                    extension_kind: "vyre.literal.u64".to_owned(),
                    payload: 42u64.to_le_bytes().to_vec(),
                })),
                operands: vec![],
                result: Some(0),
            }],
        },
    };
    emit(&desc).expect("known opaque wide literal must emit from descriptor payload");
}

#[test]
fn descriptor_structured_for_loop_emits_naga_loop() {
    let desc = KernelDescriptor {
        id: "loop".into(),
        bindings: BindingLayout {
            slots: vec![u32_output_slot(0)],
        },
        dispatch: Dispatch::new(64, 1, 1),
        body: KernelBody {
            literals: vec![LiteralValue::U32(0), LiteralValue::U32(4)],
            ops: vec![
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![0],
                    result: Some(0),
                },
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![1],
                    result: Some(1),
                },
                KernelOp {
                    kind: KernelOpKind::StructuredForLoop {
                        loop_var: "i".into(),
                    },
                    operands: vec![0, 1, 0],
                    result: None,
                },
            ],
            child_bodies: vec![KernelBody {
                literals: vec![],
                ops: vec![
                    KernelOp {
                        kind: KernelOpKind::LoopIndex {
                            loop_var: "i".into(),
                        },
                        operands: vec![],
                        result: Some(2),
                    },
                    KernelOp {
                        kind: KernelOpKind::StoreGlobal,
                        operands: vec![0, 2, 2],
                        result: None,
                    },
                ],
                child_bodies: vec![],
            }],
        },
    };

    let module = emit(&desc).expect("descriptor loop must emit through Naga");
    assert!(
        block_has_loop(&module.entry_points[0].function.body),
        "descriptor StructuredForLoop must lower to a Naga Statement::Loop"
    );
}

#[test]
fn atomic_result_can_feed_later_descriptor_ops() {
    use vyre_foundation::ir::AtomicOp;

    let desc = KernelDescriptor {
        id: "atomic-result".into(),
        bindings: BindingLayout {
            slots: vec![u32_output_slot(0)],
        },
        dispatch: Dispatch::new(64, 1, 1),
        body: KernelBody {
            literals: vec![LiteralValue::U32(0), LiteralValue::U32(1)],
            child_bodies: vec![],
            ops: vec![
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![0],
                    result: Some(0),
                },
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![1],
                    result: Some(1),
                },
                KernelOp {
                    kind: KernelOpKind::Atomic {
                        op: AtomicOp::Add,
                        ordering: MemoryOrdering::SeqCst,
                    },
                    operands: vec![0, 0, 1],
                    result: Some(2),
                },
                KernelOp {
                    kind: KernelOpKind::StoreGlobal,
                    operands: vec![0, 0, 2],
                    result: None,
                },
            ],
        },
    };

    emit(&desc).expect("atomic RMW old value must remain usable by later descriptor ops");
}

/// vyre.literal.u64 must emit as Literal::U64 preserving the full 64-bit
/// value, not narrow to u32 or error. A value above u32::MAX (1u64 << 40 =
/// 0x10000000000) would previously hard-error with "exceeds u32::MAX"; after
/// the fix it emits as Literal::U64(0x10000000000) with type u64_ty.
#[test]
fn opaque_u64_literal_above_u32_max_emits_as_u64() {
    let value: u64 = 1u64 << 40; // 0x10000000000 — above u32::MAX
    let desc = KernelDescriptor {
        id: "opaque-u64-wide".into(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: Dispatch::new(1, 1, 1),
        body: KernelBody {
            literals: vec![],
            child_bodies: vec![],
            ops: vec![KernelOp {
                kind: KernelOpKind::OpaqueExpr(Box::new(vyre_lower::OpaqueExprData {
                    extension_id: 1,
                    extension_kind: "vyre.literal.u64".to_owned(),
                    payload: value.to_le_bytes().to_vec(),
                })),
                operands: vec![],
                result: Some(0),
            }],
        },
    };
    // Before the fix: hard-errors with InvalidDescriptor("u64 literal ... exceeds u32::MAX").
    // After the fix: emits Literal::U64(0x10000000000) successfully.
    let module = emit(&desc)
        .expect("vyre.literal.u64 with value above u32::MAX must emit as Literal::U64, not error");

    // Verify the expression arena contains the full-width u64 literal, not a
    // truncated or type-changed value.
    use naga::{Expression, Literal};
    let entry = &module.entry_points[0];
    let has_u64_literal = entry
        .function
        .expressions
        .iter()
        .any(|(_, expr)| matches!(expr, Expression::Literal(Literal::U64(v)) if *v == value));
    assert!(
        has_u64_literal,
        "vyre.literal.u64 must emit Literal::U64({value}) in the expression arena; \
         got a u32 narrowing or missing literal instead"
    );
}
