//! Test: descriptor control.
use super::*;

/// Build `literal -> Cast(target)` so the emitted module's 64-bit backing
/// `Compose(vec2<u32>)` can be inspected for the high-word extension policy.
fn cast_widen_desc(literal: LiteralValue, target: DataType) -> KernelDescriptor {
    KernelDescriptor {
        id: "cast_widen".into(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: Dispatch::new(1, 1, 1),
        body: KernelBody {
            ops: vec![
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![0],
                    result: Some(0),
                },
                KernelOp {
                    kind: KernelOpKind::Cast { target },
                    operands: vec![0],
                    result: Some(1),
                },
            ],
            child_bodies: vec![],
            literals: vec![literal],
        },
    }
}

/// The single 2-component `Compose` (the vec2<u32> 64-bit backing) high-word
/// expression, so a test can assert zero- vs sign-extension.
fn high_word_of_only_vec2_compose(module: &naga::Module) -> naga::Expression {
    let entry = module.entry_points.first().expect("entry point");
    let arena = &entry.function.expressions;
    let composes: Vec<_> = arena
        .iter()
        .filter_map(|(_, e)| match e {
            naga::Expression::Compose { components, .. } if components.len() == 2 => {
                Some(components.clone())
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        composes.len(),
        1,
        "a single scalar->64-bit cast must emit exactly one vec2<u32> Compose"
    );
    arena[composes[0][1]].clone()
}

#[test]
fn signed_i32_arithmetic_shift_right_emits_valid_wgsl() {
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    // `i32 >> n` is an ARITHMETIC shift (sign-preserving). validate's IR allows
    // it. naga makes `>>` arithmetic when the LEFT operand is Sint, but WGSL
    // requires the shift AMOUNT (right operand) to be u32. The probe: emit a
    // real signed shift and run naga's validator — if the emitter coerced the
    // shift amount to i32 (to match the signed left), the module is invalid WGSL.
    let desc = KernelDescriptor {
        id: "signed_shr".into(),
        bindings: BindingLayout {
            slots: vec![vyre_lower::BindingSlot {
                slot: 0,
                element_type: DataType::I32,
                element_count: Some(1),
                memory_class: MemoryClass::Global,
                visibility: BindingVisibility::ReadWrite,
                name: "out".into(),
            }],
        },
        dispatch: Dispatch::new(1, 1, 1),
        body: KernelBody {
            ops: vec![
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![0],
                    result: Some(0),
                },
                KernelOp {
                    kind: KernelOpKind::Cast {
                        target: DataType::I32,
                    },
                    operands: vec![0],
                    result: Some(1),
                },
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![1],
                    result: Some(2),
                },
                KernelOp {
                    kind: KernelOpKind::BinOpKind(BinOp::Shr),
                    operands: vec![1, 2],
                    result: Some(3),
                },
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![2],
                    result: Some(4),
                },
                KernelOp {
                    kind: KernelOpKind::StoreGlobal,
                    operands: vec![0, 4, 3],
                    result: None,
                },
            ],
            child_bodies: vec![],
            literals: vec![
                LiteralValue::U32(0x8000_0000),
                LiteralValue::U32(1),
                LiteralValue::U32(0),
            ],
        },
    };
    let module = emit(&desc).expect("i32 >> 1 must emit");
    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    validator
        .validate(&module)
        .expect("signed arithmetic shift-right must produce valid WGSL (shift amount stays u32)");

    // Pin the semantics, not just validity: the shift must be ARITHMETIC — its
    // value operand stays Sint (that is what makes naga emit a sign-preserving
    // `>>`) while the amount is Uint (u32).
    let entry = module.entry_points.first().expect("entry point");
    let arena = &entry.function.expressions;
    let shift = arena
        .iter()
        .find_map(|(_, e)| match e {
            naga::Expression::Binary {
                op: naga::BinaryOperator::ShiftRight,
                left,
                right,
            } => Some((*left, *right)),
            _ => None,
        })
        .expect("a ShiftRight must be emitted");
    let kind_of = |h: naga::Handle<naga::Expression>| match &arena[h] {
        naga::Expression::As { kind, .. } => Some(*kind),
        naga::Expression::Literal(naga::Literal::U32(_)) => Some(naga::ScalarKind::Uint),
        naga::Expression::Literal(naga::Literal::I32(_)) => Some(naga::ScalarKind::Sint),
        _ => None,
    };
    assert_eq!(
        kind_of(shift.0),
        Some(naga::ScalarKind::Sint),
        "the shifted value must stay Sint so `>>` is arithmetic (sign-preserving)"
    );
    assert_eq!(
        kind_of(shift.1),
        Some(naga::ScalarKind::Uint),
        "the shift amount must be u32, never coerced to the value's signedness"
    );
}

#[test]
fn cast_i32_to_i64_sign_extends_high_word() {
    // A signed 32-bit source widened to a 64-bit integer must SIGN-extend:
    // the high word replicates the source's sign bit so a negative value keeps
    // its two's-complement pattern (matching the PTX `cvt.s64.s32` path and
    // Rust `i32 as i64`). Before the fix the high word was an unconditional
    // literal 0, silently turning every negative `i32 -> i64` into a large
    // positive value (Law 10 miscompile).
    let module = emit(&cast_widen_desc(LiteralValue::I32(-1), DataType::I64))
        .expect("i32 -> i64 cast must emit");
    let entry = module.entry_points.first().expect("entry point");
    let arena = &entry.function.expressions;
    let high = high_word_of_only_vec2_compose(&module);
    let naga::Expression::Binary {
        op: naga::BinaryOperator::Multiply,
        left,
        right,
    } = &high
    else {
        panic!("i32 -> i64 high word must be a sign replicate (Multiply); got {high:?}");
    };
    assert!(
        matches!(
            &arena[*right],
            naga::Expression::Literal(naga::Literal::U32(0xFFFF_FFFF))
        ),
        "sign replicate must multiply the sign bit by 0xFFFFFFFF"
    );
    let naga::Expression::Binary {
        op: naga::BinaryOperator::ShiftRight,
        right: shift_amount,
        ..
    } = &arena[*left]
    else {
        panic!("sign bit must be extracted via a ShiftRight of the low word");
    };
    assert!(
        matches!(
            &arena[*shift_amount],
            naga::Expression::Literal(naga::Literal::U32(31))
        ),
        "sign bit must be the low word shifted right by 31"
    );
}

#[test]
fn cast_u32_to_i64_zero_extends_high_word() {
    // The unsigned twin: a u32 source widened to a 64-bit integer ZERO-extends
    // (the source is non-negative), so the high word stays a literal 0 and the
    // sign-replicate chain must NOT appear.
    let module = emit(&cast_widen_desc(LiteralValue::U32(7), DataType::I64))
        .expect("u32 -> i64 cast must emit");
    let high = high_word_of_only_vec2_compose(&module);
    assert!(
        matches!(high, naga::Expression::Literal(naga::Literal::U32(0))),
        "u32 -> i64 high word must be a literal 0 (zero-extend); got {high:?}"
    );
    let entry = module.entry_points.first().expect("entry point");
    let has_sign_replicate = entry.function.expressions.iter().any(|(_, e)| {
        matches!(
            e,
            naga::Expression::Binary {
                op: naga::BinaryOperator::Multiply,
                ..
            }
        )
    });
    assert!(
        !has_sign_replicate,
        "u32 -> i64 must not emit a sign-replicate Multiply; zero-extend only"
    );
}

#[test]
fn cast_i32_to_vec2_zero_fills_lane_one() {
    // `Vec2U32` is a STRUCTURAL 2-word vector, not a 64-bit integer: lane 1 is
    // always zero-filled (matching the reference `widen_to_words`/`cast_to_vec2`
    // zero-pad), never sign-extended — even from a signed source.
    let module = emit(&cast_widen_desc(LiteralValue::I32(-1), DataType::Vec2U32))
        .expect("i32 -> vec2<u32> cast must emit");
    let high = high_word_of_only_vec2_compose(&module);
    assert!(
        matches!(high, naga::Expression::Literal(naga::Literal::U32(0))),
        "i32 -> Vec2U32 lane 1 must be a literal 0 (structural zero-pad); got {high:?}"
    );
}

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
