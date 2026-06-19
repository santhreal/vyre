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

/// Load two `U64` elements (vec2<u32> backing) and apply `binop`, storing into
/// a `U64` out buffer. Used to prove the 64-bit carry gate fires for arithmetic
/// and admits the carry-free bitwise ops.
fn u64_binop_desc(binop: BinOp) -> KernelDescriptor {
    KernelDescriptor {
        id: "u64_binop".into(),
        bindings: BindingLayout {
            slots: vec![
                BindingSlot {
                    slot: 0,
                    element_type: DataType::U64,
                    element_count: Some(4),
                    memory_class: MemoryClass::Global,
                    visibility: BindingVisibility::ReadOnly,
                    name: "src".into(),
                },
                BindingSlot {
                    slot: 1,
                    element_type: DataType::U64,
                    element_count: Some(4),
                    memory_class: MemoryClass::Global,
                    visibility: BindingVisibility::ReadWrite,
                    name: "out".into(),
                },
            ],
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
                    kind: KernelOpKind::LoadGlobal,
                    operands: vec![0, 0],
                    result: Some(1),
                },
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![1],
                    result: Some(2),
                },
                KernelOp {
                    kind: KernelOpKind::LoadGlobal,
                    operands: vec![0, 2],
                    result: Some(3),
                },
                KernelOp {
                    kind: KernelOpKind::BinOpKind(binop),
                    operands: vec![1, 3],
                    result: Some(4),
                },
                KernelOp {
                    kind: KernelOpKind::StoreGlobal,
                    operands: vec![1, 0, 4],
                    result: None,
                },
            ],
            child_bodies: vec![],
            literals: vec![LiteralValue::U32(0), LiteralValue::U32(1)],
        },
    }
}

#[test]
fn u64_carry_sensitive_binops_fail_closed_not_silently_componentwise() {
    // A vec2<u32>-backed 64-bit value has NO carry between its low and high
    // word, so a componentwise vec2 Add/Sub/Mul/Shift/Compare is silently WRONG
    // arithmetic — and crucially it would VALIDATE through naga (vec2 ops are
    // legal WGSL), so only an explicit fail-closed gate catches the bug. Pin
    // that the gate fires with its real diagnostic for every carry-sensitive op.
    for binop in [
        BinOp::Add,
        BinOp::Sub,
        BinOp::Mul,
        BinOp::Shl,
        BinOp::Shr,
        BinOp::Lt,
        BinOp::Gt,
        BinOp::Eq,
    ] {
        let err = emit(&u64_binop_desc(binop))
            .expect_err(&format!("{binop:?} on vec2<u32>-backed u64 must fail closed"));
        let msg = format!("{err:?}");
        assert!(
            msg.contains("is not lowered") && msg.contains("carry"),
            "{binop:?}: fail-closed error must name the missing carry lowering, got: {msg}"
        );
    }
}

#[test]
fn u64_bitwise_binops_emit_valid_componentwise_wgsl() {
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    // BitAnd/BitOr/BitXor carry NO information between words, so componentwise
    // vec2<u32> is the correct lowering — these must emit and validate, and the
    // stored result keeps the vec2<u32> 64-bit backing.
    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    for binop in [BinOp::BitAnd, BinOp::BitOr, BinOp::BitXor] {
        let module =
            emit(&u64_binop_desc(binop)).unwrap_or_else(|e| panic!("{binop:?}: emit failed: {e}"));
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("{binop:?} on u64: INVALID WGSL: {e:?}"));
        // Validation passing already proves type-correctness; additionally pin
        // that the 64-bit value was NOT collapsed to one word — the out buffer
        // must remain array<vec2<u32>> and a Binary of this operator must exist.
        let out_is_vec2 = module.global_variables.iter().any(|(_, g)| {
            if let naga::TypeInner::Array { base, .. } = module.types[g.ty].inner {
                matches!(
                    module.types[base].inner,
                    naga::TypeInner::Vector {
                        size: naga::VectorSize::Bi,
                        scalar: naga::Scalar {
                            kind: naga::ScalarKind::Uint,
                            ..
                        },
                    }
                )
            } else {
                false
            }
        });
        assert!(
            out_is_vec2,
            "{binop:?}: u64 buffer must stay backed by array<vec2<u32>>"
        );
        let entry = module.entry_points.first().expect("entry point");
        let expected = match binop {
            BinOp::BitAnd => naga::BinaryOperator::And,
            BinOp::BitOr => naga::BinaryOperator::InclusiveOr,
            BinOp::BitXor => naga::BinaryOperator::ExclusiveOr,
            _ => unreachable!(),
        };
        let has_op = entry.function.expressions.iter().any(|(_, e)| {
            matches!(e, naga::Expression::Binary { op, .. } if *op == expected)
        });
        assert!(
            has_op,
            "{binop:?}: expected a Binary {expected:?} over the vec2<u32> backing"
        );
    }
}

/// Load `x` and a divisor `y` from a u32 buffer, apply `binop` (Div/Mod), store
/// to a u32 out. Both operands are runtime loads so no constant fold short-
/// circuits the divide.
fn u32_div_desc(binop: BinOp) -> KernelDescriptor {
    KernelDescriptor {
        id: "u32_div".into(),
        bindings: BindingLayout {
            slots: vec![
                BindingSlot {
                    slot: 0,
                    element_type: DataType::U32,
                    element_count: Some(4),
                    memory_class: MemoryClass::Global,
                    visibility: BindingVisibility::ReadOnly,
                    name: "src".into(),
                },
                BindingSlot {
                    slot: 1,
                    element_type: DataType::U32,
                    element_count: Some(4),
                    memory_class: MemoryClass::Global,
                    visibility: BindingVisibility::ReadWrite,
                    name: "out".into(),
                },
            ],
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
                    kind: KernelOpKind::LoadGlobal,
                    operands: vec![0, 0],
                    result: Some(1),
                },
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![1],
                    result: Some(2),
                },
                KernelOp {
                    kind: KernelOpKind::LoadGlobal,
                    operands: vec![0, 2],
                    result: Some(3),
                },
                KernelOp {
                    kind: KernelOpKind::BinOpKind(binop),
                    operands: vec![1, 3],
                    result: Some(4),
                },
                KernelOp {
                    kind: KernelOpKind::StoreGlobal,
                    operands: vec![1, 0, 4],
                    result: None,
                },
            ],
            child_bodies: vec![],
            literals: vec![LiteralValue::U32(0), LiteralValue::U32(1)],
        },
    }
}

/// Load a U64 (vec2<u32> backing), apply `unop`, store to a U64 out.
fn u64_unop_desc(unop: UnOp) -> KernelDescriptor {
    KernelDescriptor {
        id: "u64_unop".into(),
        bindings: BindingLayout {
            slots: vec![
                BindingSlot {
                    slot: 0,
                    element_type: DataType::U64,
                    element_count: Some(4),
                    memory_class: MemoryClass::Global,
                    visibility: BindingVisibility::ReadOnly,
                    name: "src".into(),
                },
                BindingSlot {
                    slot: 1,
                    element_type: DataType::U64,
                    element_count: Some(4),
                    memory_class: MemoryClass::Global,
                    visibility: BindingVisibility::ReadWrite,
                    name: "out".into(),
                },
            ],
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
                    kind: KernelOpKind::LoadGlobal,
                    operands: vec![0, 0],
                    result: Some(1),
                },
                KernelOp {
                    kind: KernelOpKind::UnOpKind(unop),
                    operands: vec![1],
                    result: Some(2),
                },
                KernelOp {
                    kind: KernelOpKind::StoreGlobal,
                    operands: vec![1, 0, 2],
                    result: None,
                },
            ],
            child_bodies: vec![],
            literals: vec![LiteralValue::U32(0)],
        },
    }
}

#[test]
fn u64_cross_word_unary_ops_fail_closed_not_silently_per_word() {
    // popcount/clz/ctz/reverse_bits/negate on a vec2<u32>-backed 64-bit value
    // would run PER-WORD on the GPU (a valid-but-wrong naga Math/Unary), so the
    // 64-bit result silently diverges from the reference's true 64-bit count.
    // The gate must fail closed with its real diagnostic for each.
    for unop in [
        UnOp::Popcount,
        UnOp::Clz,
        UnOp::Ctz,
        UnOp::ReverseBits,
        UnOp::Negate,
    ] {
        let label = format!("{unop:?}");
        let err = emit(&u64_unop_desc(unop))
            .expect_err(&format!("{label} on vec2<u32>-backed u64 must fail closed"));
        let msg = format!("{err:?}");
        assert!(
            msg.contains("is not lowered") && msg.contains("per-word"),
            "{label}: fail-closed error must name the per-word hazard, got: {msg}"
        );
    }
}

#[test]
fn u64_bitwise_not_emits_valid_componentwise_wgsl() {
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    // ~x on a 64-bit value IS correct componentwise (flip every bit of both
    // words), so BitNot is the one unary the gate admits: it must emit, validate,
    // and keep the vec2<u32> backing.
    let module = emit(&u64_unop_desc(UnOp::BitNot)).expect("u64 BitNot must emit");
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|e| panic!("u64 BitNot: INVALID WGSL: {e:?}"));
    let entry = module.entry_points.first().expect("entry point");
    let has_bitwise_not = entry.function.expressions.iter().any(|(_, e)| {
        matches!(
            e,
            naga::Expression::Unary {
                op: naga::UnaryOperator::BitwiseNot,
                ..
            }
        )
    });
    assert!(
        has_bitwise_not,
        "u64 BitNot must emit a componentwise BitwiseNot over the vec2<u32> backing"
    );
}

/// Shift `x` (slot 0 idx 0) left/right by a runtime amount (slot 0 idx 1),
/// store to a u32 out. The amount is a load, so it is NOT a known constant.
fn u32_variable_shift_desc(binop: BinOp) -> KernelDescriptor {
    let mut desc = u32_div_desc(binop);
    desc.id = "u32_var_shift".into();
    desc
}

/// Shift `x` (slot 0 idx 0) by a constant in-range `amount` literal.
fn u32_const_shift_desc(binop: BinOp, amount: u32) -> KernelDescriptor {
    KernelDescriptor {
        id: "u32_const_shift".into(),
        bindings: BindingLayout {
            slots: vec![
                BindingSlot {
                    slot: 0,
                    element_type: DataType::U32,
                    element_count: Some(4),
                    memory_class: MemoryClass::Global,
                    visibility: BindingVisibility::ReadOnly,
                    name: "src".into(),
                },
                BindingSlot {
                    slot: 1,
                    element_type: DataType::U32,
                    element_count: Some(4),
                    memory_class: MemoryClass::Global,
                    visibility: BindingVisibility::ReadWrite,
                    name: "out".into(),
                },
            ],
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
                    kind: KernelOpKind::LoadGlobal,
                    operands: vec![0, 0],
                    result: Some(1),
                },
                KernelOp {
                    kind: KernelOpKind::Literal,
                    operands: vec![1],
                    result: Some(2),
                },
                KernelOp {
                    kind: KernelOpKind::BinOpKind(binop),
                    operands: vec![1, 2],
                    result: Some(3),
                },
                KernelOp {
                    kind: KernelOpKind::StoreGlobal,
                    operands: vec![1, 0, 3],
                    result: None,
                },
            ],
            child_bodies: vec![],
            literals: vec![LiteralValue::U32(0), LiteralValue::U32(amount)],
        },
    }
}

#[test]
fn variable_shift_amount_is_masked_to_bit_width() {
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    // A runtime shift amount must be masked to `& 31` so the wgpu/spirv/metal
    // path matches PTX and the reference oracle for amounts >= 32 (bare naga
    // shift leaves them undefined). Prove the mask is emitted and validates.
    for binop in [BinOp::Shl, BinOp::Shr] {
        let module = emit(&u32_variable_shift_desc(binop))
            .unwrap_or_else(|e| panic!("{binop:?}: emit failed: {e}"));
        Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{binop:?} variable shift: INVALID WGSL: {e:?}"));
        let entry = module.entry_points.first().expect("entry point");
        let arena = &entry.function.expressions;
        let expected_shift = if matches!(binop, BinOp::Shl) {
            naga::BinaryOperator::ShiftLeft
        } else {
            naga::BinaryOperator::ShiftRight
        };
        // The shift's amount operand must be `something & 31`.
        let shift_amount_is_masked = arena.iter().any(|(_, e)| {
            if let naga::Expression::Binary { op, right, .. } = e {
                if *op == expected_shift {
                    return matches!(
                        arena.try_get(*right),
                        Ok(naga::Expression::Binary {
                            op: naga::BinaryOperator::And,
                            right: mask,
                            ..
                        }) if matches!(
                            arena.try_get(*mask),
                            Ok(naga::Expression::Literal(naga::Literal::U32(31)))
                        )
                    );
                }
            }
            false
        });
        assert!(
            shift_amount_is_masked,
            "{binop:?}: variable shift amount must be masked with `& 31`"
        );
    }
}

#[test]
fn constant_in_range_shift_amount_skips_the_mask() {
    // A known in-range constant shift amount (`x >> 16`) must NOT grow an `& 31`
    // mask — it would fold to itself, so the hot path stays a bare shift.
    let module = emit(&u32_const_shift_desc(BinOp::Shr, 16)).expect("const shift emits");
    let entry = module.entry_points.first().expect("entry point");
    let arena = &entry.function.expressions;
    let shift_amount_is_bare_literal = arena.iter().any(|(_, e)| {
        if let naga::Expression::Binary {
            op: naga::BinaryOperator::ShiftRight,
            right,
            ..
        } = e
        {
            return matches!(
                arena.try_get(*right),
                Ok(naga::Expression::Literal(naga::Literal::U32(16)))
            );
        }
        false
    });
    assert!(
        shift_amount_is_bare_literal,
        "in-range constant shift amount must stay a bare literal (no `& 31`)"
    );
}

#[test]
fn unsigned_div_by_zero_is_guarded_to_oracle_max() {
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    // The wgpu/naga backend must produce the vyre-reference oracle contract
    // (u32 x/0 -> u32::MAX), not naga's bare divisor-override-to-1 result (x/0
    // -> x). Prove the guard is wired: a Select gated on `divisor == 0` whose
    // accept arm is the u32::MAX sentinel, plus the module validates.
    let module = emit(&u32_div_desc(BinOp::Div)).expect("u32 Div must emit");
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|e| panic!("guarded u32 Div: INVALID WGSL: {e:?}"));
    let entry = module.entry_points.first().expect("entry point");
    let arena = &entry.function.expressions;
    let has_max_sentinel = arena.iter().any(|(_, e)| {
        matches!(e, naga::Expression::Literal(naga::Literal::U32(v)) if *v == u32::MAX)
    });
    assert!(
        has_max_sentinel,
        "Div-by-zero guard must materialize the u32::MAX oracle sentinel"
    );
    let select_over_zero_check = arena.iter().any(|(_, e)| {
        if let naga::Expression::Select { condition, accept, .. } = e {
            let cond_is_eq_zero = matches!(
                arena.try_get(*condition),
                Ok(naga::Expression::Binary {
                    op: naga::BinaryOperator::Equal,
                    ..
                })
            );
            let accept_is_max = matches!(
                arena.try_get(*accept),
                Ok(naga::Expression::Literal(naga::Literal::U32(v))) if *v == u32::MAX
            );
            cond_is_eq_zero && accept_is_max
        } else {
            false
        }
    });
    assert!(
        select_over_zero_check,
        "Div-by-zero must be a Select(divisor == 0 ? u32::MAX : x/y)"
    );
}

#[test]
fn unsigned_mod_by_zero_is_guarded_and_valid() {
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    // u32 x % 0 -> 0 (oracle contract). The guard wraps the Modulo in a Select
    // gated on `divisor == 0`; module must validate and contain both.
    let module = emit(&u32_div_desc(BinOp::Mod)).expect("u32 Mod must emit");
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|e| panic!("guarded u32 Mod: INVALID WGSL: {e:?}"));
    let entry = module.entry_points.first().expect("entry point");
    let arena = &entry.function.expressions;
    let has_modulo = arena.iter().any(|(_, e)| {
        matches!(
            e,
            naga::Expression::Binary {
                op: naga::BinaryOperator::Modulo,
                ..
            }
        )
    });
    let has_guard_select = arena.iter().any(|(_, e)| {
        if let naga::Expression::Select { condition, .. } = e {
            matches!(
                arena.try_get(*condition),
                Ok(naga::Expression::Binary {
                    op: naga::BinaryOperator::Equal,
                    ..
                })
            )
        } else {
            false
        }
    });
    assert!(has_modulo, "u32 Mod must still emit a Modulo op");
    assert!(
        has_guard_select,
        "u32 Mod-by-zero must be guarded by a Select(divisor == 0 ? 0 : x%y)"
    );
}

#[test]
fn comparisons_on_signed_buffer_load_emit_valid_wgsl() {
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    // Comparisons of an i32 buffer load against a u32 literal must also resolve
    // (naga rejects `Less(i32, u32)`); the result is bool, stored to a u32 out.
    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    for binop in [
        BinOp::Lt,
        BinOp::Gt,
        BinOp::Le,
        BinOp::Ge,
        BinOp::Eq,
        BinOp::Ne,
    ] {
        let desc = KernelDescriptor {
            id: "signed_cmp".into(),
            bindings: BindingLayout {
                slots: vec![
                    BindingSlot {
                        slot: 0,
                        element_type: DataType::I32,
                        element_count: Some(4),
                        memory_class: MemoryClass::Global,
                        visibility: BindingVisibility::ReadOnly,
                        name: "src".into(),
                    },
                    BindingSlot {
                        slot: 1,
                        element_type: DataType::U32,
                        element_count: Some(4),
                        memory_class: MemoryClass::Global,
                        visibility: BindingVisibility::ReadWrite,
                        name: "out".into(),
                    },
                ],
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
                        kind: KernelOpKind::LoadGlobal,
                        operands: vec![0, 0],
                        result: Some(1),
                    },
                    KernelOp {
                        kind: KernelOpKind::Literal,
                        operands: vec![1],
                        result: Some(2),
                    },
                    KernelOp {
                        kind: KernelOpKind::BinOpKind(binop),
                        operands: vec![1, 2],
                        result: Some(3),
                    },
                    KernelOp {
                        kind: KernelOpKind::StoreGlobal,
                        operands: vec![1, 0, 3],
                        result: None,
                    },
                ],
                child_bodies: vec![],
                literals: vec![LiteralValue::U32(0), LiteralValue::U32(5)],
            },
        };
        let module = emit(&desc).unwrap_or_else(|e| panic!("{binop:?}: emit failed: {e}"));
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("{binop:?} on a signed buffer load: INVALID WGSL: {e:?}"));
    }
}

#[test]
fn bitops_on_signed_buffer_load_emit_valid_wgsl() {
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    // Systematic sweep of the mixed-i32/u32 class: a value loaded from a SIGNED
    // (i32) buffer (whose kind doesn't resolve through Load(Access)) combined
    // with a u32 literal. naga requires matching operand kinds; if `unify` can't
    // resolve the load it emits e.g. `And(i32, u32)` and the module is invalid.
    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::all());
    let ops = [
        BinOp::BitAnd,
        BinOp::BitOr,
        BinOp::BitXor,
        BinOp::Shl,
        BinOp::Shr,
        BinOp::Add,
        BinOp::Sub,
        BinOp::Mul,
    ];
    for binop in ops {
        let desc = KernelDescriptor {
            id: "signed_bitop".into(),
            bindings: BindingLayout {
                slots: vec![
                    BindingSlot {
                        slot: 0,
                        element_type: DataType::I32,
                        element_count: Some(4),
                        memory_class: MemoryClass::Global,
                        visibility: BindingVisibility::ReadOnly,
                        name: "src".into(),
                    },
                    BindingSlot {
                        slot: 1,
                        element_type: DataType::U32,
                        element_count: Some(4),
                        memory_class: MemoryClass::Global,
                        visibility: BindingVisibility::ReadWrite,
                        name: "out".into(),
                    },
                ],
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
                        kind: KernelOpKind::LoadGlobal,
                        operands: vec![0, 0],
                        result: Some(1),
                    },
                    KernelOp {
                        kind: KernelOpKind::Literal,
                        operands: vec![1],
                        result: Some(2),
                    },
                    KernelOp {
                        kind: KernelOpKind::BinOpKind(binop),
                        operands: vec![1, 2],
                        result: Some(3),
                    },
                    KernelOp {
                        kind: KernelOpKind::StoreGlobal,
                        operands: vec![1, 0, 3],
                        result: None,
                    },
                ],
                child_bodies: vec![],
                literals: vec![LiteralValue::U32(0), LiteralValue::U32(0xff)],
            },
        };
        let module = emit(&desc).unwrap_or_else(|e| panic!("{binop:?}: emit failed: {e}"));
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("{binop:?} on a signed buffer load: INVALID WGSL: {e:?}"));
    }
}

#[test]
fn unpack_on_signed_buffer_load_emits_valid_wgsl() {
    use naga::valid::{Capabilities, ValidationFlags, Validator};
    // `Unpack8Low` lowers to `(v >> shift) & mask` with a u32 mask. When the
    // source `v` is a load from a SIGNED (i32) buffer, the value is Sint and its
    // `scalar_kind` does not resolve through the `Load(Access)` chain, so
    // `unify_binary_operand_types` cannot match the `& mask` operands → it would
    // emit `And(i32, u32)`, which naga rejects.
    let desc = KernelDescriptor {
        id: "unpack_signed".into(),
        bindings: BindingLayout {
            slots: vec![
                vyre_lower::BindingSlot {
                    slot: 0,
                    element_type: DataType::I32,
                    element_count: Some(4),
                    memory_class: MemoryClass::Global,
                    visibility: BindingVisibility::ReadOnly,
                    name: "src".into(),
                },
                vyre_lower::BindingSlot {
                    slot: 1,
                    element_type: DataType::U32,
                    element_count: Some(4),
                    memory_class: MemoryClass::Global,
                    visibility: BindingVisibility::ReadWrite,
                    name: "out".into(),
                },
            ],
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
                    kind: KernelOpKind::LoadGlobal,
                    operands: vec![0, 0],
                    result: Some(1),
                },
                KernelOp {
                    kind: KernelOpKind::UnOpKind(UnOp::Unpack8Low),
                    operands: vec![1],
                    result: Some(2),
                },
                KernelOp {
                    kind: KernelOpKind::StoreGlobal,
                    operands: vec![1, 0, 2],
                    result: None,
                },
            ],
            child_bodies: vec![],
            literals: vec![LiteralValue::U32(0)],
        },
    };
    let module = emit(&desc).expect("unpack-on-signed-load must emit");
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .expect("unpack on a signed buffer load must produce valid WGSL");
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
