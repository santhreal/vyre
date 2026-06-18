use super::*;

#[test]
fn cast_emits_cvt_with_target_dtype() {
    let kernel = KernelDescriptor {
        id: "cast".into(),
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
                    kind: KernelOpKind::Cast {
                        target: DataType::F32,
                    },
                    operands: vec![0],
                    result: Some(1),
                },
            ],
            child_bodies: vec![],
            literals: vec![LiteralValue::U32(7)],
        },
    };
    let s = emit(&kernel).unwrap();
    assert!(s.contains("cvt.rn.f32.u32"));
}

#[test]
fn f32_to_bool_cast_uses_unordered_not_equal_for_nan_truthiness() {
    let kernel = KernelDescriptor {
        id: "cast_f32_bool".into(),
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
                    kind: KernelOpKind::Cast {
                        target: DataType::Bool,
                    },
                    operands: vec![0],
                    result: Some(1),
                },
            ],
            child_bodies: vec![],
            literals: vec![LiteralValue::F32(f32::NAN)],
        },
    };

    let s = emit(&kernel).unwrap();
    assert!(
        s.contains("setp.neu.f32"),
        "f32 truthiness must treat NaN as true to match reference casts:\n{s}"
    );
}

#[test]
fn f32_not_equal_comparison_uses_unordered_predicate_for_nan_truthiness() {
    let kernel = KernelDescriptor {
        id: "f32_ne_nan".into(),
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
                    kind: KernelOpKind::Literal,
                    operands: vec![1],
                    result: Some(1),
                },
                KernelOp {
                    kind: KernelOpKind::BinOpKind(BinOp::Ne),
                    operands: vec![0, 1],
                    result: Some(2),
                },
            ],
            child_bodies: vec![],
            literals: vec![LiteralValue::F32(f32::NAN), LiteralValue::F32(1.0)],
        },
    };

    let s = emit(&kernel).unwrap();
    assert!(
        s.contains("setp.neu.f32"),
        "f32 Ne must be unordered-not-equal so NaN != x matches the reference oracle:\n{s}"
    );
}

#[test]
fn bool_to_f32_cast_materializes_predicate_before_numeric_conversion() {
    let kernel = KernelDescriptor {
        id: "cast_bool_f32".into(),
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
                    kind: KernelOpKind::Cast {
                        target: DataType::F32,
                    },
                    operands: vec![0],
                    result: Some(1),
                },
            ],
            child_bodies: vec![],
            literals: vec![LiteralValue::Bool(true)],
        },
    };

    let s = emit(&kernel).unwrap();
    assert!(
        s.contains("selp.u32") && s.contains("cvt.rn.f32.u32"),
        "Bool->F32 must materialize %p as a u32 word before cvt; PTX cannot cvt directly from predicate registers:\n{s}"
    );
}

#[test]
fn bool_to_i32_cast_materializes_predicate_word() {
    let kernel = KernelDescriptor {
        id: "cast_bool_i32".into(),
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
                    kind: KernelOpKind::Cast {
                        target: DataType::I32,
                    },
                    operands: vec![0],
                    result: Some(1),
                },
            ],
            child_bodies: vec![],
            literals: vec![LiteralValue::Bool(true)],
        },
    };

    let s = emit(&kernel).unwrap();
    assert!(
        s.contains("selp.u32"),
        "Bool->I32 must materialize %p as a 0/1 word:\n{s}"
    );
}
