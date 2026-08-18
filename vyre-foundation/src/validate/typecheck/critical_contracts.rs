use crate::ir::{BinOp, BufferDecl, DataType, Expr, Node, Program};
use crate::validate::validate;

const ABS_DIFF_I32_MESSAGE: &str =
    "can overflow (i32::MIN - i32::MAX invokes target-text signed-integer UB). Fix: cast operands to U32 before AbsDiff, or rewrite as an explicit branch.";

const NEGATE_I32_MESSAGE: &str =
    "Fix: use `0 - x` for wrapping i32 negation, cast to U32 before Negate, or guard with Select(i32::MIN, 0, -x).";

const SATURATING_MESSAGE: &str =
    "legal set is only U32 in the current lowering. Fix: cast both operands to U32, or clamp explicitly for I32/F32.";

const INTEGER_64_MESSAGE: &str =
    "64-bit integer arithmetic is outside vyre-foundation's cross-backend arithmetic contract. Fix: express the operation as a U32 pair with explicit carry/borrow, or use a backend-specific op whose schema declares native 64-bit arithmetic.";

fn assert_rejected(expr: Expr, output_ty: DataType, expected: &str) {
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, output_ty)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), expr)],
    );
    let errors = validate(&program);
    assert!(
        errors
            .iter()
            .any(|error| error.message().contains(expected)),
        "expected critical rejection: {expected}"
    );
}

#[test]
fn val_001_abs_diff_on_i32_is_rejected() {
    let expr = Expr::abs_diff(Expr::i32(i32::MIN), Expr::i32(42));
    assert_rejected(expr, DataType::I32, ABS_DIFF_I32_MESSAGE);
}

#[test]
fn val_002_negate_on_i32_is_rejected() {
    let expr = Expr::negate(Expr::i32(i32::MIN));
    assert_rejected(expr, DataType::I32, NEGATE_I32_MESSAGE);
}

#[test]
fn val_003_saturating_i32_and_f32_are_rejected() {
    let i32_expr = Expr::BinOp {
        op: BinOp::SaturatingAdd,
        left: Box::new(Expr::i32(1)),
        right: Box::new(Expr::i32(2)),
    };
    assert_rejected(i32_expr, DataType::I32, SATURATING_MESSAGE);

    let f32_expr = Expr::BinOp {
        op: BinOp::SaturatingMul,
        left: Box::new(Expr::f32(1.0)),
        right: Box::new(Expr::f32(2.0)),
    };
    assert_rejected(f32_expr, DataType::F32, SATURATING_MESSAGE);
}

#[test]
fn val_004_arithmetic_on_i64_u64_is_rejected() {
    let i64_expr = Expr::BinOp {
        op: BinOp::Add,
        left: Box::new(Expr::i64(4)),
        right: Box::new(Expr::i64(2)),
    };
    assert_rejected(i64_expr, DataType::I64, INTEGER_64_MESSAGE);

    let u64_expr = Expr::BinOp {
        op: BinOp::Mul,
        left: Box::new(Expr::u64(4)),
        right: Box::new(Expr::u64(2)),
    };
    assert_rejected(u64_expr, DataType::U64, INTEGER_64_MESSAGE);
}

#[test]
fn val_005_bool_ordered_comparisons_are_rejected_while_equality_is_accepted() {
    for op in [BinOp::Lt, BinOp::Gt, BinOp::Le, BinOp::Ge] {
        let bool_ordered = Expr::BinOp {
            op,
            left: Box::new(Expr::bool(true)),
            right: Box::new(Expr::bool(false)),
        };
        assert_rejected(
            bool_ordered,
            DataType::Bool,
            "ordered comparison",
        );

        let bool_mismatched = Expr::BinOp {
            op,
            left: Box::new(Expr::bool(true)),
            right: Box::new(Expr::u32(1)),
        };
        let program = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::Bool)],
            [1, 1, 1],
            vec![Node::store("out", Expr::u32(0), bool_mismatched)],
        );
        let errors = validate(&program);
        assert!(
            errors.iter().any(|e| e.code().as_str() == "V096"),
            "ordered comparison with bool+u32 must emit V096: {errors:?}"
        );
    }

    // Equality on bools is accepted
    for op in [BinOp::Eq, BinOp::Ne] {
        let bool_eq = Expr::BinOp {
            op,
            left: Box::new(Expr::bool(true)),
            right: Box::new(Expr::bool(false)),
        };
        let program = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::Bool)],
            [1, 1, 1],
            vec![Node::store("out", Expr::u32(0), bool_eq)],
        );
        let errors = validate(&program);
        assert!(
            !errors.iter().any(|e| e.code().as_str() == "V096"),
            "equality on bools must be accepted: {errors:?}"
        );
    }
}

#[test]
fn val_006_static_integer_division_by_zero_is_rejected_while_float_is_accepted() {
    let u32_zero_div = Expr::div(Expr::u32(10), Expr::u32(0));
    assert_rejected(
        u32_zero_div,
        DataType::U32,
        "binary operation `Div` has a statically-zero divisor",
    );

    let i32_zero_div = Expr::div(Expr::i32(10), Expr::i32(0));
    assert_rejected(
        i32_zero_div,
        DataType::I32,
        "binary operation `Div` has a statically-zero divisor",
    );

    // IEEE-754 float division by static zero is accepted
    for f32_zero in [
        Expr::f32(0.0),
        Expr::LitF32(-0.0),
        Expr::cast(DataType::F32, Expr::u32(0)),
    ] {
        let f32_div = Expr::div(Expr::f32(1.0), f32_zero);
        let program = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::F32)],
            [1, 1, 1],
            vec![Node::store("out", Expr::u32(0), f32_div)],
        );
        let errors = validate(&program);
        assert!(
            !errors.iter().any(|e| e.code().as_str() == "V044"),
            "float division by zero is IEEE-754 defined and must not emit V044: {errors:?}"
        );
    }
}

#[test]
fn val_007_subgroup_backend_capabilities_resolution() {
    use crate::validate::{
        validate_with_options, BackendCapabilities, BackendValidationCapabilities, ValidationOptions,
    };

    struct SubgroupGpu;
    impl BackendValidationCapabilities for SubgroupGpu {
        fn backend_name(&self) -> &'static str {
            "subgroup-gpu"
        }
        fn supports_cast_target(&self, _target: &DataType) -> bool {
            true
        }
        fn supports_subgroup_ops(&self) -> bool {
            true
        }
    }

    struct NonSubgroupGpu;
    impl BackendValidationCapabilities for NonSubgroupGpu {
        fn backend_name(&self) -> &'static str {
            "non-subgroup-gpu"
        }
        fn supports_cast_target(&self, _target: &DataType) -> bool {
            true
        }
        fn supports_subgroup_ops(&self) -> bool {
            false
        }
    }

    let prog = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::subgroup_add(Expr::u32(1)))],
    );

    // 1. Backend trait with subgroup support -> accepted
    let gpu = SubgroupGpu;
    let opts = ValidationOptions {
        backend: Some(&gpu),
        backend_capabilities: None,
        allow_shadowing: false,
    };
    let report = validate_with_options(&prog, opts);
    assert!(
        !report.errors.iter().any(|e| e.code().as_str() == "V041"),
        "backend trait with subgroup support must accept subgroup ops: {:?}",
        report.errors
    );

    // 2. Backend trait without subgroup support -> rejected with V041
    let non_gpu = NonSubgroupGpu;
    let opts = ValidationOptions {
        backend: Some(&non_gpu),
        backend_capabilities: None,
        allow_shadowing: false,
    };
    let report = validate_with_options(&prog, opts);
    assert!(
        report.errors.iter().any(|e| e.code().as_str() == "V041"),
        "backend trait without subgroup support must emit V041"
    );

    // 3. Backend capabilities snapshot with subgroup support -> accepted
    let caps = BackendCapabilities {
        supports_subgroup_ops: true,
        ..BackendCapabilities::default()
    };
    let opts = ValidationOptions::universal().with_backend_capabilities(caps);
    let report = validate_with_options(&prog, opts);
    assert!(
        !report.errors.iter().any(|e| e.code().as_str() == "V041"),
        "backend capabilities with subgroup support must accept subgroup ops: {:?}",
        report.errors
    );

    // 4. Default universal validation -> rejected with V041
    let report = validate_with_options(&prog, ValidationOptions::universal());
    assert!(
        report.errors.iter().any(|e| e.code().as_str() == "V041"),
        "universal validation without backend must emit V041"
    );
}

#[test]
fn val_008_wrapping_ops_and_mul_high_operand_rejections() {
    // Wrapping ops reject Bool
    for op in [BinOp::WrappingAdd, BinOp::WrappingSub] {
        let bool_left = Expr::BinOp {
            op,
            left: Box::new(Expr::bool(true)),
            right: Box::new(Expr::u32(1)),
        };
        let prog = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32)],
            [1, 1, 1],
            vec![Node::store("out", Expr::u32(0), bool_left)],
        );
        let errors = validate(&prog);
        assert!(
            errors.iter().any(|e| e.code().as_str() == "V091"),
            "`{op:?}` with bool left operand must emit V091: {errors:?}"
        );

        let bool_right = Expr::BinOp {
            op,
            left: Box::new(Expr::u32(1)),
            right: Box::new(Expr::bool(false)),
        };
        let prog = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32)],
            [1, 1, 1],
            vec![Node::store("out", Expr::u32(0), bool_right)],
        );
        let errors = validate(&prog);
        assert!(
            errors.iter().any(|e| e.code().as_str() == "V092"),
            "`{op:?}` with bool right operand must emit V092: {errors:?}"
        );

        let f32_left = Expr::BinOp {
            op,
            left: Box::new(Expr::f32(1.0)),
            right: Box::new(Expr::u32(1)),
        };
        let prog = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32)],
            [1, 1, 1],
            vec![Node::store("out", Expr::u32(0), f32_left)],
        );
        let errors = validate(&prog);
        assert!(
            errors.iter().any(|e| e.code().as_str() == "V091"),
            "`{op:?}` with f32 left operand must emit V091: {errors:?}"
        );

        let mixed_int = Expr::BinOp {
            op,
            left: Box::new(Expr::u32(1)),
            right: Box::new(Expr::i32(2)),
        };
        let prog = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32)],
            [1, 1, 1],
            vec![Node::store("out", Expr::u32(0), mixed_int)],
        );
        let errors = validate(&prog);
        assert!(
            errors.iter().any(|e| e.code().as_str() == "V093"),
            "`{op:?}` with mixed u32/i32 must emit V093: {errors:?}"
        );
    }

    // MulHigh tests
    let mul_high_valid = Expr::BinOp {
        op: BinOp::MulHigh,
        left: Box::new(Expr::u32(10)),
        right: Box::new(Expr::u32(20)),
    };
    let prog = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32)],
         [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), mul_high_valid)],
    );
    let errors = validate(&prog);
    assert!(
        !errors.iter().any(|e| e.code().as_str() == "V094"),
        "MulHigh with u32 operands must be accepted: {errors:?}"
    );

    for non_u32 in [Expr::i32(10), Expr::f32(10.0), Expr::bool(true)] {
        let mul_high_invalid = Expr::BinOp {
            op: BinOp::MulHigh,
            left: Box::new(non_u32),
            right: Box::new(Expr::u32(20)),
        };
        let prog = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32)],
            [1, 1, 1],
            vec![Node::store("out", Expr::u32(0), mul_high_invalid)],
        );
        let errors = validate(&prog);
        assert!(
            errors.iter().any(|e| e.code().as_str() == "V094"),
            "MulHigh with non-u32 operand must emit V094: {errors:?}"
        );
    }
}

#[test]
fn val_009_all_specialized_location_variants_convert_to_diagnostics() {
    use crate::validate::ValidationLocation;
    use std::borrow::Cow;

    let loc_prog = ValidationLocation::Program.diagnostic_location();
    assert_eq!(loc_prog.op_id, "program");
    assert!(loc_prog.operand_idx.is_none());

    let loc_axis = ValidationLocation::WorkgroupAxis(2).diagnostic_location();
    assert_eq!(loc_axis.op_id, "program.workgroup_size");
    assert_eq!(loc_axis.operand_idx, Some(2));

    let loc_buf = ValidationLocation::Buffer(Cow::Borrowed("scratch")).diagnostic_location();
    assert_eq!(loc_buf.op_id, "program.buffer");
    assert_eq!(loc_buf.attr_name.as_deref(), Some("scratch"));

    let loc_node = ValidationLocation::Node(42).diagnostic_location();
    assert_eq!(loc_node.op_id, "program.node");
    assert_eq!(loc_node.graph_node, Some(42));

    let loc_expr = ValidationLocation::Expression { node: 5, depth: 3 }.diagnostic_location();
    assert_eq!(loc_expr.op_id, "program.expression");
    assert_eq!(loc_expr.graph_node, Some(5));
    assert_eq!(loc_expr.operand_idx, Some(3));

    let loc_op = ValidationLocation::Operand { node: 8, operand: 1 }.diagnostic_location();
    assert_eq!(loc_op.op_id, "program.expression");
    assert_eq!(loc_op.graph_node, Some(8));
    assert_eq!(loc_op.operand_idx, Some(1));

    let loc_trav = ValidationLocation::Traversal { ordinal: 7 }.diagnostic_location();
    assert_eq!(loc_trav.op_id, "program.validation");
    assert_eq!(loc_trav.graph_node, Some(7));

    let loc_oper = ValidationLocation::Operation(Cow::Borrowed("math.fma")).diagnostic_location();
    assert_eq!(loc_oper.op_id, "math.fma");
}
