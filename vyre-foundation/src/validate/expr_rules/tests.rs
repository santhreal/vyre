use super::*;
use crate::dialect_lookup::{Signature, TypedParam};
use crate::ir_inner::model::expr::{ExprNode, Ident};
use crate::operation::{OperationRegistration, OperationTier};
use crate::validate::BackendValidationCapabilities;
use rustc_hash::FxHashMap;
use std::any::Any;
use std::sync::Arc;

#[derive(Debug)]
struct SubgroupBackend {
    supports_subgroup_ops: bool,
}

impl BackendValidationCapabilities for SubgroupBackend {
    fn backend_name(&self) -> &'static str {
        "test-backend"
    }

    fn supports_cast_target(&self, target: &DataType) -> bool {
        matches!(target, DataType::U32)
    }

    fn supports_subgroup_ops(&self) -> bool {
        self.supports_subgroup_ops
    }
}

fn validate_subgroup_expr(expr: Expr, options: ValidationOptions<'_>) -> ValidationReport {
    let mut report = ValidationReport::default();
    let buffers = FxHashMap::default();
    let scope = FxHashMap::default();
    validate_expr(&expr, &buffers, &scope, options, &mut report, 0);
    report
}

const CALL_OP_ID: &str = "test::call_u32";
const CALL_SIGNATURE: Signature = Signature {
    inputs: &[TypedParam {
        name: "x",
        ty: "u32",
    }],
    outputs: &[],
    attrs: &[],
    bytes_extraction: false,
};

inventory::submit! {
    OperationRegistration::new_unconstrained(
        CALL_OP_ID,
        OperationTier::External,
        None,
        None,
        None,
    )
    .with_signature(CALL_SIGNATURE)
    .with_category("test")
}

#[derive(Debug)]
struct TestExprExtension;

impl ExprNode for TestExprExtension {
    fn extension_kind(&self) -> &'static str {
        "test.expr"
    }

    fn debug_identity(&self) -> &str {
        "test-expr"
    }

    fn result_type(&self) -> Option<DataType> {
        Some(DataType::U32)
    }

    fn cse_safe(&self) -> bool {
        true
    }

    fn stable_fingerprint(&self) -> [u8; 32] {
        [7; 32]
    }

    fn validate_extension(&self) -> Result<(), String> {
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[test]
fn expr_match_guard_stays_exhaustive() {
    fn guard(expr: &Expr) {
        match expr {
            Expr::LitU32(_)
            | Expr::LitI32(_)
            | Expr::LitF32(_)
            | Expr::LitBool(_)
            | Expr::Var(_)
            | Expr::BufferRef { .. }
            | Expr::Load { .. }
            | Expr::BufLen { .. }
            | Expr::InvocationId { .. }
            | Expr::LogicalIndex { .. }
            | Expr::LogicalTileId { .. }
            | Expr::LogicalWithinTileId { .. }
            | Expr::WorkgroupId { .. }
            | Expr::LocalId { .. }
            | Expr::SubgroupLocalId
            | Expr::SubgroupSize
            | Expr::BinOp { .. }
            | Expr::UnOp { .. }
            | Expr::Call { .. }
            | Expr::Select { .. }
            | Expr::Cast { .. }
            | Expr::Fma { .. }
            | Expr::Atomic { .. }
            | Expr::SubgroupBallot { .. }
            | Expr::SubgroupShuffle { .. }
            | Expr::SubgroupReduce { .. }
            | Expr::Opaque(_) => {}
        }
    }

    let exprs = [
        Expr::LitU32(1),
        Expr::LitI32(-1),
        Expr::LitF32(1.0),
        Expr::LitBool(true),
        Expr::Var(Ident::from("x")),
        Expr::buffer_ref("buf"),
        Expr::Load {
            buffer: Ident::from("buf"),
            index: Box::new(Expr::LitU32(0)),
        },
        Expr::BufLen {
            buffer: Ident::from("buf"),
        },
        Expr::InvocationId { axis: 0 },
        Expr::WorkgroupId { axis: 0 },
        Expr::LocalId { axis: 0 },
        Expr::BinOp {
            op: crate::ir_inner::model::op_signature::BinOp::Add,
            left: Box::new(Expr::LitU32(1)),
            right: Box::new(Expr::LitU32(2)),
        },
        Expr::UnOp {
            op: crate::ir_inner::model::op_signature::UnOp::LogicalNot,
            operand: Box::new(Expr::LitBool(false)),
        },
        Expr::Call {
            op_id: Ident::from("op"),
            args: vec![Expr::LitU32(1)],
        },
        Expr::Select {
            cond: Box::new(Expr::LitBool(true)),
            true_val: Box::new(Expr::LitU32(1)),
            false_val: Box::new(Expr::LitU32(0)),
        },
        Expr::Cast {
            target: DataType::U32,
            value: Box::new(Expr::LitU32(1)),
        },
        Expr::Fma {
            a: Box::new(Expr::LitF32(1.0)),
            b: Box::new(Expr::LitF32(2.0)),
            c: Box::new(Expr::LitF32(3.0)),
        },
        Expr::Atomic {
            op: crate::ir_inner::model::op_signature::AtomicOp::Add,
            buffer: Ident::from("buf"),
            index: Box::new(Expr::LitU32(0)),
            expected: None,
            value: Box::new(Expr::LitU32(1)),
            ordering: crate::memory_model::MemoryOrdering::SeqCst,
        },
        Expr::SubgroupBallot {
            cond: Box::new(Expr::bool(true)),
        },
        Expr::SubgroupShuffle {
            value: Box::new(Expr::u32(1)),
            lane: Box::new(Expr::u32(0)),
        },
        Expr::subgroup_add(Expr::u32(1)),
        Expr::Opaque(Arc::new(TestExprExtension)),
    ];

    for expr in &exprs {
        guard(expr);
    }
}

#[test]
fn subgroup_expression_without_backend_is_rejected() {
    let report = validate_subgroup_expr(
        Expr::subgroup_add(Expr::u32(1)),
        ValidationOptions::default(),
    );
    assert!(
        report.errors.iter().any(|error| error
            .message()
            .contains("subgroup expressions require backend subgroup-ops support")),
        "subgroup expression without backend capability must be rejected, got {:?}",
        report.errors
    );
}

#[test]
fn subgroup_expression_with_supported_backend_is_accepted() {
    let backend = SubgroupBackend {
        supports_subgroup_ops: true,
    };
    let report = validate_subgroup_expr(
        Expr::SubgroupShuffle {
            value: Box::new(Expr::u32(1)),
            lane: Box::new(Expr::u32(0)),
        },
        ValidationOptions::default().with_backend(&backend),
    );
    assert!(
        report.errors.is_empty(),
        "supported subgroup backend must allow validation, got {:?}",
        report.errors
    );
}

#[test]
fn subgroup_bitwise_reduction_rejects_f32_operand() {
    // V047: And/Or/Xor are bitwise reductions with no meaning over floats.
    // Target emitters and the reference oracle both fail closed on an f32
    // operand; validation must reject it at the type boundary too, with a
    // backend that DOES support subgroup ops (so V041 is not the reason).
    let backend = SubgroupBackend {
        supports_subgroup_ops: true,
    };
    let cases: [(fn(Expr) -> Expr, &str); 3] = [
        (Expr::subgroup_and, "And"),
        (Expr::subgroup_or, "Or"),
        (Expr::subgroup_xor, "Xor"),
    ];
    for (ctor, op_name) in cases {
        let report = validate_subgroup_expr(
            ctor(Expr::LitF32(1.5)),
            ValidationOptions::default().with_backend(&backend),
        );
        assert!(
            report.errors.iter().any(|error| {
                error.code().as_str() == "V047"
                    && error.cause().contains(op_name)
                    && error.cause().contains("bitwise reduction")
                    && error.cause().contains("rejects f32 operands")
                    && !error.corrective_action().is_empty()
            }),
            "subgroup `{op_name}` over an f32 operand must be rejected with V047, got {:?}",
            report.errors
        );
    }
}

#[test]
fn subgroup_bitwise_reduction_accepts_integer_operand() {
    // The positive twin: u32 is a valid operand for And/Or/Xor (no V047).
    let backend = SubgroupBackend {
        supports_subgroup_ops: true,
    };
    for ctor in [Expr::subgroup_and, Expr::subgroup_or, Expr::subgroup_xor] {
        let report = validate_subgroup_expr(
            ctor(Expr::u32(0b1010)),
            ValidationOptions::default().with_backend(&backend),
        );
        assert!(
            report.errors.is_empty(),
            "integer bitwise subgroup reduction is valid and must not be flagged, got {:?}",
            report.errors
        );
    }
}

#[test]
fn subgroup_arithmetic_reduction_accepts_f32_operand() {
    // f32 is rejected ONLY for bitwise ops. Add/Mul/Min/Max over f32 are
    // legitimate (the whole point of generalizing workgroup_max_f32 to a
    // subgroup reduction), so V047 must not fire for them.
    let backend = SubgroupBackend {
        supports_subgroup_ops: true,
    };
    for ctor in [
        Expr::subgroup_add,
        Expr::subgroup_mul,
        Expr::subgroup_min,
        Expr::subgroup_max,
    ] {
        let report = validate_subgroup_expr(
            ctor(Expr::LitF32(2.5)),
            ValidationOptions::default().with_backend(&backend),
        );
        assert!(
            !report
                .errors
                .iter()
                .any(|error| error.code().as_str() == "V047"),
            "f32 arithmetic subgroup reduction must not be rejected as bitwise, got {:?}",
            report.errors
        );
    }
}

#[test]
fn unknown_call_uses_canonical_operation_registry() {
    let report = validate_subgroup_expr(
        Expr::call("missing::call", vec![Expr::u32(1)]),
        ValidationOptions::default(),
    );
    assert!(
        report
            .errors
            .iter()
            .any(|error| error.code().as_str() == "V016"),
        "unknown call must be rejected by the canonical registry: {:?}",
        report.errors
    );
}

#[test]
fn call_signature_mismatch_uses_canonical_operation_registry() {
    let report = validate_subgroup_expr(
        Expr::call(CALL_OP_ID, vec![Expr::bool(true)]),
        ValidationOptions::default(),
    );
    assert!(
        report
            .errors
            .iter()
            .any(|error| error.code().as_str() == "V022"),
        "typed call mismatch must be rejected from the canonical signature: {:?}",
        report.errors
    );
}
