//! Static type checking over IR expressions.
//!
//! [`expr_type`] is the one answer in this crate to "what type does this
//! expression have"; [`operands`] is what each operator accepts in each
//! position, and reads the walker for its answer.

/// What each operator accepts in each operand position.
mod operands;

/// The static type walker and its name environment.
mod expr_type;

pub(crate) use expr_type::{expr_type, ScopeTypes, TypeEnv};
pub(crate) use operands::{validate_binop_operands, validate_unop_operand};

#[cfg(test)]
mod critical_contracts;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_inner::model::expr::Expr;
    use crate::ir_inner::model::op_signature::{BinOp, DataType};
    use crate::ir_inner::model::program::BufferDecl;
    use crate::validate::Binding;
    use rustc_hash::FxHashMap;

    fn empty_buffers() -> FxHashMap<&'static str, &'static BufferDecl> {
        FxHashMap::default()
    }
    fn empty_scope() -> FxHashMap<crate::ir::Ident, Binding> {
        FxHashMap::default()
    }

    fn bin(op: BinOp, l: Expr, r: Expr) -> Expr {
        Expr::BinOp {
            op,
            left: Box::new(l),
            right: Box::new(r),
        }
    }

    fn ty(expr: &Expr) -> Option<DataType> {
        expr_type(expr, &mut ScopeTypes::new(&empty_buffers(), &empty_scope()))
    }

    #[test]
    fn and_or_type_is_bool() {
        for op in [BinOp::And, BinOp::Or] {
            let e = bin(op, Expr::LitBool(true), Expr::LitBool(false));
            assert_eq!(
                ty(&e),
                Some(DataType::Bool),
                "And/Or must type as Bool (reference interpreter produces Value::Bool)"
            );
        }
    }

    #[test]
    fn comparisons_type_is_bool() {
        for op in [
            BinOp::Eq,
            BinOp::Ne,
            BinOp::Lt,
            BinOp::Gt,
            BinOp::Le,
            BinOp::Ge,
        ] {
            let e = bin(op, Expr::LitU32(1), Expr::LitU32(2));
            assert_eq!(ty(&e), Some(DataType::Bool), "comparison must type as Bool");
        }
    }

    #[test]
    fn bitwise_type_is_integer() {
        let e = bin(BinOp::BitAnd, Expr::LitU32(1), Expr::LitU32(2));
        assert_eq!(ty(&e), Some(DataType::U32));
    }

    #[test]
    fn bool_plus_int_is_rejected() -> Result<(), String> {
        // Regression for REF-002: `(a && b) + 1`  -  previously accepted because
        // And was typed U32. Now And types as Bool, so arithmetic must reject.
        let and_expr = bin(BinOp::And, Expr::LitBool(true), Expr::LitBool(false));
        let add_expr = bin(BinOp::Add, and_expr, Expr::LitU32(1));
        let mut errors = Vec::new();
        if let Expr::BinOp { op, left, right } = &add_expr {
            validate_binop_operands(
                *op,
                left,
                right,
                &empty_buffers(),
                &empty_scope(),
                false,
                &mut errors,
            );
        } else {
            return Err("expected BinOp".to_string());
        }
        assert_eq!(
            errors.len(),
            1,
            "bool + int must produce exactly one type error"
        );
        assert!(
            errors[0].message().contains("Bool") || errors[0].message().contains("type"),
            "type error must mention Bool mismatch: {}",
            errors[0].message()
        );
        Ok(())
    }

    #[test]
    fn div_by_static_zero_is_rejected() {
        let mut errors = Vec::new();
        validate_binop_operands(
            BinOp::Div,
            &Expr::LitU32(9),
            &Expr::LitU32(0),
            &empty_buffers(),
            &empty_scope(),
            false,
            &mut errors,
        );
        assert!(errors.iter().any(|error| error.code().as_str() == "V044"));
    }

    #[test]
    fn div_by_casted_static_zero_is_rejected() {
        let mut errors = Vec::new();
        validate_binop_operands(
            BinOp::Div,
            &Expr::LitU32(9),
            &Expr::Cast {
                target: DataType::U32,
                value: Box::new(Expr::LitI32(0)),
            },
            &empty_buffers(),
            &empty_scope(),
            false,
            &mut errors,
        );
    }

    #[test]
    fn mod_by_static_zero_is_rejected() {
        let mut errors = Vec::new();
        validate_binop_operands(
            BinOp::Mod,
            &Expr::LitU32(9),
            &Expr::LitU32(0),
            &empty_buffers(),
            &empty_scope(),
            false,
            &mut errors,
        );
    }
    #[test]
    fn float_div_by_static_zero_is_accepted() {
        let mut errors = Vec::new();
        validate_binop_operands(
            BinOp::Div,
            &Expr::LitF32(1.0),
            &Expr::LitF32(0.0),
            &empty_buffers(),
            &empty_scope(),
            false,
            &mut errors,
        );
        assert!(
            errors.is_empty(),
            "float division by zero is defined in IEEE-754 and must not emit V044: {errors:?}"
        );
    }

    #[test]
    fn ordered_bool_comparisons_are_rejected() {
        for op in [BinOp::Lt, BinOp::Gt, BinOp::Le, BinOp::Ge] {
            let mut errors = Vec::new();
            validate_binop_operands(
                op,
                &Expr::LitBool(true),
                &Expr::LitBool(false),
                &empty_buffers(),
                &empty_scope(),
                false,
                &mut errors,
            );
            assert!(
                errors.iter().any(|e| e.code().as_str() == "V096"),
                "ordered comparison `{op:?}` on Bool must emit V096: {errors:?}"
            );
        }
    }

    #[test]
    fn equality_bool_comparisons_are_accepted() {
        for op in [BinOp::Eq, BinOp::Ne] {
            let mut errors = Vec::new();
            validate_binop_operands(
                op,
                &Expr::LitBool(true),
                &Expr::LitBool(false),
                &empty_buffers(),
                &empty_scope(),
                false,
                &mut errors,
            );
            assert!(
                errors.is_empty(),
                "equality comparison `{op:?}` on Bool must be accepted: {errors:?}"
            );
        }
    }

    #[test]
    fn wrapping_add_sub_matching_integers_accepted() {
        for op in [BinOp::WrappingAdd, BinOp::WrappingSub] {
            // u32 + u32
            let mut errors = Vec::new();
            validate_binop_operands(
                op,
                &Expr::LitU32(1),
                &Expr::LitU32(2),
                &empty_buffers(),
                &empty_scope(),
                false,
                &mut errors,
            );
            assert!(
                errors.is_empty(),
                "u32 `{op:?}` must be accepted: {errors:?}"
            );

            // i32 + i32
            let mut errors = Vec::new();
            validate_binop_operands(
                op,
                &Expr::LitI32(1),
                &Expr::LitI32(2),
                &empty_buffers(),
                &empty_scope(),
                false,
                &mut errors,
            );
            assert!(
                errors.is_empty(),
                "i32 `{op:?}` must be accepted: {errors:?}"
            );
        }
    }

    #[test]
    fn wrapping_add_sub_mismatched_and_non_integer_rejected() {
        for op in [BinOp::WrappingAdd, BinOp::WrappingSub] {
            // Bool + Bool
            let mut errors = Vec::new();
            validate_binop_operands(
                op,
                &Expr::LitBool(true),
                &Expr::LitBool(false),
                &empty_buffers(),
                &empty_scope(),
                false,
                &mut errors,
            );
            assert!(
                errors
                    .iter()
                    .any(|e| e.code().as_str() == "V091" || e.code().as_str() == "V092"),
                "bool `{op:?}` must be rejected with V091/V092: {errors:?}"
            );

            // F32 + F32
            let mut errors = Vec::new();
            validate_binop_operands(
                op,
                &Expr::LitF32(1.0),
                &Expr::LitF32(2.0),
                &empty_buffers(),
                &empty_scope(),
                false,
                &mut errors,
            );
            assert!(
                errors
                    .iter()
                    .any(|e| e.code().as_str() == "V091" || e.code().as_str() == "V092"),
                "f32 `{op:?}` must be rejected with V091/V092: {errors:?}"
            );

            // u32 + i32
            let mut errors = Vec::new();
            validate_binop_operands(
                op,
                &Expr::LitU32(1),
                &Expr::LitI32(2),
                &empty_buffers(),
                &empty_scope(),
                false,
                &mut errors,
            );
            assert!(
                errors.iter().any(|e| e.code().as_str() == "V093"),
                "mixed-width `{op:?}` must be rejected with V093: {errors:?}"
            );
        }
    }

    #[test]
    fn mul_high_operand_rules() {
        // u32 * u32 -> accepted
        let mut errors = Vec::new();
        validate_binop_operands(
            BinOp::MulHigh,
            &Expr::LitU32(10),
            &Expr::LitU32(20),
            &empty_buffers(),
            &empty_scope(),
            false,
            &mut errors,
        );
        assert!(
            errors.is_empty(),
            "u32 MulHigh must be accepted: {errors:?}"
        );

        // i32 * i32 -> rejected with V094
        let mut errors = Vec::new();
        validate_binop_operands(
            BinOp::MulHigh,
            &Expr::LitI32(10),
            &Expr::LitI32(20),
            &empty_buffers(),
            &empty_scope(),
            false,
            &mut errors,
        );
        assert!(
            errors.iter().any(|e| e.code().as_str() == "V094"),
            "i32 MulHigh must be rejected with V094: {errors:?}"
        );

        // f32 * f32 -> rejected with V094
        let mut errors = Vec::new();
        validate_binop_operands(
            BinOp::MulHigh,
            &Expr::LitF32(1.0),
            &Expr::LitF32(2.0),
            &empty_buffers(),
            &empty_scope(),
            false,
            &mut errors,
        );
        assert!(
            errors.iter().any(|e| e.code().as_str() == "V094"),
            "f32 MulHigh must be rejected with V094: {errors:?}"
        );
    }

    #[test]
    fn subgroup_binop_capability_awareness() {
        for op in [
            BinOp::Shuffle,
            BinOp::Ballot,
            BinOp::WaveReduce,
            BinOp::WaveBroadcast,
        ] {
            // without capability -> rejected with V097
            let mut errors = Vec::new();
            validate_binop_operands(
                op,
                &Expr::LitU32(1),
                &Expr::LitU32(2),
                &empty_buffers(),
                &empty_scope(),
                false,
                &mut errors,
            );
            assert!(
                errors.iter().any(|e| e.code().as_str() == "V097"),
                "`{op:?}` without subgroup support must emit V097: {errors:?}"
            );

            // with capability -> accepted
            let mut errors = Vec::new();
            validate_binop_operands(
                op,
                &Expr::LitU32(1),
                &Expr::LitU32(2),
                &empty_buffers(),
                &empty_scope(),
                true,
                &mut errors,
            );
            assert!(
                errors.is_empty(),
                "`{op:?}` with subgroup support must be accepted: {errors:?}"
            );
        }
    }
}
