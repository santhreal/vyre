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
#[path = "typecheck_critical_test.rs"]
mod typecheck_critical_test;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_inner::model::expr::Expr;
    use crate::ir_inner::model::program::BufferDecl;
    use crate::ir_inner::model::spec_types::{BinOp, DataType};
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
            &mut errors,
        );
        assert!(errors.iter().any(|error| error.code().as_str() == "V044"));
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
            &mut errors,
        );
        assert!(errors.iter().any(|error| error.code().as_str() == "V044"));
    }
}
