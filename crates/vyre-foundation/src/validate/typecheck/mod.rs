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
    use crate::validate::{Binding, ValidationError};
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

    fn check_binop(
        op: BinOp,
        left: &Expr,
        right: &Expr,
        subgroup_ops: bool,
    ) -> Vec<ValidationError> {
        let mut errors = Vec::new();
        validate_binop_operands(
            op,
            left,
            right,
            &empty_buffers(),
            &empty_scope(),
            subgroup_ops,
            &mut errors,
        );
        errors
    }

    fn has_code(errors: &[ValidationError], code: &str) -> bool {
        errors.iter().any(|e| e.code().as_str() == code)
    }

    #[test]
    fn and_or_and_comparisons_type_as_bool_while_bitwise_is_integer() {
        for op in [BinOp::And, BinOp::Or] {
            let e = bin(op, Expr::LitBool(true), Expr::LitBool(false));
            assert_eq!(ty(&e), Some(DataType::Bool));
        }
        for op in [
            BinOp::Eq,
            BinOp::Ne,
            BinOp::Lt,
            BinOp::Gt,
            BinOp::Le,
            BinOp::Ge,
        ] {
            let e = bin(op, Expr::LitU32(1), Expr::LitU32(2));
            assert_eq!(ty(&e), Some(DataType::Bool));
        }
        assert_eq!(
            ty(&bin(BinOp::BitAnd, Expr::LitU32(1), Expr::LitU32(2))),
            Some(DataType::U32)
        );
    }

    #[test]
    fn bool_plus_int_is_rejected() {
        let and_expr = bin(BinOp::And, Expr::LitBool(true), Expr::LitBool(false));
        let errors = check_binop(BinOp::Add, &and_expr, &Expr::LitU32(1), false);
        assert_eq!(errors.len(), 1, "bool + int must produce one type error");
        let msg = errors[0].message();
        assert!(msg.contains("Bool") || msg.contains("type"), "{msg}");
    }

    #[test]
    fn static_zero_div_and_mod_rules() {
        for (op, zero_rhs) in [
            (BinOp::Div, Expr::LitU32(0)),
            (
                BinOp::Div,
                Expr::Cast {
                    target: DataType::U32,
                    value: Box::new(Expr::LitI32(0)),
                },
            ),
            (
                BinOp::Div,
                Expr::Cast {
                    target: DataType::U32,
                    value: Box::new(Expr::LitU32(0)),
                },
            ),
            (BinOp::Mod, Expr::LitU32(0)),
        ] {
            let errors = check_binop(op, &Expr::LitU32(9), &zero_rhs, false);
            assert!(
                has_code(&errors, "V044"),
                "{op:?} / {zero_rhs:?}: {errors:?}"
            );
        }
        let float_errors = check_binop(BinOp::Div, &Expr::LitF32(1.0), &Expr::LitF32(0.0), false);
        assert!(float_errors.is_empty(), "{float_errors:?}");
    }

    #[test]
    fn comparison_bool_ordered_vs_equality_split() {
        for op in [BinOp::Lt, BinOp::Gt, BinOp::Le, BinOp::Ge] {
            let errors = check_binop(op, &Expr::LitBool(true), &Expr::LitBool(false), false);
            assert!(has_code(&errors, "V096"), "{op:?}: {errors:?}");
        }
        for op in [BinOp::Eq, BinOp::Ne] {
            let errors = check_binop(op, &Expr::LitBool(true), &Expr::LitBool(false), false);
            assert!(errors.is_empty(), "{op:?}: {errors:?}");
        }
    }

    #[test]
    fn wrapping_add_sub_operand_rules() {
        for op in [BinOp::WrappingAdd, BinOp::WrappingSub] {
            for (l, r) in [
                (Expr::LitU32(1), Expr::LitU32(2)),
                (Expr::LitI32(1), Expr::LitI32(2)),
            ] {
                let errors = check_binop(op, &l, &r, false);
                assert!(errors.is_empty(), "{op:?} ({l:?}, {r:?}): {errors:?}");
            }
            for (l, r) in [
                (Expr::LitBool(true), Expr::LitBool(false)),
                (Expr::LitF32(1.0), Expr::LitF32(2.0)),
            ] {
                let errors = check_binop(op, &l, &r, false);
                assert!(
                    has_code(&errors, "V091") || has_code(&errors, "V092"),
                    "{op:?} ({l:?}, {r:?}): {errors:?}"
                );
            }
            let mixed = check_binop(op, &Expr::LitU32(1), &Expr::LitI32(2), false);
            assert!(has_code(&mixed, "V093"), "{op:?}: {mixed:?}");
        }
    }

    #[test]
    fn mul_high_and_subgroup_operand_rules() {
        assert!(
            check_binop(BinOp::MulHigh, &Expr::LitU32(10), &Expr::LitU32(20), false).is_empty()
        );
        for (l, r) in [
            (Expr::LitI32(10), Expr::LitI32(20)),
            (Expr::LitF32(1.0), Expr::LitF32(2.0)),
        ] {
            let errors = check_binop(BinOp::MulHigh, &l, &r, false);
            assert!(has_code(&errors, "V094"), "{errors:?}");
        }
        for op in [
            BinOp::Shuffle,
            BinOp::Ballot,
            BinOp::WaveReduce,
            BinOp::WaveBroadcast,
        ] {
            let unsupported = check_binop(op, &Expr::LitU32(1), &Expr::LitU32(2), false);
            assert!(has_code(&unsupported, "V097"), "{op:?}: {unsupported:?}");
            let supported = check_binop(op, &Expr::LitU32(1), &Expr::LitU32(2), true);
            assert!(supported.is_empty(), "{op:?}: {supported:?}");
        }
    }
}
