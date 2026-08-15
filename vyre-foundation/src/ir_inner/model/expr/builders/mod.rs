mod ops {
    use crate::ir_inner::model::expr::Expr;
    use crate::ir_inner::model::spec_types::{BinOp, UnOp};

    #[must_use]
    #[inline]
    pub(super) fn binary(op: BinOp, left: Expr, right: Expr) -> Expr {
        Expr::BinOp {
            op,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    #[must_use]
    #[inline]
    pub(super) fn unary(op: UnOp, operand: Expr) -> Expr {
        Expr::UnOp {
            op,
            operand: Box::new(operand),
        }
    }
}

mod operators;
mod wide_literals;
