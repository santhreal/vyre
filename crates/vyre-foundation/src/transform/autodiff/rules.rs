//! Symbolic differentiation rules for Vyre IR expressions.
//!
//! Each rule takes the forward expression + the adjoint flowing into its
//! output and returns a list of `(child_expr, child_adjoint)` pairs that
//! propagate the chain rule down.

use crate::ir::{BinOp, Expr, UnOp};

use super::AutodiffError;

/// One adjoint contribution: the adjoint value to add to a child expression.
#[derive(Debug, Clone)]
pub struct AdjointContrib {
    /// The child expression this adjoint flows to.
    pub child: Expr,
    /// The adjoint value: `parent_adjoint * local_jacobian`.
    pub adjoint: Expr,
}

fn neg(operand: Expr) -> Expr {
    Expr::UnOp {
        op: UnOp::Negate,
        operand: Box::new(operand),
    }
}

fn unary(op: UnOp, operand: Expr) -> Expr {
    Expr::UnOp {
        op,
        operand: Box::new(operand),
    }
}

/// Compute the adjoint contributions for a `BinOp` expression.
///
/// Given `z = left ⊕ right` and `dz` (adjoint on z), returns `(d_left, d_right)`.
///
/// # Errors
///
/// Returns `NotDifferentiable` for integer/bitwise/comparison ops.
#[expect(
    clippy::too_many_lines,
    reason = "binary autodiff rules are a compact mathematical dispatch table and must remain auditable as a complete BinOp surface"
)]
pub fn binop_adjoints(
    op: BinOp,
    left: &Expr,
    right: &Expr,
    adjoint: &Expr,
) -> Result<Vec<AdjointContrib>, AutodiffError> {
    let (dl, dr) = match op {
        // d(a+b)/da = 1, d(a+b)/db = 1
        BinOp::Add => {
            let dz = adjoint.clone();
            (dz.clone(), dz)
        }
        // d(a-b)/da = 1, d(a-b)/db = -1
        BinOp::Sub => {
            let dz = adjoint.clone();
            (dz.clone(), neg(dz))
        }
        // d(a*b)/da = b, d(a*b)/db = a
        BinOp::Mul => {
            let dz = adjoint.clone();
            (
                Expr::mul(dz.clone(), right.clone()),
                Expr::mul(dz, left.clone()),
            )
        }
        // d(a/b)/da = 1/b, d(a/b)/db = -a/b²
        BinOp::Div => {
            let (dz, r) = (adjoint.clone(), right.clone());
            (
                Expr::div(dz.clone(), r.clone()),
                neg(Expr::div(
                    Expr::mul(dz, left.clone()),
                    Expr::mul(r.clone(), r),
                )),
            )
        }
        // d(min(a,b))/da = (a < b) ? dz : 0, d(min(a,b))/db = (b <= a) ? dz : 0
        // Subgradient: route adjoint to the arg that was selected.
        BinOp::Min => {
            let dz = adjoint.clone();
            (
                Expr::select(
                    Expr::lt(left.clone(), right.clone()),
                    dz.clone(),
                    Expr::f32(0.0),
                ),
                Expr::select(Expr::le(left.clone(), right.clone()), Expr::f32(0.0), dz),
            )
        }
        // max: mirror of min
        BinOp::Max => {
            let cmp = Expr::gt(left.clone(), right.clone());
            let dz = adjoint.clone();
            (
                Expr::select(cmp.clone(), dz.clone(), Expr::f32(0.0)),
                Expr::select(cmp, Expr::f32(0.0), dz),
            )
        }
        // Non-differentiable
        BinOp::Mod
        | BinOp::BitAnd
        | BinOp::BitOr
        | BinOp::BitXor
        | BinOp::Shl
        | BinOp::Shr
        | BinOp::WrappingAdd
        | BinOp::WrappingSub
        | BinOp::Eq
        | BinOp::Ne
        | BinOp::Lt
        | BinOp::Gt
        | BinOp::Le
        | BinOp::Ge
        | BinOp::And
        | BinOp::Or
        | BinOp::AbsDiff
        | BinOp::SaturatingAdd
        | BinOp::SaturatingSub
        | BinOp::SaturatingMul
        | BinOp::Shuffle
        | BinOp::Ballot
        | BinOp::WaveReduce
        | BinOp::WaveBroadcast
        | BinOp::RotateLeft
        | BinOp::RotateRight
        | BinOp::Opaque(_)
        | _ => {
            return Err(AutodiffError::NotDifferentiable {
                op: format!("BinOp::{op:?}"),
                fix: "replace with a differentiable equivalent or gate behind a stop-gradient barrier"
                    .into(),
            });
        }
    };
    Ok(vec![
        AdjointContrib {
            child: left.clone(),
            adjoint: dl,
        },
        AdjointContrib {
            child: right.clone(),
            adjoint: dr,
        },
    ])
}

/// Compute the adjoint contribution for a `UnOp` expression.
///
/// Given `z = op(x)` and `dz`, returns `dx = dz * dz/dx`.
///
/// # Errors
///
/// Returns `NotDifferentiable` for integer/bitwise ops.
#[expect(
    clippy::too_many_lines,
    reason = "unary autodiff rules are a compact mathematical dispatch table and must remain auditable as a complete UnOp surface"
)]
pub fn unop_adjoint(
    op: &UnOp,
    operand: &Expr,
    adjoint: &Expr,
) -> Result<AdjointContrib, AutodiffError> {
    if matches!(
        op,
        UnOp::BitNot
            | UnOp::LogicalNot
            | UnOp::Popcount
            | UnOp::Clz
            | UnOp::Ctz
            | UnOp::ReverseBits
            | UnOp::Floor
            | UnOp::Ceil
            | UnOp::Round
            | UnOp::Trunc
            | UnOp::Sign
            | UnOp::IsNan
            | UnOp::IsInf
            | UnOp::IsFinite
            | UnOp::Unpack4Low
            | UnOp::Unpack4High
            | UnOp::Unpack8Low
            | UnOp::Unpack8High
            | UnOp::BitcastF32ToU32
            | UnOp::BitcastU32ToF32
            | UnOp::Opaque(_)
    ) {
        return Err(AutodiffError::NotDifferentiable {
            op: format!("UnOp::{op:?}"),
            fix: "replace with a differentiable equivalent or gate behind a stop-gradient barrier"
                .into(),
        });
    }
    let (x, dz) = (operand.clone(), adjoint.clone());
    let dx = match op {
        UnOp::Negate => neg(dz),
        UnOp::Exp => Expr::mul(dz, unary(UnOp::Exp, x)),
        UnOp::Log => Expr::div(dz, x),
        UnOp::Sqrt => Expr::div(dz, Expr::mul(Expr::f32(2.0), unary(UnOp::Sqrt, x))),
        UnOp::Tanh => {
            let t = unary(UnOp::Tanh, x);
            Expr::mul(dz, Expr::sub(Expr::f32(1.0), Expr::mul(t.clone(), t)))
        }
        UnOp::Sin => Expr::mul(dz, unary(UnOp::Cos, x)),
        UnOp::Cos => neg(Expr::mul(dz, unary(UnOp::Sin, x))),
        UnOp::Abs => Expr::mul(dz, unary(UnOp::Sign, x)),
        UnOp::Exp2 => Expr::mul(
            Expr::mul(dz, unary(UnOp::Exp2, x)),
            Expr::f32(core::f32::consts::LN_2),
        ),
        UnOp::Log2 => Expr::div(dz, Expr::mul(x, Expr::f32(core::f32::consts::LN_2))),
        UnOp::Tan => {
            let t = unary(UnOp::Tan, x);
            Expr::mul(dz, Expr::add(Expr::f32(1.0), Expr::mul(t.clone(), t)))
        }
        UnOp::Sinh => Expr::mul(dz, unary(UnOp::Cosh, x)),
        UnOp::Cosh => Expr::mul(dz, unary(UnOp::Sinh, x)),
        UnOp::Asin => Expr::div(
            dz,
            unary(
                UnOp::Sqrt,
                Expr::sub(Expr::f32(1.0), Expr::mul(x.clone(), x)),
            ),
        ),
        UnOp::Acos => neg(Expr::div(
            dz,
            unary(
                UnOp::Sqrt,
                Expr::sub(Expr::f32(1.0), Expr::mul(x.clone(), x)),
            ),
        )),
        UnOp::Atan => Expr::div(dz, Expr::add(Expr::f32(1.0), Expr::mul(x.clone(), x))),
        UnOp::InverseSqrt => {
            let sqrt_x = unary(UnOp::Sqrt, x.clone());
            neg(Expr::div(
                dz,
                Expr::mul(Expr::f32(2.0), Expr::mul(x, sqrt_x)),
            ))
        }
        UnOp::Reciprocal => neg(Expr::div(dz, Expr::mul(x.clone(), x))),
        _ => {
            return Err(AutodiffError::NotDifferentiable {
                op: format!("UnOp::{op:?}"),
                fix: "replace with a differentiable equivalent or gate behind a stop-gradient barrier".into(),
            });
        }
    };
    Ok(AdjointContrib {
        child: operand.clone(),
        adjoint: dx,
    })
}

/// Adjoint for `Expr::Fma { a, b, c }` = `a * b + c`.
///
/// `d/da = b * dz`, `d/db = a * dz`, `d/dc = dz`.
#[must_use]
pub fn fma_adjoints(a: &Expr, b: &Expr, c: &Expr, adjoint: &Expr) -> Vec<AdjointContrib> {
    let dz = adjoint.clone();
    vec![
        AdjointContrib {
            child: a.clone(),
            adjoint: Expr::mul(dz.clone(), b.clone()),
        },
        AdjointContrib {
            child: b.clone(),
            adjoint: Expr::mul(dz.clone(), a.clone()),
        },
        AdjointContrib {
            child: c.clone(),
            adjoint: dz,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{BinOp, Expr, UnOp};

    #[test]
    fn test_binop_add_adjoint() {
        let l = Expr::var("l");
        let r = Expr::var("r");
        let adj = Expr::var("adj");
        let contribs = binop_adjoints(BinOp::Add, &l, &r, &adj).unwrap();
        assert_eq!(contribs.len(), 2);
        assert_eq!(contribs[0].child, l);
        assert_eq!(contribs[0].adjoint, adj);
        assert_eq!(contribs[1].child, r);
        assert_eq!(contribs[1].adjoint, adj);
    }

    #[test]
    fn test_binop_mul_adjoint() {
        let l = Expr::var("l");
        let r = Expr::var("r");
        let adj = Expr::var("adj");
        let contribs = binop_adjoints(BinOp::Mul, &l, &r, &adj).unwrap();
        assert_eq!(contribs.len(), 2);
        assert_eq!(contribs[0].adjoint, Expr::mul(adj.clone(), r));
        assert_eq!(contribs[1].adjoint, Expr::mul(adj, l));
    }

    #[test]
    fn test_binop_not_differentiable() {
        let l = Expr::var("l");
        let r = Expr::var("r");
        let adj = Expr::var("adj");
        let err =
            binop_adjoints(BinOp::BitAnd, &l, &r, &adj).expect_err("BitAnd is not differentiable");
        assert!(
            matches!(err, AutodiffError::NotDifferentiable { ref op, .. } if op == "BinOp::BitAnd"),
            "non-differentiable binop error: {err:?}"
        );
    }

    #[test]
    fn test_unop_negate_adjoint() {
        let op = Expr::var("op");
        let adj = Expr::var("adj");
        let contrib = unop_adjoint(&UnOp::Negate, &op, &adj).unwrap();
        assert_eq!(contrib.child, op);
        assert!(matches!(
            contrib.adjoint,
            Expr::UnOp {
                op: UnOp::Negate,
                ..
            }
        ));
    }

    #[test]
    fn test_unop_exp_adjoint() {
        let op = Expr::var("op");
        let adj = Expr::var("adj");
        let contrib = unop_adjoint(&UnOp::Exp, &op, &adj).unwrap();
        assert!(matches!(
            contrib.adjoint,
            Expr::BinOp { op: BinOp::Mul, .. }
        ));
    }

    #[test]
    fn test_unop_not_differentiable() {
        let op = Expr::var("op");
        let adj = Expr::var("adj");
        let err = unop_adjoint(&UnOp::Floor, &op, &adj).expect_err("Floor is not differentiable");
        assert!(
            matches!(err, AutodiffError::NotDifferentiable { .. }),
            "non-differentiable unop error: {err:?}"
        );
    }

    #[test]
    fn test_fma_adjoint() {
        let a = Expr::var("a");
        let b = Expr::var("b");
        let c = Expr::var("c");
        let adj = Expr::var("adj");
        let contribs = fma_adjoints(&a, &b, &c, &adj);
        assert_eq!(contribs.len(), 3);
        assert_eq!(contribs[0].adjoint, Expr::mul(adj.clone(), b));
        assert_eq!(contribs[1].adjoint, Expr::mul(adj.clone(), a));
        assert_eq!(contribs[2].adjoint, adj);
    }
}
