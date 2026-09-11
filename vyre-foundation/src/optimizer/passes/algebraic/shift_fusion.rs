//! Shift identities and chained-shift fusion for `Shl` and `Shr`.
//!
//! Constant folding and strength reduction both reach a shift whose operands
//! are literals, and both must answer the same question the same way, so the
//! answer lives here once.

use crate::ir::{BinOp, Expr};

/// Reduce a `Shl` or `Shr` whose operands make the result knowable: a zero
/// left operand, a shift by zero, or a chain of two literal shifts in the
/// same direction.
pub(crate) fn reduce_shift(op: BinOp, left: &Expr, right: &Expr) -> Option<Expr> {
    if matches!(left, Expr::LitU32(0) | Expr::LitI32(0)) {
        return Some(left.clone());
    }
    if matches!(right, Expr::LitU32(0)) {
        return Some(left.clone());
    }
    let Expr::BinOp {
        op: inner_op,
        left: x,
        right: inner_shift,
    } = left
    else {
        return None;
    };
    if *inner_op != op {
        return None;
    }
    let (Expr::LitU32(a), Expr::LitU32(b)) = (inner_shift.as_ref(), right) else {
        return None;
    };
    // Target text masks a shift count with `& 31`, so each shift moves `count & 31` bits.
    let total = (a & 31) + (b & 31);
    // V094 rejects a shift operand that is not `u32`, so no sign bit can be replicated and
    // every bit is gone once the counts reach the width. A single fused shift cannot say
    // that, because its own count is masked as well and `x << 32` would emit `x`.
    if total > 31 {
        return Some(Expr::u32(0));
    }
    Some(Expr::BinOp {
        op,
        left: x.clone(),
        right: Box::new(Expr::u32(total)),
    })
}

#[cfg(test)]
mod tests {
    use super::reduce_shift;
    use crate::ir::{BinOp, Expr};

    fn evaluate(expr: &Expr, x: u32) -> u32 {
        match expr {
            Expr::LitU32(value) => *value,
            Expr::Var(name) if &**name == "x" => x,
            Expr::BinOp { op, left, right } => {
                let (left, right) = (evaluate(left, x), evaluate(right, x));
                match op {
                    BinOp::Shl => left << (right & 31),
                    BinOp::Shr => left >> (right & 31),
                    other => panic!("the shift evaluator does not model {other:?}"),
                }
            }
            other => panic!("the shift evaluator does not model {other:?}"),
        }
    }

    #[test]
    fn a_fused_shift_chain_agrees_with_the_chain_it_replaces() {
        let operands = [0u32, 1, 2, 0x8000_0001, 0xFFFF_FFFF, 0x0F0F_0F0F];
        for op in [BinOp::Shl, BinOp::Shr] {
            for inner_count in 0u32..=40 {
                for outer_count in 0u32..=40 {
                    let inner = Expr::BinOp {
                        op,
                        left: Box::new(Expr::var("x")),
                        right: Box::new(Expr::u32(inner_count)),
                    };
                    let outer = Expr::u32(outer_count);
                    let chain = Expr::BinOp {
                        op,
                        left: Box::new(inner.clone()),
                        right: Box::new(outer.clone()),
                    };
                    let Some(reduced) = reduce_shift(op, &inner, &outer) else {
                        panic!(
                            "declined to reduce a literal {op:?} chain {inner_count} then \
                             {outer_count}"
                        );
                    };
                    for x in operands {
                        assert_eq!(
                            evaluate(&reduced, x),
                            evaluate(&chain, x),
                            "{op:?} chain {inner_count} then {outer_count} disagrees for {x}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_shift_of_a_different_direction_does_not_fuse() {
        let inner = Expr::BinOp {
            op: BinOp::Shl,
            left: Box::new(Expr::var("x")),
            right: Box::new(Expr::u32(3)),
        };
        assert_eq!(reduce_shift(BinOp::Shr, &inner, &Expr::u32(4)), None);
    }
}
