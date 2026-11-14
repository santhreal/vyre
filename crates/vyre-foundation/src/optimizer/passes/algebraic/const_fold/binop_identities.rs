// Algebraic identity simplifications for binary operators.
//
// Each rule encodes an algebraic identity that eliminates redundant GPU
// instructions. Rules fire when one operand is a literal identity/annihilator
// element, or when both operands are syntactically identical.
//
// Contributors: add new rules to the appropriate section below. Each section
// is delimited by a Unicode box-drawing comment header.

use crate::ir::Expr;
use crate::optimizer::algebraic_rules::{binop_identity_replacement, IdentityReplacement};
use crate::region_ssa::ScalarLiteral;

/// True iff `expr` is an integer literal (u32 or i32). Used by the
/// distributive expansion rule to gate the rewrite so
/// it only fires when one of the new sub-multiplications will fold
/// in the next const-fold pass.
fn lit_int(expr: &Expr) -> Option<()> {
    match expr {
        Expr::LitU32(_) | Expr::LitI32(_) => Some(()),
        _ => None,
    }
}

/// True for expressions that are pure and side-effect-free.
/// Used to guard reflexive *idempotent* folding (`x & x → x`,
/// `min(x, x) → x`) so we never elide a Load whose ordering matters.
/// Those folds return the operand itself, so they are sound for every
/// type, including a float `x` that is `NaN`, where `min(NaN, NaN)`
/// is `NaN` and the rule yields `x` unchanged.
pub(super) fn is_simple_pure(expr: &Expr) -> bool {
    matches!(
        crate::visit::expr_magnitude(expr),
        crate::visit::ExprMagnitude::HostFact | crate::visit::ExprMagnitude::Binding(_)
    )
}

/// True for operands where reflexive *comparison* folding (`x == x → true`,
/// `x != x → false`, `x < x → false`, …) is sound.
///
/// This is strictly stronger than [`is_simple_pure`]. Comparison folds
/// collapse the operand to a constant `bool`, so they are only valid
/// when reflexivity actually holds, which fails for IEEE-754 `NaN`:
/// `NaN == NaN` is `false` and `NaN != NaN` is `true`, and both the
/// reference oracle (`vyre-reference::binop_f32`) and every target
/// emitter honor that. `x != x` is the canonical
/// hand-rolled NaN test, so folding it away type-blind is a real
/// miscompile, not a corner case.
///
/// `Var` is therefore EXCLUDED: const_fold is type-blind, so a bare
/// `Var` may bind a float that is `NaN` at runtime. `LitF32` is excluded
/// too, two float literals are folded by the typed literal evaluator,
/// which computes `NaN == NaN → false` correctly; this structural rule
/// must not shortcut it. (A `LitF32(NaN)` would also fail the
/// `left == right` precondition under the IR's native-`f32` equality,
/// but excluding it here keeps the contract explicit rather than
/// load-bearing on that subtlety.) Integer/bool literals and the u32
/// lane/workgroup builtins are always reflexive-safe.
///
/// Recovering reflexive comparison folding for a *provably-integer*
/// `Var` requires the type-fact cache
/// (`optimizer::fact_cache::type_facts`) to be threaded into
/// const_fold so the operand's `DataType` can be proven non-float
/// before the fold fires; until then this declines `Var`, per the
/// soundness-over-reach contract.
fn is_reflexive_cmp_safe(expr: &Expr) -> bool {
    matches!(
        crate::visit::expr_magnitude(expr),
        crate::visit::ExprMagnitude::HostFact
    ) && !matches!(expr, Expr::LitF32(_))
}

/// Algebraic identity simplifications for binary operators.
/// These rewrites are always valid and don't require literal operands  -
/// they fire when one operand is a literal identity/annihilator element.
///
/// Critical for GPU kernels where loop unrolling and index arithmetic
/// generate patterns like `base + 0`, `stride * 1`, or `mask & 0`.
#[allow(
    clippy::too_many_lines,
    clippy::match_same_arms,
    reason = "binary identity table intentionally groups algebraic laws by operator family"
)]
pub(super) fn simplify_binop(op: crate::ir::BinOp, left: &Expr, right: &Expr) -> Option<Expr> {
    use crate::ir::BinOp;

    // Helper: is this a u32/i32/f32 zero?
    let is_zero = |e: &Expr| match e {
        Expr::LitU32(0) | Expr::LitI32(0) => true,
        Expr::LitF32(v) => lit_f32_eq(*v, 0.0),
        _ => false,
    };
    // Helper: is this a u32/i32/f32 one?
    let is_one = |e: &Expr| match e {
        Expr::LitU32(1) | Expr::LitI32(1) => true,
        Expr::LitF32(v) => lit_f32_eq(*v, 1.0),
        _ => false,
    };

    if let Some(replacement) = binop_identity_replacement(
        op,
        left == right && is_simple_pure(left),
        expr_scalar_literal(left),
        expr_scalar_literal(right),
    ) {
        return Some(match replacement {
            IdentityReplacement::Left => left.clone(),
            IdentityReplacement::Right => right.clone(),
        });
    }

    match op {
        BinOp::Add => reassociate_int_lit(
            BinOp::Add,
            left,
            right,
            u32::checked_add,
            i32::checked_add,
            Expr::add,
        )
        .or_else(|| factor_common_mul(left, right, u32::checked_add, i32::checked_add)),

        BinOp::Sub => {
            if left == right && is_reflexive_cmp_safe(left) {
                return Some(Expr::u32(0));
            }
            factor_common_mul(left, right, u32::checked_sub, i32::checked_sub)
        }

        BinOp::Mul => {
            if let Some(folded) = reassociate_int_lit(
                BinOp::Mul,
                left,
                right,
                u32::checked_mul,
                i32::checked_mul,
                Expr::mul,
            )
            .or_else(|| distribute_mul_int(left, right))
            {
                return Some(folded);
            }
            if is_one(right) {
                return Some(left.clone());
            }
            if is_one(left) {
                return Some(right.clone());
            }
            let is_neg_one = |e: &Expr| match e {
                Expr::LitI32(-1) => true,
                Expr::LitF32(v) => lit_f32_eq(*v, -1.0),
                _ => false,
            };
            if is_neg_one(right) {
                return Some(Expr::negate(left.clone()));
            }
            if is_neg_one(left) {
                return Some(Expr::negate(right.clone()));
            }
            if matches!(right, Expr::LitU32(0) | Expr::LitI32(0)) {
                return Some(right.clone());
            }
            if matches!(left, Expr::LitU32(0) | Expr::LitI32(0)) {
                return Some(left.clone());
            }
            None
        }

        // x / 1 → x
        BinOp::Div if is_one(right) => Some(left.clone()),
        // Literal i32 / -1 is folded by the shared literal evaluator when
        // defined. Non-literal i32 can be MIN at runtime, where target-text division
        // overflows, so do not rewrite it to negate.
        BinOp::Div if matches!(right, Expr::LitI32(-1)) => None,

        // ─── Reciprocal sqrt fusion ──────────────────────────
        // 1.0 / sqrt(x) → rsqrt(x)
        // Maps to a single GPU SFU instruction (4 cycles) instead of
        // sqrt + fdiv (8+32 = 40 cycles). Critical for normalization
        // kernels (LayerNorm, RMSNorm).
        BinOp::Div if matches!(left, Expr::LitF32(v) if lit_f32_eq(*v, 1.0)) => {
            if let Expr::UnOp {
                op: crate::ir::UnOp::Sqrt,
                operand,
            } = right
            {
                return Some(Expr::UnOp {
                    op: crate::ir::UnOp::InverseSqrt,
                    operand: operand.clone(),
                });
            }
            None
        }

        // ─── Trig / identity peepholes ────────────────────────
        BinOp::Div => {
            // sin(x) / cos(x) → tan(x)  (saves 1 transcendental)
            if let (
                Expr::UnOp {
                    op: crate::ir::UnOp::Sin,
                    operand: sin_arg,
                },
                Expr::UnOp {
                    op: crate::ir::UnOp::Cos,
                    operand: cos_arg,
                },
            ) = (left, right)
            {
                if sin_arg == cos_arg {
                    return Some(Expr::UnOp {
                        op: crate::ir::UnOp::Tan,
                        operand: sin_arg.clone(),
                    });
                }
            }
            // x / x is NOT folded to 1: it is unsound when x can be 0.
            // The reference oracle defines unsigned `0 / 0` as `u32::MAX`
            // (`div_u32`), not 1, and the emitters' guarded division
            // agrees, so folding to 1 diverges from BOTH
            // for u32 x=0; signed `0 / 0` errors in the oracle
            // (`div_i32`). const_fold is type- and value-blind here, so it
            // cannot prove x != 0 (or that x is unsigned). Decline.
            // Literal `k / k` still folds via the typed literal evaluator
            // (which reproduces the oracle's `u32::MAX` for 0/0).
            None
        }

        // x & 0 → 0 (integer bitwise annihilator)
        BinOp::BitAnd if matches!(right, Expr::LitU32(0) | Expr::LitI32(0)) => Some(right.clone()),
        BinOp::BitAnd if matches!(left, Expr::LitU32(0) | Expr::LitI32(0)) => Some(left.clone()),

        // x | 0 → x,  0 | x → x (integer bitwise identity)
        BinOp::BitOr if matches!(right, Expr::LitU32(0) | Expr::LitI32(0)) => Some(left.clone()),
        BinOp::BitOr if matches!(left, Expr::LitU32(0) | Expr::LitI32(0)) => Some(right.clone()),

        // x ^ 0 → x,  0 ^ x → x (xor identity)
        BinOp::BitXor if matches!(right, Expr::LitU32(0) | Expr::LitI32(0)) => Some(left.clone()),
        BinOp::BitXor if matches!(left, Expr::LitU32(0) | Expr::LitI32(0)) => Some(right.clone()),

        // Shift identities and chained-shift fusion have one owner.
        BinOp::Shl | BinOp::Shr => {
            crate::optimizer::passes::algebraic::shift_fusion::reduce_shift(op, left, right)
        }

        // ─── Self-operand identities ─────────────────────────
        // These fire on loop-unrolling artifacts and duplicated
        // index expressions that produce syntactically identical
        // operands.  Each rule removes an entire instruction from
        // the GPU shader.

        // x ^ x → 0  (xor self-inverse)
        BinOp::BitXor if left == right => Some(Expr::u32(0)),
        // x & x → x  (bitwise idempotent)
        BinOp::BitAnd if left == right => Some(left.clone()),
        // x | x → x  (bitwise idempotent)
        BinOp::BitOr if left == right => Some(left.clone()),

        // ─── Comparison self-identities ──────────────────────
        // x == x → true,  x != x → false   -  only when `x` is provably
        // non-float AND value-deterministic. Two guards apply:
        //   1. Relaxed memory ordering: two `Load` reads of the same
        //      buffer can observe distinct values, so `Eq(Load, Load)`
        //      must not fold.
        //   2. IEEE-754 NaN: for a float `x` that is `NaN` at runtime,
        //      `x == x` is *false* and `x != x` is *true* (the
        //      reference oracle and every target emitter
        //      both honor this), so folding type-blind would miscompile
        //      the canonical hand-rolled NaN check.
        // `is_reflexive_cmp_safe` admits only integer/bool literals and
        // the u32 builtins, it EXCLUDES `Var` (unknown type, may be a
        // float NaN) and `LitF32`. See its doc for how a provably-integer
        // `Var` could be recovered via the type-fact cache.
        BinOp::Eq | BinOp::Le | BinOp::Ge if left == right && is_reflexive_cmp_safe(left) => {
            Some(Expr::bool(true))
        }
        BinOp::Ne | BinOp::Lt | BinOp::Gt if left == right && is_reflexive_cmp_safe(left) => {
            Some(Expr::bool(false))
        }

        BinOp::Mod if is_one(right) => Some(Expr::u32(0)),

        BinOp::Min | BinOp::Max if left == right => Some(left.clone()),
        BinOp::Min if matches!(right, Expr::LitU32(u32::MAX)) => Some(left.clone()),
        BinOp::Min if matches!(left, Expr::LitU32(u32::MAX)) => Some(right.clone()),
        BinOp::Max if matches!(right, Expr::LitU32(0)) => Some(left.clone()),
        BinOp::Max if matches!(left, Expr::LitU32(0)) => Some(right.clone()),
        BinOp::Min if matches!(left, Expr::LitU32(0)) || matches!(right, Expr::LitU32(0)) => {
            Some(Expr::u32(0))
        }
        BinOp::Max
            if matches!(left, Expr::LitU32(u32::MAX))
                || matches!(right, Expr::LitU32(u32::MAX)) =>
        {
            Some(Expr::u32(u32::MAX))
        }

        BinOp::Lt if matches!(right, Expr::LitU32(0)) || matches!(left, Expr::LitU32(u32::MAX)) => {
            Some(Expr::bool(false))
        }
        BinOp::Gt if matches!(right, Expr::LitU32(u32::MAX)) || matches!(left, Expr::LitU32(0)) => {
            Some(Expr::bool(false))
        }
        BinOp::Ge if matches!(right, Expr::LitU32(0)) || matches!(left, Expr::LitU32(u32::MAX)) => {
            Some(Expr::bool(true))
        }
        BinOp::Le if matches!(right, Expr::LitU32(u32::MAX)) || matches!(left, Expr::LitU32(0)) => {
            Some(Expr::bool(true))
        }

        BinOp::WrappingAdd | BinOp::SaturatingAdd | BinOp::WrappingSub | BinOp::SaturatingSub
            if is_zero(right) =>
        {
            Some(left.clone())
        }
        BinOp::WrappingAdd | BinOp::SaturatingAdd if is_zero(left) => Some(right.clone()),
        BinOp::WrappingSub | BinOp::SaturatingSub if left == right => Some(Expr::u32(0)),
        BinOp::SaturatingMul if is_one(right) => Some(left.clone()),
        BinOp::SaturatingMul if is_one(left) => Some(right.clone()),
        BinOp::SaturatingMul if is_zero(left) || is_zero(right) => Some(Expr::u32(0)),

        BinOp::And if matches!(left, Expr::LitBool(true)) => Some(right.clone()),
        BinOp::And if matches!(right, Expr::LitBool(true)) => Some(left.clone()),
        BinOp::And
            if matches!(left, Expr::LitBool(false)) || matches!(right, Expr::LitBool(false)) =>
        {
            Some(Expr::bool(false))
        }
        BinOp::Or
            if matches!(left, Expr::LitBool(true)) || matches!(right, Expr::LitBool(true)) =>
        {
            Some(Expr::bool(true))
        }
        BinOp::Or if matches!(left, Expr::LitBool(false)) => Some(right.clone()),
        BinOp::Or if matches!(right, Expr::LitBool(false)) => Some(left.clone()),
        BinOp::And | BinOp::Or if left == right => Some(left.clone()),

        BinOp::And if is_not_of(right, left) || is_not_of(left, right) => Some(Expr::bool(false)),
        BinOp::Or if is_not_of(right, left) || is_not_of(left, right) => Some(Expr::bool(true)),
        BinOp::And if absorbs_binop(BinOp::Or, right, left) => Some(left.clone()),
        BinOp::And if absorbs_binop(BinOp::Or, left, right) => Some(right.clone()),
        BinOp::Or if absorbs_binop(BinOp::And, right, left) => Some(left.clone()),
        BinOp::Or if absorbs_binop(BinOp::And, left, right) => Some(right.clone()),

        BinOp::BitAnd if matches!(right, Expr::LitU32(u32::MAX)) => Some(left.clone()),
        BinOp::BitAnd if matches!(left, Expr::LitU32(u32::MAX)) => Some(right.clone()),
        BinOp::BitOr
            if matches!(left, Expr::LitU32(u32::MAX))
                || matches!(right, Expr::LitU32(u32::MAX)) =>
        {
            Some(Expr::u32(u32::MAX))
        }
        BinOp::BitXor if matches!(right, Expr::LitU32(u32::MAX)) => Some(Expr::UnOp {
            op: crate::ir::UnOp::BitNot,
            operand: Box::new(left.clone()),
        }),
        BinOp::BitXor if matches!(left, Expr::LitU32(u32::MAX)) => Some(Expr::UnOp {
            op: crate::ir::UnOp::BitNot,
            operand: Box::new(right.clone()),
        }),

        _ => None,
    }
}

fn is_not_of(candidate: &Expr, target: &Expr) -> bool {
    matches!(
        candidate,
        Expr::UnOp {
            op: crate::ir::UnOp::LogicalNot,
            operand,
        } if operand.as_ref() == target
    )
}

fn absorbs_binop(inner: crate::ir::BinOp, candidate: &Expr, target: &Expr) -> bool {
    matches!(
        candidate,
        Expr::BinOp { op, left, right } if *op == inner && (left.as_ref() == target || right.as_ref() == target)
    )
}

fn fold_int_pair(
    k1: &Expr,
    k2: &Expr,
    u32_op: fn(u32, u32) -> Option<u32>,
    i32_op: fn(i32, i32) -> Option<i32>,
) -> Option<Expr> {
    match (k1, k2) {
        (Expr::LitU32(a), Expr::LitU32(b)) => u32_op(*a, *b).map(Expr::u32),
        (Expr::LitI32(a), Expr::LitI32(b)) => i32_op(*a, *b).map(Expr::i32),
        _ => None,
    }
}

fn reassociate_int_lit(
    op: crate::ir::BinOp,
    left: &Expr,
    right: &Expr,
    u32_op: fn(u32, u32) -> Option<u32>,
    i32_op: fn(i32, i32) -> Option<i32>,
    build: fn(Expr, Expr) -> Expr,
) -> Option<Expr> {
    let Expr::BinOp {
        op: inner_op,
        left: l,
        right: r,
    } = left
    else {
        return None;
    };
    if *inner_op != op {
        return None;
    }
    fold_int_pair(r, right, u32_op, i32_op)
        .map(|k| build((**l).clone(), k))
        .or_else(|| fold_int_pair(l, right, u32_op, i32_op).map(|k| build((**r).clone(), k)))
}

fn factor_common_mul(
    left: &Expr,
    right: &Expr,
    u32_op: fn(u32, u32) -> Option<u32>,
    i32_op: fn(i32, i32) -> Option<i32>,
) -> Option<Expr> {
    let (
        Expr::BinOp {
            op: crate::ir::BinOp::Mul,
            left: la,
            right: lk,
        },
        Expr::BinOp {
            op: crate::ir::BinOp::Mul,
            left: ra,
            right: rk,
        },
    ) = (left, right)
    else {
        return None;
    };
    (la == ra)
        .then(|| fold_int_pair(lk, rk, u32_op, i32_op).map(|k| Expr::mul((**la).clone(), k)))
        .flatten()
}

fn distribute_mul_int(left: &Expr, right: &Expr) -> Option<Expr> {
    use crate::ir::BinOp;
    let build_add_sub = |op: BinOp, l: Expr, r: Expr| match op {
        BinOp::Add => Expr::add(l, r),
        _ => Expr::BinOp {
            op,
            left: Box::new(l),
            right: Box::new(r),
        },
    };
    if lit_int(left).is_some() {
        if let Expr::BinOp {
            op: op @ (BinOp::Add | BinOp::Sub),
            left: a,
            right: b,
        } = right
        {
            if lit_int(a).is_some() || lit_int(b).is_some() {
                return Some(build_add_sub(
                    *op,
                    Expr::mul(left.clone(), (**a).clone()),
                    Expr::mul(left.clone(), (**b).clone()),
                ));
            }
        }
    }
    if lit_int(right).is_some() {
        if let Expr::BinOp {
            op: op @ (BinOp::Add | BinOp::Sub),
            left: a,
            right: b,
        } = left
        {
            if lit_int(a).is_some() || lit_int(b).is_some() {
                return Some(build_add_sub(
                    *op,
                    Expr::mul((**a).clone(), right.clone()),
                    Expr::mul((**b).clone(), right.clone()),
                ));
            }
        }
    }
    None
}

fn expr_scalar_literal(expr: &Expr) -> Option<ScalarLiteral> {
    match expr {
        Expr::LitU32(value) => Some(ScalarLiteral::U32(*value)),
        Expr::LitI32(value) => Some(ScalarLiteral::I32(*value)),
        Expr::LitF32(value) => Some(ScalarLiteral::F32(*value)),
        Expr::LitBool(value) => Some(ScalarLiteral::Bool(*value)),
        _ => None,
    }
}

// ─── stronger range fold ─────────────────────────────
// Mod(x, N) where x.max < N -> x
// We use a tiny single-block lookbehind to find if `x` was defined as `LitU32(c)` where `c < N`.
pub(super) fn fold_mod_lookbehind(
    nodes: &[crate::ir::Node],
    changed: &mut bool,
) -> Vec<crate::ir::Node> {
    use crate::ir::{Expr, Node};
    use rustc_hash::FxHashMap;
    use std::sync::Arc;

    let mut out = Vec::with_capacity(nodes.len());
    let mut local_lits: FxHashMap<crate::ir::Ident, u32> = FxHashMap::default();

    for node in nodes {
        match node {
            Node::Let { name, value } => {
                let new_value = rewrite_expr_for_mod(value, &local_lits, changed);
                if let Expr::LitU32(c) = &new_value {
                    local_lits.insert(name.clone(), *c);
                } else if let Expr::LitI32(c) = &new_value {
                    if *c >= 0 {
                        if let Ok(value) = u32::try_from(*c) {
                            local_lits.insert(name.clone(), value);
                        }
                    }
                }
                out.push(Node::Let {
                    name: name.clone(),
                    value: new_value,
                });
            }
            Node::Assign { name, value } => {
                local_lits.remove(name);
                out.push(Node::Assign {
                    name: name.clone(),
                    value: rewrite_expr_for_mod(value, &local_lits, changed),
                });
            }
            Node::Store {
                buffer,
                index,
                value,
            } => {
                out.push(Node::Store {
                    buffer: buffer.clone(),
                    index: rewrite_expr_for_mod(index, &local_lits, changed),
                    value: rewrite_expr_for_mod(value, &local_lits, changed),
                });
            }
            Node::If {
                cond,
                then,
                otherwise,
            } => {
                out.push(Node::If {
                    cond: rewrite_expr_for_mod(cond, &local_lits, changed),
                    then: fold_mod_lookbehind(then, changed),
                    otherwise: fold_mod_lookbehind(otherwise, changed),
                });
            }
            Node::Loop {
                var,
                from,
                to,
                body,
            } => {
                out.push(Node::Loop {
                    var: var.clone(),
                    from: rewrite_expr_for_mod(from, &local_lits, changed),
                    to: rewrite_expr_for_mod(to, &local_lits, changed),
                    body: fold_mod_lookbehind(body, changed),
                });
            }
            Node::Block(body) => {
                out.push(Node::Block(fold_mod_lookbehind(body, changed)));
            }
            Node::Region {
                generator,
                source_region,
                body,
            } => {
                out.push(Node::Region {
                    generator: generator.clone(),
                    source_region: source_region.clone(),
                    body: Arc::new(fold_mod_lookbehind(body, changed)),
                });
            }
            Node::Trap { address, tag } => {
                out.push(Node::Trap {
                    address: Box::new(rewrite_expr_for_mod(address, &local_lits, changed)),
                    tag: tag.clone(),
                });
            }
            _ => out.push(node.clone()),
        }
    }
    out
}

fn rewrite_expr_for_mod(
    expr: &crate::ir::Expr,
    local_lits: &rustc_hash::FxHashMap<crate::ir::Ident, u32>,
    changed: &mut bool,
) -> crate::ir::Expr {
    use crate::ir::{BinOp, Expr};
    let mut transformer = |e: &Expr| -> Option<Expr> {
        if let Expr::BinOp {
            op: BinOp::Mod,
            left,
            right,
        } = e
        {
            if let (Expr::Var(x), Expr::LitU32(n)) = (left.as_ref(), right.as_ref()) {
                if let Some(&c) = local_lits.get(x) {
                    if c < *n {
                        *changed = true;
                        return Some(Expr::Var(x.clone()));
                    }
                }
            }
        }
        None
    };
    let cow = crate::optimizer::rewrite::rewrite_expr(expr, &mut transformer);
    cow.into_owned()
}

#[inline]
fn lit_f32_eq(value: f32, expected: f32) -> bool {
    value.to_bits() == expected.to_bits()
}
