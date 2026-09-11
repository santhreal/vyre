//! Constant-fold binary identity contracts.

use super::super::*;
use crate::ir::Expr;

mod reflexive_comparison_and_self_identities {
    use super::*;

    // ──── Reflexive comparison folds: `Var` must NOT fold (float-NaN) ────
    //
    // const_fold is type-blind: a bare `Var` may bind a float that is `NaN`
    // at runtime, where `x == x` is *false* and `x != x` is *true* under
    // IEEE-754 (the `vyre-reference::binop_f32` oracle and every target
    // emitter honor this). Folding `Var cmp Var` to a
    // bool literal type-blind miscompiles the canonical hand-rolled NaN
    // check, so the `is_reflexive_cmp_safe` guard rejects `Var`. These six
    // tests pin the decline; before the guard was tightened they asserted
    // the (unsound) folded bool literal.

    #[test]
    fn eq_self_var_does_not_fold() {
        let x = Expr::var("x");
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::Eq,
                left: Box::new(x.clone()),
                right: Box::new(x)
            }),
            None,
            "Eq(Var, Var) must not fold: x may be float NaN where x == x is false"
        );
    }
    #[test]
    fn ne_self_var_does_not_fold() {
        let x = Expr::var("x");
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::Ne,
                left: Box::new(x.clone()),
                right: Box::new(x)
            }),
            None,
            "Ne(Var, Var) must not fold: x may be float NaN where x != x is true"
        );
    }
    #[test]
    fn lt_self_var_does_not_fold() {
        let x = Expr::var("x");
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::Lt,
                left: Box::new(x.clone()),
                right: Box::new(x)
            }),
            None,
            "Lt(Var, Var) must not fold: float NaN comparisons are all false"
        );
    }
    #[test]
    fn gt_self_var_does_not_fold() {
        let x = Expr::var("x");
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::Gt,
                left: Box::new(x.clone()),
                right: Box::new(x)
            }),
            None,
            "Gt(Var, Var) must not fold: float NaN comparisons are all false"
        );
    }
    #[test]
    fn le_self_var_does_not_fold() {
        let x = Expr::var("x");
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::Le,
                left: Box::new(x.clone()),
                right: Box::new(x)
            }),
            None,
            "Le(Var, Var) must not fold: NaN <= NaN is false, not true"
        );
    }
    #[test]
    fn ge_self_var_does_not_fold() {
        let x = Expr::var("x");
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::Ge,
                left: Box::new(x.clone()),
                right: Box::new(x)
            }),
            None,
            "Ge(Var, Var) must not fold: NaN >= NaN is false, not true"
        );
    }

    // ──── Reflexive comparison folds: provably-u32 builtins DO fold ────
    //
    // `InvocationId` is a deterministic u32 lane index that can never be a
    // float NaN, so reflexive comparison folding stays sound and active for
    // it. These prove the sound path is preserved (not blanket-disabled).

    #[test]
    fn eq_self_invocation_id_folds_true() {
        let g = Expr::InvocationId { axis: 0 };
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::Eq,
                left: Box::new(g.clone()),
                right: Box::new(g)
            }),
            Some(Expr::bool(true)),
            "Eq(InvocationId, InvocationId) is a sound reflexive fold (u32, never NaN)"
        );
    }
    #[test]
    fn ne_self_invocation_id_folds_false() {
        let g = Expr::InvocationId { axis: 0 };
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::Ne,
                left: Box::new(g.clone()),
                right: Box::new(g)
            }),
            Some(Expr::bool(false)),
            "Ne(InvocationId, InvocationId) folds to false (u32, never NaN)"
        );
    }
    #[test]
    fn lt_self_literal_folds_false() {
        // Integer-literal self-comparison stays foldable.
        let k = Expr::u32(7);
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::Lt,
                left: Box::new(k.clone()),
                right: Box::new(k)
            }),
            Some(Expr::bool(false)),
            "Lt(7, 7) folds to false"
        );
    }

    // ──── binop_identities: mod/min/max/div ────────────────────

    #[test]
    fn mod_one_is_zero() {
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::Mod,
                left: Box::new(Expr::var("x")),
                right: Box::new(Expr::u32(1))
            }),
            Some(Expr::u32(0))
        );
    }
    #[test]
    fn mod_self_var_does_not_fold() {
        // x % x must NOT fold to 0: const_fold is type/value-blind, and
        // signed `0 % 0` errors in the oracle (rem_i32). Folding to 0
        // fabricates a value where the i32 program is undefined.
        let x = Expr::var("x");
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::Mod,
                left: Box::new(x.clone()),
                right: Box::new(x)
            }),
            None,
            "Mod(Var, Var) must not fold: signed 0 % 0 is undefined in the oracle"
        );
    }
    #[test]
    fn mod_literal_self_still_folds() {
        // The typed literal evaluator still folds concrete `k % k`, and
        // reproduces the oracle's unsigned `0 % 0 = 0` (rem_u32).
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::Mod,
                left: Box::new(Expr::u32(6)),
                right: Box::new(Expr::u32(6))
            }),
            Some(Expr::u32(0)),
            "6 % 6 folds to 0 via the literal evaluator"
        );
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::Mod,
                left: Box::new(Expr::u32(0)),
                right: Box::new(Expr::u32(0))
            }),
            Some(Expr::u32(0)),
            "unsigned 0 % 0 folds to 0, matching the rem_u32 oracle"
        );
    }
    #[test]
    fn min_self_is_self() {
        let x = Expr::var("x");
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::Min,
                left: Box::new(x.clone()),
                right: Box::new(x.clone())
            }),
            Some(x)
        );
    }
    #[test]
    fn max_self_is_self() {
        let x = Expr::var("x");
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::Max,
                left: Box::new(x.clone()),
                right: Box::new(x.clone())
            }),
            Some(x)
        );
    }
    #[test]
    fn div_self_var_does_not_fold() {
        // x / x must NOT fold to 1: unsigned `0 / 0` is u32::MAX in the
        // oracle (div_u32) and every guarded target emitter, not 1, and
        // signed `0 / 0` errors (div_i32). const_fold cannot prove x != 0
        // or that x is unsigned, so folding to 1 is a miscompile for x=0.
        let x = Expr::var("x");
        assert_eq!(
            fold_expr(&Expr::div(x.clone(), x)),
            None,
            "Div(Var, Var) must not fold: 0 / 0 is u32::MAX (u32) or undefined (i32), never 1"
        );
    }
    #[test]
    fn div_literal_self_still_folds() {
        // Concrete `k / k` still folds, and `0 / 0` reproduces the oracle's
        // unsigned u32::MAX (div_u32) rather than the bogus 1.
        assert_eq!(
            fold_expr(&Expr::div(Expr::u32(6), Expr::u32(6))),
            Some(Expr::u32(1)),
            "6 / 6 folds to 1 via the literal evaluator"
        );
        assert_eq!(
            fold_expr(&Expr::div(Expr::u32(0), Expr::u32(0))),
            Some(Expr::u32(u32::MAX)),
            "unsigned 0 / 0 folds to u32::MAX, matching the div_u32 oracle"
        );
    }

    // ──── binop_identities: wrapping/saturating ────────────────

    #[test]
    fn wrapping_add_zero() {
        let x = Expr::var("x");
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::WrappingAdd,
                left: Box::new(x.clone()),
                right: Box::new(Expr::u32(0))
            }),
            Some(x)
        );
    }
    #[test]
    fn wrapping_sub_self() {
        let x = Expr::var("x");
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::WrappingSub,
                left: Box::new(x.clone()),
                right: Box::new(x)
            }),
            Some(Expr::u32(0))
        );
    }
    #[test]
    fn saturating_add_zero() {
        let x = Expr::var("x");
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::SaturatingAdd,
                left: Box::new(x.clone()),
                right: Box::new(Expr::u32(0))
            }),
            Some(x)
        );
    }
    #[test]
    fn saturating_sub_self() {
        let x = Expr::var("x");
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::SaturatingSub,
                left: Box::new(x.clone()),
                right: Box::new(x)
            }),
            Some(Expr::u32(0))
        );
    }
    #[test]
    fn saturating_mul_one() {
        let x = Expr::var("x");
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::SaturatingMul,
                left: Box::new(x.clone()),
                right: Box::new(Expr::u32(1))
            }),
            Some(x)
        );
    }
    #[test]
    fn saturating_mul_zero() {
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::SaturatingMul,
                left: Box::new(Expr::var("x")),
                right: Box::new(Expr::u32(0))
            }),
            Some(Expr::u32(0))
        );
    }

    // ──── binop_identities: logical boolean ────────────────────

    #[test]
    fn and_true_id() {
        let x = Expr::var("x");
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::And,
                left: Box::new(Expr::bool(true)),
                right: Box::new(x.clone())
            }),
            Some(x)
        );
    }
    #[test]
    fn and_false_ann() {
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::And,
                left: Box::new(Expr::bool(false)),
                right: Box::new(Expr::var("x"))
            }),
            Some(Expr::bool(false))
        );
    }
    #[test]
    fn or_true_ann() {
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::Or,
                left: Box::new(Expr::bool(true)),
                right: Box::new(Expr::var("x"))
            }),
            Some(Expr::bool(true))
        );
    }
    #[test]
    fn or_false_id() {
        let x = Expr::var("x");
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::Or,
                left: Box::new(Expr::bool(false)),
                right: Box::new(x.clone())
            }),
            Some(x)
        );
    }

    // ──── binop_identities: all-ones mask ──────────────────────

    #[test]
    fn bitand_all_ones() {
        let x = Expr::var("x");
        assert_eq!(
            fold_expr(&Expr::bitand(x.clone(), Expr::u32(u32::MAX))),
            Some(x)
        );
    }
    #[test]
    fn bitor_all_ones() {
        assert_eq!(
            fold_expr(&Expr::bitor(Expr::var("x"), Expr::u32(u32::MAX))),
            Some(Expr::u32(u32::MAX))
        );
    }

    // ──── chained-predicate boolean simplification ─────────

    #[test]
    fn and_x_not_x_is_false_contradiction() {
        let x = Expr::var("c");
        let not_x = Expr::UnOp {
            op: crate::ir::UnOp::LogicalNot,
            operand: Box::new(x.clone()),
        };
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::And,
                left: Box::new(x),
                right: Box::new(not_x)
            }),
            Some(Expr::bool(false))
        );
    }

    #[test]
    fn and_not_x_x_is_false_contradiction_left_not() {
        let x = Expr::var("c");
        let not_x = Expr::UnOp {
            op: crate::ir::UnOp::LogicalNot,
            operand: Box::new(x.clone()),
        };
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::And,
                left: Box::new(not_x),
                right: Box::new(x)
            }),
            Some(Expr::bool(false))
        );
    }

    #[test]
    fn or_x_not_x_is_true_tautology() {
        let x = Expr::var("c");
        let not_x = Expr::UnOp {
            op: crate::ir::UnOp::LogicalNot,
            operand: Box::new(x.clone()),
        };
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::Or,
                left: Box::new(x),
                right: Box::new(not_x)
            }),
            Some(Expr::bool(true))
        );
    }

    #[test]
    fn or_not_x_x_is_true_tautology_left_not() {
        let x = Expr::var("c");
        let not_x = Expr::UnOp {
            op: crate::ir::UnOp::LogicalNot,
            operand: Box::new(x.clone()),
        };
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::Or,
                left: Box::new(not_x),
                right: Box::new(x)
            }),
            Some(Expr::bool(true))
        );
    }

    #[test]
    fn absorption_and_over_or() {
        let x = Expr::var("x");
        let y = Expr::var("y");
        let or_xy = Expr::BinOp {
            op: crate::ir::BinOp::Or,
            left: Box::new(x.clone()),
            right: Box::new(y),
        };
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::And,
                left: Box::new(x.clone()),
                right: Box::new(or_xy)
            }),
            Some(x)
        );
    }

    #[test]
    fn absorption_or_over_and() {
        let x = Expr::var("x");
        let y = Expr::var("y");
        let and_xy = Expr::BinOp {
            op: crate::ir::BinOp::And,
            left: Box::new(x.clone()),
            right: Box::new(y),
        };
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::Or,
                left: Box::new(x.clone()),
                right: Box::new(and_xy)
            }),
            Some(x)
        );
    }

    #[test]
    fn reflexive_eq_on_load_does_not_fold() {
        // Adversarial: Eq(Load, Load) MUST NOT fold  -  repeated Loads can
        // observe distinct memory under relaxed ordering. The
        // is_simple_pure guard rejects Loads.
        let load = Expr::load("buf", Expr::u32(0));
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::Eq,
                left: Box::new(load.clone()),
                right: Box::new(load)
            }),
            None,
            "Eq(Load, Load) must not fold"
        );
    }

    // ──── range-based fold identities ──────────────────────

    #[test]
    fn min_with_u32_max_is_identity() {
        let x = Expr::var("x");
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::Min,
                left: Box::new(x.clone()),
                right: Box::new(Expr::u32(u32::MAX))
            }),
            Some(x.clone())
        );
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::Min,
                left: Box::new(Expr::u32(u32::MAX)),
                right: Box::new(x.clone())
            }),
            Some(x)
        );
    }
}

mod boundary_and_distribution_identities {
    use super::*;

    #[test]
    fn max_with_zero_is_identity() {
        let x = Expr::var("x");
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::Max,
                left: Box::new(x.clone()),
                right: Box::new(Expr::u32(0))
            }),
            Some(x.clone())
        );
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::Max,
                left: Box::new(Expr::u32(0)),
                right: Box::new(x.clone())
            }),
            Some(x)
        );
    }

    #[test]
    fn min_with_zero_is_zero() {
        let x = Expr::var("x");
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::Min,
                left: Box::new(x),
                right: Box::new(Expr::u32(0))
            }),
            Some(Expr::u32(0))
        );
    }

    #[test]
    fn max_with_u32_max_is_u32_max() {
        let x = Expr::var("x");
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::Max,
                left: Box::new(x),
                right: Box::new(Expr::u32(u32::MAX))
            }),
            Some(Expr::u32(u32::MAX))
        );
    }

    #[test]
    fn lt_zero_for_u32_is_false() {
        let x = Expr::var("x");
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::Lt,
                left: Box::new(x),
                right: Box::new(Expr::u32(0))
            }),
            Some(Expr::bool(false))
        );
    }

    #[test]
    fn ge_zero_for_u32_is_true() {
        let x = Expr::var("x");
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::Ge,
                left: Box::new(x),
                right: Box::new(Expr::u32(0))
            }),
            Some(Expr::bool(true))
        );
    }

    #[test]
    fn le_u32_max_is_true() {
        let x = Expr::var("x");
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::Le,
                left: Box::new(x),
                right: Box::new(Expr::u32(u32::MAX))
            }),
            Some(Expr::bool(true))
        );
    }

    #[test]
    fn gt_u32_max_is_false() {
        let x = Expr::var("x");
        assert_eq!(
            fold_expr(&Expr::BinOp {
                op: crate::ir::BinOp::Gt,
                left: Box::new(x),
                right: Box::new(Expr::u32(u32::MAX))
            }),
            Some(Expr::bool(false))
        );
    }

    // ──── distributive expansion for const-fold feed ────

    /// `Mul(c, Add(a, k))` with both literals folds the right-side
    /// product on the next pass: `c·a + c·k` → `c·a + (c*k)`. The rule
    /// fires at the structural level here; the literal fold of
    /// `Mul(c, Lit(k))` is the responsibility of the existing literal
    /// evaluator.
    #[test]
    fn distributes_mul_lit_over_add_when_one_arm_is_literal() {
        let folded = fold_expr(&Expr::BinOp {
            op: crate::ir::BinOp::Mul,
            left: Box::new(Expr::u32(3)),
            right: Box::new(Expr::BinOp {
                op: crate::ir::BinOp::Add,
                left: Box::new(Expr::var("x")),
                right: Box::new(Expr::u32(7)),
            }),
        });
        assert_eq!(
            folded,
            Some(Expr::add(
                Expr::mul(Expr::u32(3), Expr::var("x")),
                Expr::mul(Expr::u32(3), Expr::u32(7)),
            ))
        );
    }

    /// Symmetric: `Mul(Add(k, b), c)` distributes too.
    #[test]
    fn distributes_add_lit_times_mul_lit_when_one_arm_is_literal() {
        let folded = fold_expr(&Expr::BinOp {
            op: crate::ir::BinOp::Mul,
            left: Box::new(Expr::BinOp {
                op: crate::ir::BinOp::Add,
                left: Box::new(Expr::u32(5)),
                right: Box::new(Expr::var("y")),
            }),
            right: Box::new(Expr::u32(4)),
        });
        assert_eq!(
            folded,
            Some(Expr::add(
                Expr::mul(Expr::u32(5), Expr::u32(4)),
                Expr::mul(Expr::var("y"), Expr::u32(4)),
            ))
        );
    }

    /// i32 literals follow the same wrapping-integer arithmetic, so the
    /// rewrite fires here too.
    #[test]
    fn distributes_mul_lit_i32_over_add_when_one_arm_is_literal() {
        let folded = fold_expr(&Expr::BinOp {
            op: crate::ir::BinOp::Mul,
            left: Box::new(Expr::i32(3)),
            right: Box::new(Expr::BinOp {
                op: crate::ir::BinOp::Add,
                left: Box::new(Expr::var("x")),
                right: Box::new(Expr::i32(7)),
            }),
        });
        assert_eq!(
            folded,
            Some(Expr::add(
                Expr::mul(Expr::i32(3), Expr::var("x")),
                Expr::mul(Expr::i32(3), Expr::i32(7)),
            ))
        );
    }

    /// Negative: `Mul(c, Add(a, b))` where neither addend is a literal
    /// must NOT distribute. Without a literal sibling there is no
    /// guarantee the rewrite reduces instruction count post-fold, and
    /// blind expansion would just bloat the IR.
    #[test]
    fn does_not_distribute_when_neither_addend_is_literal() {
        let folded = fold_expr(&Expr::BinOp {
            op: crate::ir::BinOp::Mul,
            left: Box::new(Expr::u32(3)),
            right: Box::new(Expr::BinOp {
                op: crate::ir::BinOp::Add,
                left: Box::new(Expr::var("x")),
                right: Box::new(Expr::var("y")),
            }),
        });
        assert_eq!(folded, None);
    }

    /// Negative: `Mul(non-lit-c, Add(a, k))`  -  without a literal scalar
    /// on the multiplied side, the rewrite would not fold either new
    /// product, so the rule does not fire.
    #[test]
    fn does_not_distribute_when_scalar_is_not_literal() {
        let folded = fold_expr(&Expr::BinOp {
            op: crate::ir::BinOp::Mul,
            left: Box::new(Expr::var("c")),
            right: Box::new(Expr::BinOp {
                op: crate::ir::BinOp::Add,
                left: Box::new(Expr::var("x")),
                right: Box::new(Expr::u32(7)),
            }),
        });
        assert_eq!(folded, None);
    }

    /// Negative: float multiplication is not associative under rounding,
    /// so `f32 * (f32 + f32)` MUST NOT distribute even when literals are
    /// present. The rounding path through one fused multiply differs
    /// from two separate multiplies + an add.
    #[test]
    fn does_not_distribute_for_float_operands() {
        let folded = fold_expr(&Expr::BinOp {
            op: crate::ir::BinOp::Mul,
            left: Box::new(Expr::f32(3.0)),
            right: Box::new(Expr::BinOp {
                op: crate::ir::BinOp::Add,
                left: Box::new(Expr::var("x")),
                right: Box::new(Expr::f32(7.0)),
            }),
        });
        assert_eq!(folded, None);
    }

    /// Positive: `Mul` whose right side is `Sub` distributes.
    #[test]
    fn distributes_mul_lit_over_sub_when_one_arm_is_literal() {
        let folded = fold_expr(&Expr::BinOp {
            op: crate::ir::BinOp::Mul,
            left: Box::new(Expr::u32(3)),
            right: Box::new(Expr::BinOp {
                op: crate::ir::BinOp::Sub,
                left: Box::new(Expr::var("x")),
                right: Box::new(Expr::u32(7)),
            }),
        });
        let expected = Expr::BinOp {
            op: crate::ir::BinOp::Sub,
            left: Box::new(Expr::mul(Expr::u32(3), Expr::var("x"))),
            right: Box::new(Expr::mul(Expr::u32(3), Expr::u32(7))),
        };
        assert_eq!(folded, Some(expected));
    }

    /// Symmetric: `Mul(Sub(k, b), c)` distributes too.
    #[test]
    fn distributes_sub_lit_times_mul_lit_when_one_arm_is_literal() {
        let folded = fold_expr(&Expr::BinOp {
            op: crate::ir::BinOp::Mul,
            left: Box::new(Expr::BinOp {
                op: crate::ir::BinOp::Sub,
                left: Box::new(Expr::u32(7)),
                right: Box::new(Expr::var("x")),
            }),
            right: Box::new(Expr::u32(3)),
        });
        let expected = Expr::BinOp {
            op: crate::ir::BinOp::Sub,
            left: Box::new(Expr::mul(Expr::u32(7), Expr::u32(3))),
            right: Box::new(Expr::mul(Expr::var("x"), Expr::u32(3))),
        };
        assert_eq!(folded, Some(expected));
    }

    // ─── stronger range fold Mod(x, N) ───────────────────

    fn test_mod_program(c: u32, n: u32) -> crate::optimizer::PassResult {
        use crate::ir::{BufferDecl, DataType, Node, Program};
        use crate::optimizer::passes::algebraic::const_fold::ConstFold;

        let entry = vec![
            Node::let_bind("x", Expr::u32(c)),
            Node::let_bind(
                "y",
                Expr::BinOp {
                    op: crate::ir::BinOp::Mod,
                    left: Box::new(Expr::var("x")),
                    right: Box::new(Expr::u32(n)),
                },
            ),
            Node::store("out", Expr::u32(0), Expr::var("y")),
        ];
        let program = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
            [1, 1, 1],
            entry,
        );
        ConstFold::transform(program)
    }

    fn extract_let_y_value(nodes: &[crate::ir::Node]) -> Option<Expr> {
        for n in nodes {
            match n {
                crate::ir::Node::Let { name, value } if name.as_str() == "y" => {
                    return Some(value.clone())
                }
                crate::ir::Node::Region { body, .. } => {
                    if let Some(v) = extract_let_y_value(body) {
                        return Some(v);
                    }
                }
                _ => {}
            }
        }
        None
    }

    #[test]
    fn stronger_range_fold_mod_positive() {
        let result = test_mod_program(5, 10);
        assert!(result.changed);
        assert_eq!(
            extract_let_y_value(result.program.entry()),
            Some(Expr::var("x"))
        );
    }

    #[test]
    fn stronger_range_fold_mod_negative_c_ge_n() {
        let result = test_mod_program(15, 10);
        // Not changed by lookbehind, but might be folded by normal literal const_fold.
        // If it is folded, it's 5. If not, it remains BinOp.
        // Either way, it doesn't fold to Var("x").
        let y = extract_let_y_value(result.program.entry());
        assert_ne!(y, Some(Expr::var("x")));
    }

    #[test]
    fn stronger_range_fold_mod_negative_not_literal() {
        use crate::ir::{BufferDecl, DataType, Node, Program};
        use crate::optimizer::passes::algebraic::const_fold::ConstFold;

        let entry = vec![
            Node::let_bind("x", Expr::add(Expr::var("z"), Expr::u32(1))),
            Node::let_bind(
                "y",
                Expr::BinOp {
                    op: crate::ir::BinOp::Mod,
                    left: Box::new(Expr::var("x")),
                    right: Box::new(Expr::u32(10)),
                },
            ),
        ];
        let program = Program::wrapped(
            vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
            [1, 1, 1],
            entry,
        );
        let result = ConstFold::transform(program);
        assert!(!result.changed);
    }
}
