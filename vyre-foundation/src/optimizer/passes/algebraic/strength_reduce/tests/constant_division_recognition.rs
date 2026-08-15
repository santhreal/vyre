//! Pre-lowering constant-division recognition.
//!
//! Four rewrites run before the general lowering table because lowering a
//! child destroys the constant they key on: Lemire's divisibility test, common
//! factor cancellation between a dividend's multiplier and the divisor,
//! fusion of a constant division chain, and narrowing of a nested modulus.
//!
//! Every test here is differential. It runs the real recognition entry point,
//! evaluates the rewritten expression, and compares against the operator the
//! source expression names. None of them assert expression shape, so a
//! different but equally correct emission keeps them green while a wrong one
//! cannot.
//!
//! The divisor space is enumerated at run time by asking the pass which
//! divisors it admits, so a divisor a later change starts rewriting is proved
//! here without an edit. A floor on the admitted count keeps an accidentally
//! empty admission set from passing vacuously.

use super::modulo_constant::eval_u32;
use super::*;
/// Divisors the tests offer to the pass: every small value, plus the large
/// end of the range where `u32::MAX / d` is small and the Lemire limit is
/// tightest.
fn candidate_divisors() -> Vec<u32> {
    (0u32..=512)
        .chain((1u32..=64).map(|i| u32::MAX / i))
        .chain([65_535, 65_536, 65_537, 1_000_000_007])
        .collect()
}

/// Operand values the single-divisor sweeps use.
fn sample_operands() -> Vec<u32> {
    let mut values: Vec<u32> = (0u32..=600).collect();
    values.extend((0..32).map(|bit| 1u32 << bit));
    values.extend((0..32).map(|bit| (1u32 << bit).wrapping_sub(1)));
    values.extend([u32::MAX, u32::MAX - 1, 0x8000_0000, 0x7FFF_FFFF]);
    // A deterministic spread across the full range; no test may depend on a
    // seed the harness does not own.
    let mut state = 0x2545_F491u32;
    for _ in 0..2_000 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        values.push(state);
    }
    values
}

/// Smaller sweep for the tests that enumerate thousands of constant pairs.
fn coarse_operands() -> Vec<u32> {
    let mut values: Vec<u32> = (0u32..=64).collect();
    values.extend((0..32).map(|bit| 1u32 << bit));
    values.extend((0..32).map(|bit| (1u32 << bit).wrapping_sub(1)));
    values.extend([u32::MAX, u32::MAX - 1, 0x8000_0000, 0x7FFF_FFFF]);
    values
}

/// Evaluate a comparison the divisibility test emits.
fn eval_bool_expr(expr: &Expr, x: u32) -> bool {
    let Expr::BinOp { op, left, right } = expr else {
        panic!("expected a comparison at the root, got {expr:?}");
    };
    let l = eval_u32(left, x);
    let r = eval_u32(right, x);
    match op {
        BinOp::Le => l <= r,
        BinOp::Gt => l > r,
        BinOp::Eq => l == r,
        BinOp::Ne => l != r,
        other => panic!("evaluator has no comparison arm for {other:?}"),
    }
}

fn remainder_compared_to_zero(divisor: u32, op: BinOp) -> Expr {
    Expr::BinOp {
        op,
        left: Box::new(Expr::rem(Expr::var("x"), Expr::u32(divisor))),
        right: Box::new(Expr::u32(0)),
    }
}

#[test]
fn divisibility_test_agrees_with_the_remainder_it_replaces() {
    let operands = sample_operands();
    let mut admitted = 0usize;
    for divisor in candidate_divisors() {
        let source = remainder_compared_to_zero(divisor, BinOp::Eq);
        let Some(rewritten) = recognize_source_shape(&source) else {
            continue;
        };
        admitted += 1;
        for &x in &operands {
            assert_eq!(
                eval_bool_expr(&rewritten, x),
                x % divisor == 0,
                "divisibility test disagrees at x={x}, d={divisor}"
            );
        }
    }
    assert!(
        admitted >= 400,
        "recognition admitted only {admitted} divisors; the divisibility rewrite is not firing"
    );
}

#[test]
fn non_divisibility_test_agrees_with_the_remainder_it_replaces() {
    let operands = sample_operands();
    let mut admitted = 0usize;
    for divisor in candidate_divisors() {
        let source = remainder_compared_to_zero(divisor, BinOp::Ne);
        let Some(rewritten) = recognize_source_shape(&source) else {
            continue;
        };
        admitted += 1;
        for &x in &operands {
            assert_eq!(
                eval_bool_expr(&rewritten, x),
                x % divisor != 0,
                "non-divisibility test disagrees at x={x}, d={divisor}"
            );
        }
    }
    assert!(
        admitted >= 400,
        "the non-divisibility rewrite is not firing"
    );
}

#[test]
fn only_divisors_with_no_cheaper_answer_are_admitted() {
    for divisor in candidate_divisors() {
        let source = remainder_compared_to_zero(divisor, BinOp::Eq);
        if recognize_source_shape(&source).is_none() {
            assert!(
                divisor <= 1 || divisor.is_power_of_two(),
                "divisor {divisor} was declined but has no cheaper lowering"
            );
        }
    }
}

/// `(x & mask) * c / d`: the mask supplies the range proof the cancellation
/// needs, so the rewrite is admitted exactly when `mask * c` does not wrap.
fn masked_product_over(mask: u32, multiplier: u32, divisor: u32) -> Expr {
    Expr::div(
        Expr::mul(
            Expr::bitand(Expr::var("x"), Expr::u32(mask)),
            Expr::u32(multiplier),
        ),
        Expr::u32(divisor),
    )
}

#[test]
fn cancelling_a_common_factor_preserves_the_quotient() {
    let operands = coarse_operands();
    let mut fired = 0usize;
    for mask_bits in 1u32..=24 {
        let mask = (1u32 << mask_bits) - 1;
        for multiplier in 1u32..=24 {
            for divisor in 2u32..=24 {
                let source = masked_product_over(mask, multiplier, divisor);
                let Some(rewritten) = recognize_source_shape(&source) else {
                    continue;
                };
                fired += 1;
                for &x in &operands {
                    assert_eq!(
                        eval_u32(&rewritten, x),
                        eval_u32(&source, x),
                        "cancellation changed the quotient at x={x}, mask={mask}, \
                         c={multiplier}, d={divisor}"
                    );
                }
            }
        }
    }
    assert!(fired >= 100, "factor cancellation fired only {fired} times");
}

#[test]
fn an_unbounded_dividend_blocks_cancellation() {
    // Without a range proof, `(x * 2) / 2` is not `x`: at x = 2^31 the product
    // wraps to zero and the quotient is zero. The rewrite must decline.
    let source = Expr::div(Expr::mul(Expr::var("x"), Expr::u32(2)), Expr::u32(2));
    assert!(
        recognize_source_shape(&source).is_none(),
        "cancellation fired on an operand with no provable bound"
    );
}

#[test]
fn a_product_that_can_wrap_blocks_cancellation() {
    // The mask allows values up to 2^31 - 1, so `x * 4` wraps.
    let source = masked_product_over(0x7FFF_FFFF, 4, 2);
    assert!(
        recognize_source_shape(&source).is_none(),
        "cancellation fired on a product that can wrap"
    );
}

#[test]
fn a_constant_division_chain_fuses_into_one_division() {
    let operands = coarse_operands();
    for outer in 1u32..=40 {
        for inner in 1u32..=40 {
            let source = Expr::div(
                Expr::div(Expr::var("x"), Expr::u32(inner)),
                Expr::u32(outer),
            );
            let Some(rewritten) = recognize_source_shape(&source) else {
                panic!("division chain {inner} then {outer} was not fused");
            };
            for &x in &operands {
                assert_eq!(
                    eval_u32(&rewritten, x),
                    eval_u32(&source, x),
                    "fusion changed the quotient at x={x}, inner={inner}, outer={outer}"
                );
            }
        }
    }
}

#[test]
fn a_nested_modulus_narrows_only_when_the_outer_divides_the_inner() {
    let operands = coarse_operands();
    for inner in 1u32..=48 {
        for outer in 1u32..=48 {
            let source = Expr::rem(
                Expr::rem(Expr::var("x"), Expr::u32(inner)),
                Expr::u32(outer),
            );
            match recognize_source_shape(&source) {
                Some(rewritten) => {
                    assert_eq!(
                        inner % outer,
                        0,
                        "narrowed {inner} then {outer} where the outer does not divide the inner"
                    );
                    for &x in &operands {
                        assert_eq!(
                            eval_u32(&rewritten, x),
                            eval_u32(&source, x),
                            "narrowing changed the remainder at x={x}, \
                             inner={inner}, outer={outer}"
                        );
                    }
                }
                None => assert_ne!(
                    inner % outer,
                    0,
                    "declined to narrow {inner} then {outer} where the outer does divide the inner"
                ),
            }
        }
    }
}

#[test]
fn an_over_width_shift_chain_keeps_both_shifts() {
    // `(x >> 20) >> 20` is zero for an unsigned operand and -1 for a negative
    // signed one. Nothing at this point knows which, so folding the pair to a
    // constant is a miscompile on the signed half of the range.
    for (inner, outer) in [(20u32, 20u32), (31, 1), (16, 24), (30, 30)] {
        let source = Expr::shr(
            Expr::shr(Expr::var("x"), Expr::u32(inner)),
            Expr::u32(outer),
        );
        assert!(
            reduce_expr(&source).is_none(),
            "fused an over-width right-shift chain {inner} then {outer}"
        );
        let source = Expr::shl(
            Expr::shl(Expr::var("x"), Expr::u32(inner)),
            Expr::u32(outer),
        );
        assert!(
            reduce_expr(&source).is_none(),
            "fused an over-width left-shift chain {inner} then {outer}"
        );
    }
}

#[test]
fn an_in_width_shift_chain_still_fuses() {
    let source = Expr::shr(Expr::shr(Expr::var("x"), Expr::u32(3)), Expr::u32(4));
    let rewritten = reduce_expr(&source).expect("in-width shift chain must still fuse");
    for x in sample_operands() {
        assert_eq!(eval_u32(&rewritten, x), x >> 7);
    }
}

/// Count the operation nodes a lowered expression emits.
fn operation_count(expr: &Expr) -> usize {
    match expr {
        Expr::BinOp { left, right, .. } => 1 + operation_count(left) + operation_count(right),
        _ => 0,
    }
}

/// Apply the lowering table until it stops changing, the way the scheduler
/// re-runs the pass to fixpoint.
fn lower_to_fixpoint(expr: &Expr) -> Expr {
    let mut current = expr.clone();
    for _ in 0..16 {
        let next = crate::optimizer::rewrite::rewrite_expr(&current, &mut reduce_expr).into_owned();
        if next == current {
            return current;
        }
        current = next;
    }
    current
}

#[test]
fn recognition_emits_fewer_operations_than_lowering_the_remainder() {
    // The measurement the divisibility rewrite exists for: the general
    // remainder lowering builds a multiply-high, a shift, a fixup on some
    // divisors, a multiply, a subtract and a compare, and reads the operand
    // twice. Lemire's test is a multiply, a rotate and a compare.
    for divisor in [3u32, 5, 7, 10, 100, 1_000, 1_000_000_007] {
        let source = remainder_compared_to_zero(divisor, BinOp::Eq);
        let lowered = operation_count(&lower_to_fixpoint(&source));
        let recognized = operation_count(&lower_to_fixpoint(
            &recognize_source_shape(&source).expect("divisibility test must fire"),
        ));
        assert!(
            recognized < lowered,
            "d={divisor}: recognition emits {recognized} operations, lowering emits {lowered}"
        );
        assert!(
            lowered >= 5,
            "d={divisor}: the remainder lowering shrank to {lowered} operations;              the measured baseline for this claim was five"
        );
        assert!(
            recognized <= 3,
            "d={divisor}: Lemire's test must stay within three operations, got {recognized}"
        );
    }
}
