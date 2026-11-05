use vyre_foundation::ir::BinOp;

use super::analysis::mul_range;
use super::*;
use crate::descriptor_builder::{binop, body, descriptor, lit, op};
use crate::{KernelBody, KernelDescriptor, KernelOpKind, LiteralValue};

/// A single-thread kernel with no bindings, which is all the range
/// analysis reads.
fn build(body: impl Into<KernelBody>) -> KernelDescriptor {
    descriptor("k").dispatch(1, 1, 1).body(body).build()
}

/// The ranges inferred for a body whose literals are `values` and whose
/// ops are `values.len()` literal reads, one per pool entry.
fn literals(values: impl IntoIterator<Item = LiteralValue>) -> ValueRangeReport {
    let values: Vec<_> = values.into_iter().collect();
    let reads = (0..values.len()).map(|i| lit(i as u32, i as u32));
    analyze(&build(body().ops(reads).literals(values)))
}

/// The range inferred for `lhs kind rhs` with both operands literal, which
/// is the shape of every binary-operator case below. `None` means the
/// analysis refused to propagate.
fn fold(kind: BinOp, lhs: LiteralValue, rhs: LiteralValue) -> Option<IntRange> {
    let desc = build(
        body()
            .op(lit(0, 0))
            .op(lit(1, 1))
            .op(binop(kind, 0, 1, 2))
            .literals([lhs, rhs]),
    );
    analyze(&desc).ranges.get(&2).copied()
}

fn u32_lit(value: u32) -> LiteralValue {
    LiteralValue::U32(value)
}

fn i32_lit(value: i32) -> LiteralValue {
    LiteralValue::I32(value)
}

#[test]
fn empty_kernel_no_ranges() {
    let r = analyze(&build(body()));
    assert!(r.ranges.is_empty());
    assert_eq!(r.known_count(), 0);
}

#[test]
fn lit_u32_yields_singleton() {
    let r = literals([u32_lit(42)]);
    assert_eq!(r.ranges[&0], IntRange::singleton(42));
    assert!(r.ranges[&0].is_singleton());
}

#[test]
fn lit_i32_negative_yields_correct_range() {
    let r = literals([i32_lit(-7)]);
    assert_eq!(r.ranges[&0], IntRange::singleton(-7));
}

#[test]
fn bool_true_is_one_false_is_zero() {
    let r = literals([LiteralValue::Bool(true), LiteralValue::Bool(false)]);
    assert_eq!(r.ranges[&0], IntRange::singleton(1));
    assert_eq!(r.ranges[&1], IntRange::singleton(0));
}

#[test]
fn min_of_two_lits_propagates() {
    // Both operands are singletons; result range is min..=min.
    assert_eq!(
        fold(BinOp::Min, u32_lit(3), u32_lit(5)),
        Some(IntRange::singleton(3))
    );
}

#[test]
fn max_of_two_lits_propagates() {
    assert_eq!(
        fold(BinOp::Max, u32_lit(3), u32_lit(5)),
        Some(IntRange::singleton(5))
    );
}

#[test]
fn add_propagates_singleton_ranges() {
    // 3 + 5 → [8, 8]
    assert_eq!(
        fold(BinOp::Add, u32_lit(3), u32_lit(5)),
        Some(IntRange::singleton(8))
    );
}

#[test]
fn sub_flips_operand_bounds() {
    // l - r where l ∈ [a,b] and r ∈ [c,d] → [a-d, b-c]
    // For singletons: 10 - 3 = 7
    assert_eq!(
        fold(BinOp::Sub, i32_lit(10), i32_lit(3)),
        Some(IntRange::singleton(7))
    );
}

#[test]
fn bitand_with_mask_bounds_to_zero_through_mask() {
    // x & 0xFF where x is unknown but BitAnd(x, 0xFF) bounds to [0, 0xFF].
    // Phase 1 only knows x's range when x is itself a literal,
    // so use lit-lit here.
    // l.max = 0x12345678, r.max = 0xFF; min(...) = 0xFF.
    assert_eq!(
        fold(BinOp::BitAnd, u32_lit(0x1234_5678), u32_lit(0xFF)),
        Some(IntRange { min: 0, max: 0xFF })
    );
}

#[test]
fn shl_propagates_with_singleton_shift() {
    // 5 << 3 = 40
    assert_eq!(
        fold(BinOp::Shl, u32_lit(5), u32_lit(3)),
        Some(IntRange::singleton(40))
    );
}

#[test]
fn shr_propagates_with_singleton_shift() {
    // 40 >> 3 = 5
    assert_eq!(
        fold(BinOp::Shr, u32_lit(40), u32_lit(3)),
        Some(IntRange::singleton(5))
    );
}

#[test]
fn shl_with_huge_shift_not_propagated() {
    // shift ≥ 32: refuse (would be overflow on i64 too in extreme cases).
    assert_eq!(fold(BinOp::Shl, u32_lit(1), u32_lit(64)), None);
}

#[test]
fn bitor_propagates_with_singletons() {
    // 0xF0 | 0x0F = 0xFF
    // l.max=0xF0, r.max=0x0F → max = 0xF0|0x0F = 0xFF.
    // l.min=0xF0, r.min=0x0F → min = max(0xF0, 0x0F) = 0xF0.
    assert_eq!(
        fold(BinOp::BitOr, u32_lit(0xF0), u32_lit(0x0F)),
        Some(IntRange {
            min: 0xF0,
            max: 0xFF
        })
    );
}

#[test]
fn bitand_negative_operand_not_propagated() {
    // Neg operand → BitAnd refused.
    assert_eq!(fold(BinOp::BitAnd, i32_lit(-1), i32_lit(0xFF)), None);
}

#[test]
fn mul_singletons() {
    assert_eq!(
        fold(BinOp::Mul, i32_lit(7), i32_lit(-3)),
        Some(IntRange::singleton(-21))
    );
}

#[test]
fn mul_range_corner_helper() {
    // [2, 5] * [3, 4] = corners 6, 8, 15, 20 → [6, 20].
    let r = mul_range(IntRange { min: 2, max: 5 }, IntRange { min: 3, max: 4 });
    assert_eq!(r, Some(IntRange { min: 6, max: 20 }));

    // [-2, 3] * [-1, 4] = corners 2, -8, -3, 12 → [-8, 12].
    let r = mul_range(IntRange { min: -2, max: 3 }, IntRange { min: -1, max: 4 });
    assert_eq!(r, Some(IntRange { min: -8, max: 12 }));
}

#[test]
fn add_chains_propagate() {
    // (3 + 5) + 7 = 15
    let desc = build(
        body()
            .op(lit(0, 0))
            .op(lit(1, 1))
            .op(lit(2, 2))
            .op(binop(BinOp::Add, 0, 1, 3))
            .op(binop(BinOp::Add, 3, 2, 4))
            .literals([u32_lit(3), u32_lit(5), u32_lit(7)]),
    );
    let r = analyze(&desc);
    assert_eq!(r.ranges[&4], IntRange::singleton(15));
}

#[test]
fn non_lit_op_no_range() {
    // LocalInvocationId  -  can't statically bound in phase 1.
    let desc = build(body().op(op(KernelOpKind::LocalInvocationId, [0], 0)));
    let r = analyze(&desc);
    assert!(!r.ranges.contains_key(&0));
}

#[test]
fn as_constant_returns_value_for_singleton() {
    let r = literals([u32_lit(42)]);
    assert_eq!(r.as_constant(0), Some(42));
    assert_eq!(r.as_constant(99), None); // unknown id
}

#[test]
fn as_constant_returns_none_for_non_singleton() {
    // Phase 1 folds lit + lit to a singleton, so a non-singleton range
    // cannot come out of the analysis; construct the report directly.
    let mut report = ValueRangeReport::default();
    report.ranges.insert(7, IntRange { min: 0, max: 10 });
    assert_eq!(report.as_constant(7), None);
}

#[test]
fn report_accessors() {
    let r = literals([u32_lit(0), u32_lit(42)]);
    // is_definitely
    assert_eq!(r.is_definitely(0, 0), Some(true));
    assert_eq!(r.is_definitely(0, 1), Some(false));
    assert_eq!(r.is_definitely(99, 0), None); // unknown id
                                              // is_definitely_below
    assert_eq!(r.is_definitely_below(1, 100), Some(true));
    assert_eq!(r.is_definitely_below(1, 42), Some(false)); // 42 < 42 false
                                                           // is_definitely_at_least
    assert_eq!(r.is_definitely_at_least(1, 42), Some(true));
    assert_eq!(r.is_definitely_at_least(1, 43), Some(false));
    // get
    assert_eq!(r.get(0), Some(IntRange::singleton(0)));
    assert_eq!(r.get(99), None);
}

#[test]
fn range_helpers() {
    let r = IntRange { min: 3, max: 7 };
    assert!(r.contains(5));
    assert!(r.contains(3));
    assert!(r.contains(7));
    assert!(!r.contains(2));
    assert!(!r.contains(8));
    assert!(!r.is_singleton());

    let s = IntRange::singleton(42);
    assert!(s.is_singleton());

    let u = IntRange { min: 0, max: 5 }.union(IntRange { min: 3, max: 10 });
    assert_eq!(u, IntRange { min: 0, max: 10 });
}
