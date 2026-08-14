use vyre_foundation::ir::BinOp;

use super::analysis::mul_range;
use super::*;
use crate::{
    BindingLayout, Dispatch, KernelBody, KernelDescriptor, KernelOp, KernelOpKind, LiteralValue,
};

fn build(ops: Vec<KernelOp>, lits: Vec<LiteralValue>) -> KernelDescriptor {
    KernelDescriptor {
        id: "k".into(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: Dispatch::new(1, 1, 1),
        body: KernelBody {
            ops,
            child_bodies: vec![],
            literals: lits,
        },
    }
}

#[test]
fn empty_kernel_no_ranges() {
    let r = analyze(&build(vec![], vec![]));
    assert!(r.ranges.is_empty());
    assert_eq!(r.known_count(), 0);
}

#[test]
fn lit_u32_yields_singleton() {
    let desc = build(
        vec![KernelOp {
            kind: KernelOpKind::Literal,
            operands: vec![0],
            result: Some(0),
        }],
        vec![LiteralValue::U32(42)],
    );
    let r = analyze(&desc);
    assert_eq!(r.ranges[&0], IntRange::singleton(42));
    assert!(r.ranges[&0].is_singleton());
}

#[test]
fn lit_i32_negative_yields_correct_range() {
    let desc = build(
        vec![KernelOp {
            kind: KernelOpKind::Literal,
            operands: vec![0],
            result: Some(0),
        }],
        vec![LiteralValue::I32(-7)],
    );
    let r = analyze(&desc);
    assert_eq!(r.ranges[&0], IntRange::singleton(-7));
}

#[test]
fn bool_true_is_one_false_is_zero() {
    let desc = build(
        vec![
            KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![0],
                result: Some(0),
            },
            KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![1],
                result: Some(1),
            },
        ],
        vec![LiteralValue::Bool(true), LiteralValue::Bool(false)],
    );
    let r = analyze(&desc);
    assert_eq!(r.ranges[&0], IntRange::singleton(1));
    assert_eq!(r.ranges[&1], IntRange::singleton(0));
}

#[test]
fn min_of_two_lits_propagates() {
    let desc = build(
        vec![
            KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![0],
                result: Some(0),
            },
            KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![1],
                result: Some(1),
            },
            KernelOp {
                kind: KernelOpKind::BinOpKind(BinOp::Min),
                operands: vec![0, 1],
                result: Some(2),
            },
        ],
        vec![LiteralValue::U32(3), LiteralValue::U32(5)],
    );
    let r = analyze(&desc);
    // Both operands are singletons; result range is min..=min.
    assert_eq!(r.ranges[&2], IntRange::singleton(3));
}

#[test]
fn max_of_two_lits_propagates() {
    let desc = build(
        vec![
            KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![0],
                result: Some(0),
            },
            KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![1],
                result: Some(1),
            },
            KernelOp {
                kind: KernelOpKind::BinOpKind(BinOp::Max),
                operands: vec![0, 1],
                result: Some(2),
            },
        ],
        vec![LiteralValue::U32(3), LiteralValue::U32(5)],
    );
    let r = analyze(&desc);
    assert_eq!(r.ranges[&2], IntRange::singleton(5));
}

#[test]
fn add_propagates_singleton_ranges() {
    // 3 + 5 → [8, 8]
    let desc = build(
        vec![
            KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![0],
                result: Some(0),
            },
            KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![1],
                result: Some(1),
            },
            KernelOp {
                kind: KernelOpKind::BinOpKind(BinOp::Add),
                operands: vec![0, 1],
                result: Some(2),
            },
        ],
        vec![LiteralValue::U32(3), LiteralValue::U32(5)],
    );
    let r = analyze(&desc);
    assert_eq!(r.ranges[&2], IntRange::singleton(8));
}

#[test]
fn sub_flips_operand_bounds() {
    // l - r where l ∈ [a,b] and r ∈ [c,d] → [a-d, b-c]
    // For singletons: 10 - 3 = 7
    let desc = build(
        vec![
            KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![0],
                result: Some(0),
            },
            KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![1],
                result: Some(1),
            },
            KernelOp {
                kind: KernelOpKind::BinOpKind(BinOp::Sub),
                operands: vec![0, 1],
                result: Some(2),
            },
        ],
        vec![LiteralValue::I32(10), LiteralValue::I32(3)],
    );
    let r = analyze(&desc);
    assert_eq!(r.ranges[&2], IntRange::singleton(7));
}

#[test]

fn bitand_with_mask_bounds_to_zero_through_mask() {
    // x & 0xFF where x is unknown but BitAnd(x, 0xFF) bounds to [0, 0xFF].
    // Phase 1 only knows x's range when x is itself a literal,
    // so use lit-lit here.
    let desc = build(
        vec![
            KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![0],
                result: Some(0),
            },
            KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![1],
                result: Some(1),
            },
            KernelOp {
                kind: KernelOpKind::BinOpKind(BinOp::BitAnd),
                operands: vec![0, 1],
                result: Some(2),
            },
        ],
        vec![LiteralValue::U32(0x12345678), LiteralValue::U32(0xFF)],
    );
    let r = analyze(&desc);
    // l.max = 0x12345678, r.max = 0xFF; min(...) = 0xFF.
    assert_eq!(r.ranges[&2], IntRange { min: 0, max: 0xFF });
}

#[test]
fn shl_propagates_with_singleton_shift() {
    // 5 << 3 = 40
    let desc = build(
        vec![
            KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![0],
                result: Some(0),
            },
            KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![1],
                result: Some(1),
            },
            KernelOp {
                kind: KernelOpKind::BinOpKind(BinOp::Shl),
                operands: vec![0, 1],
                result: Some(2),
            },
        ],
        vec![LiteralValue::U32(5), LiteralValue::U32(3)],
    );
    let r = analyze(&desc);
    assert_eq!(r.ranges[&2], IntRange::singleton(40));
}

#[test]
fn shr_propagates_with_singleton_shift() {
    // 40 >> 3 = 5
    let desc = build(
        vec![
            KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![0],
                result: Some(0),
            },
            KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![1],
                result: Some(1),
            },
            KernelOp {
                kind: KernelOpKind::BinOpKind(BinOp::Shr),
                operands: vec![0, 1],
                result: Some(2),
            },
        ],
        vec![LiteralValue::U32(40), LiteralValue::U32(3)],
    );
    let r = analyze(&desc);
    assert_eq!(r.ranges[&2], IntRange::singleton(5));
}

#[test]
fn shl_with_huge_shift_not_propagated() {
    // shift ≥ 32: refuse (would be overflow on i64 too in extreme cases).
    let desc = build(
        vec![
            KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![0],
                result: Some(0),
            },
            KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![1],
                result: Some(1),
            },
            KernelOp {
                kind: KernelOpKind::BinOpKind(BinOp::Shl),
                operands: vec![0, 1],
                result: Some(2),
            },
        ],
        vec![LiteralValue::U32(1), LiteralValue::U32(64)],
    );
    let r = analyze(&desc);
    assert!(!r.ranges.contains_key(&2));
}

#[test]
fn bitor_propagates_with_singletons() {
    // 0xF0 | 0x0F = 0xFF
    let desc = build(
        vec![
            KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![0],
                result: Some(0),
            },
            KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![1],
                result: Some(1),
            },
            KernelOp {
                kind: KernelOpKind::BinOpKind(BinOp::BitOr),
                operands: vec![0, 1],
                result: Some(2),
            },
        ],
        vec![LiteralValue::U32(0xF0), LiteralValue::U32(0x0F)],
    );
    let r = analyze(&desc);
    // l.max=0xF0, r.max=0x0F → max = 0xF0|0x0F = 0xFF.
    // l.min=0xF0, r.min=0x0F → min = max(0xF0, 0x0F) = 0xF0.
    assert_eq!(
        r.ranges[&2],
        IntRange {
            min: 0xF0,
            max: 0xFF
        }
    );
}

#[test]
fn bitand_negative_operand_not_propagated() {
    let desc = build(
        vec![
            KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![0],
                result: Some(0),
            },
            KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![1],
                result: Some(1),
            },
            KernelOp {
                kind: KernelOpKind::BinOpKind(BinOp::BitAnd),
                operands: vec![0, 1],
                result: Some(2),
            },
        ],
        vec![LiteralValue::I32(-1), LiteralValue::I32(0xFF)],
    );
    let r = analyze(&desc);
    // Neg operand → BitAnd refused.
    assert!(!r.ranges.contains_key(&2));
}

#[test]
fn mul_singletons() {
    let desc = build(
        vec![
            KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![0],
                result: Some(0),
            },
            KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![1],
                result: Some(1),
            },
            KernelOp {
                kind: KernelOpKind::BinOpKind(BinOp::Mul),
                operands: vec![0, 1],
                result: Some(2),
            },
        ],
        vec![LiteralValue::I32(7), LiteralValue::I32(-3)],
    );
    let r = analyze(&desc);
    assert_eq!(r.ranges[&2], IntRange::singleton(-21));
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
        vec![
            KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![0],
                result: Some(0),
            },
            KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![1],
                result: Some(1),
            },
            KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![2],
                result: Some(2),
            },
            KernelOp {
                kind: KernelOpKind::BinOpKind(BinOp::Add),
                operands: vec![0, 1],
                result: Some(3),
            },
            KernelOp {
                kind: KernelOpKind::BinOpKind(BinOp::Add),
                operands: vec![3, 2],
                result: Some(4),
            },
        ],
        vec![
            LiteralValue::U32(3),
            LiteralValue::U32(5),
            LiteralValue::U32(7),
        ],
    );
    let r = analyze(&desc);
    assert_eq!(r.ranges[&4], IntRange::singleton(15));
}

#[test]
fn non_lit_op_no_range() {
    // LocalInvocationId  -  can't statically bound in phase 1.
    let desc = build(
        vec![KernelOp {
            kind: KernelOpKind::LocalInvocationId,
            operands: vec![0],
            result: Some(0),
        }],
        vec![],
    );
    let r = analyze(&desc);
    assert!(!r.ranges.contains_key(&0));
}

#[test]
fn as_constant_returns_value_for_singleton() {
    let desc = build(
        vec![KernelOp {
            kind: KernelOpKind::Literal,
            operands: vec![0],
            result: Some(0),
        }],
        vec![LiteralValue::U32(42)],
    );
    let r = analyze(&desc);
    assert_eq!(r.as_constant(0), Some(42));
    assert_eq!(r.as_constant(99), None); // unknown id
}

#[test]
fn as_constant_returns_none_for_non_singleton() {
    // Build an Add of two ranges that produces a non-singleton.
    // Phase 1: lit + lit folds to singleton, so we can't easily
    // produce a non-singleton via the analyses. Test via direct
    // ValueRangeReport construction.
    let mut report = ValueRangeReport::default();
    report.ranges.insert(7, IntRange { min: 0, max: 10 });
    assert_eq!(report.as_constant(7), None);
}

#[test]
fn report_accessors() {
    let desc = build(
        vec![
            KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![0],
                result: Some(0),
            },
            KernelOp {
                kind: KernelOpKind::Literal,
                operands: vec![1],
                result: Some(1),
            },
        ],
        vec![LiteralValue::U32(0), LiteralValue::U32(42)],
    );
    let r = analyze(&desc);
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
