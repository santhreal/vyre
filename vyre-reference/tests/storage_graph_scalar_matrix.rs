//! The storage-graph scalar sweep: every literal width, one table, one loop.
//!
//! WHY: the sweep used to exist four times, once per scalar width, and the
//! copies had drifted apart in every axis a sweep has. `f32` sampled 2048 binary
//! and 4096 unary cases where `u32` and `u64` sampled 4096 and 8192, so the
//! width with the most delicate semantics was the least exercised, and nothing
//! said whether that was a float domain restriction or an editing accident. It
//! was an accident: the case count only indexes the corpus, and the `f32` corpus
//! is a legal source of 4096 pairs. `i32` swept its unary surface once per corpus
//! value instead of sampling, and left `SaturatingSub` and `SaturatingMul`
//! unexercised although the oracle defines them.
//!
//! So the shape of the sweep is data now, one row per width, and the row is
//! checked against the frozen `NodeStorage` surface: a new scalar literal has no
//! row, and the suite says so by name rather than sweeping four widths out of
//! five forever.
//!
//! Each row declares its operations by declaring their expected results. The
//! declaration is proven in both directions: a declared operation must evaluate
//! to the declared value, and an undeclared one must be refused by name. That
//! second half is what pins the oracle's real capability surface, which is not
//! uniform across widths: `i32` defines one unary operation where `u64` defines
//! seven, `f32` has no remainder, and `u32` has no negation. Widening any of
//! those turns this suite red until the row records the decision.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use vyre_foundation::ir::{BinOp, NodeId, NodeStorage, UnOp, Value as IrValue};
use vyre_reference::ieee754::canonical_f32;
use vyre_reference::{run_storage_graph, ReferenceError};

#[path = "support/scalar_corpus.rs"]
mod scalar_corpus;
#[path = "../../tests/support/spec_variant_tables.rs"]
mod spec_variant_tables;

use scalar_corpus::{f32_corpus, i32_corpus, u32_corpus, u64_corpus};
use spec_variant_tables::{builtin_bin_ops, builtin_un_ops};

/// Binary cases a sampled row draws.
const BINARY_DEPTH: usize = 4096;

/// Unary cases a sampled row draws.
const UNARY_DEPTH: usize = 8192;

/// The refusal every width names for an operation it does not define.
const UNSUPPORTED: &str = "unsupported";

/// The refusal the signed widths name for a division the target leaves open.
const UNDEFINED_SIGNED_DIVISION: &str = "undefined target-text semantics";

/// The refusal mixed-width operands name.
const TYPE_MISMATCH: &str = "type mismatch";

/// What the oracle must do with one case.
enum Expected {
    /// Evaluate to exactly this value.
    Value(IrValue),
    /// Fail with a diagnostic containing this text.
    Refused(&'static str),
}

/// How a case index becomes a corpus index.
#[derive(Clone, Copy)]
enum Draw {
    /// `index * multiplier + addend`, modulo the corpus length. The multiplier
    /// stays coprime with that length so the draw walks every value instead of
    /// revisiting a subset, which is checked rather than assumed.
    Strided {
        /// Stride through the corpus.
        multiplier: usize,
        /// Offset the stride starts from.
        addend: usize,
    },
    /// The case index IS the corpus index: one case per value, or for a binary
    /// sweep one case per ordered pair. Only a corpus short enough to enumerate
    /// can use this.
    Exhaustive,
}

impl Draw {
    /// Operand indices for one binary case.
    fn pair(self, case: usize, len: usize) -> (usize, usize) {
        match self {
            Self::Strided { multiplier, addend } => (
                case % len,
                case.wrapping_mul(multiplier).wrapping_add(addend) % len,
            ),
            Self::Exhaustive => ((case / len) % len, case % len),
        }
    }

    /// Operand index for one unary case.
    fn single(self, case: usize, len: usize) -> usize {
        match self {
            Self::Strided { multiplier, addend } => {
                case.wrapping_mul(multiplier).wrapping_add(addend) % len
            }
            Self::Exhaustive => case % len,
        }
    }
}

/// One scalar width's row of the sweep.
struct Width {
    /// The `NodeStorage` literal variant this row sweeps, spelled as the frozen
    /// public-API snapshot spells it.
    literal: &'static str,
    /// How the oracle names this width in its refusals.
    diagnostic: &'static str,
    /// Binary cases this row runs.
    binary_cases: usize,
    /// Unary cases this row runs.
    unary_cases: usize,
    /// How binary operands are drawn from the corpus.
    binary_draw: Draw,
    /// How the unary operand is drawn from the corpus.
    unary_draw: Draw,
    /// Binary operations this width's expectations currently define. A floor,
    /// not a count: the derived surface may grow, and shrinking it is a
    /// decision somebody has to record here with its reason.
    binary_ops_floor: usize,
    /// Unary operations this width currently defines, same rule.
    unary_ops_floor: usize,
    /// Why this row's case counts differ from the sampled depth. Empty when
    /// they do not.
    exception: &'static str,
    /// The values this row sweeps, as literal nodes.
    corpus: fn() -> Vec<NodeStorage>,
    /// The declared binary surface: `None` for an operation this width does not
    /// define.
    binary: fn(BinOp, &NodeStorage, &NodeStorage) -> Option<Expected>,
    /// The declared unary surface, same convention.
    unary: fn(&UnOp, &NodeStorage) -> Option<Expected>,
    /// Cases a sampled draw must not be trusted to reach: a divisor of zero, a
    /// signed overflow edge. A boundary that only the draw covers is a boundary
    /// that silently stops being covered when the corpus changes.
    edges: Vec<(BinOp, NodeStorage, NodeStorage)>,
}

impl Width {
    /// Binary operations this row declares, derived from its expectations.
    fn binary_ops(&self, corpus: &[NodeStorage]) -> Vec<BinOp> {
        let (left, right) = self.probe(corpus);
        builtin_bin_ops()
            .into_iter()
            .filter(|op| (self.binary)(*op, left, right).is_some())
            .collect()
    }

    /// Unary operations this row declares, derived from its expectations.
    fn unary_ops(&self, corpus: &[NodeStorage]) -> Vec<UnOp> {
        let (operand, _) = self.probe(corpus);
        builtin_un_ops()
            .into_iter()
            .filter(|op| (self.unary)(op, operand).is_some())
            .collect()
    }

    /// Two corpus entries to probe the declared surface with.
    fn probe<'corpus>(
        &self,
        corpus: &'corpus [NodeStorage],
    ) -> (&'corpus NodeStorage, &'corpus NodeStorage) {
        (&corpus[0], &corpus[1 % corpus.len()])
    }
}

fn widths() -> Vec<Width> {
    vec![
        Width {
            literal: "LitU32",
            diagnostic: "u32",
            binary_cases: BINARY_DEPTH,
            unary_cases: UNARY_DEPTH,
            binary_draw: Draw::Strided {
                multiplier: 37,
                addend: 11,
            },
            unary_draw: Draw::Strided {
                multiplier: 19,
                addend: 3,
            },
            binary_ops_floor: 29,
            unary_ops_floor: 6,
            exception: "",
            corpus: || u32_corpus().into_iter().map(NodeStorage::LitU32).collect(),
            binary: u32_binary,
            unary: u32_unary,
            edges: divisor_edges(
                &[0, 1, 2, i32::MAX as u32, i32::MIN as u32, u32::MAX],
                0,
                NodeStorage::LitU32,
            ),
        },
        Width {
            literal: "LitU64",
            diagnostic: "u64",
            binary_cases: BINARY_DEPTH,
            unary_cases: UNARY_DEPTH,
            binary_draw: Draw::Strided {
                multiplier: 43,
                addend: 19,
            },
            unary_draw: Draw::Strided {
                multiplier: 23,
                addend: 7,
            },
            binary_ops_floor: 27,
            unary_ops_floor: 7,
            exception: "",
            corpus: || u64_corpus().into_iter().map(NodeStorage::LitU64).collect(),
            binary: u64_binary,
            unary: u64_unary,
            edges: divisor_edges(
                &[
                    0,
                    1,
                    2,
                    u64::from(u32::MAX),
                    u64::from(u32::MAX) + 1,
                    u64::MAX,
                ],
                0,
                NodeStorage::LitU64,
            ),
        },
        Width {
            literal: "LitI32",
            diagnostic: "i32",
            binary_cases: BINARY_DEPTH,
            unary_cases: UNARY_DEPTH,
            binary_draw: Draw::Strided {
                multiplier: 41,
                addend: 17,
            },
            unary_draw: Draw::Strided {
                multiplier: 29,
                addend: 13,
            },
            binary_ops_floor: 23,
            unary_ops_floor: 1,
            exception: "",
            corpus: || i32_corpus().into_iter().map(NodeStorage::LitI32).collect(),
            binary: i32_binary,
            unary: i32_unary,
            edges: signed_division_edges(),
        },
        Width {
            literal: "LitF32",
            diagnostic: "f32",
            binary_cases: BINARY_DEPTH,
            unary_cases: UNARY_DEPTH,
            binary_draw: Draw::Strided {
                multiplier: 29,
                addend: 5,
            },
            unary_draw: Draw::Strided {
                multiplier: 13,
                addend: 7,
            },
            binary_ops_floor: 12,
            unary_ops_floor: 3,
            exception: "",
            corpus: || f32_corpus().into_iter().map(NodeStorage::LitF32).collect(),
            binary: f32_binary,
            unary: f32_unary,
            edges: float_divisor_edges(),
        },
        Width {
            literal: "LitBool",
            diagnostic: "bool",
            binary_cases: 4,
            unary_cases: 2,
            binary_draw: Draw::Exhaustive,
            unary_draw: Draw::Exhaustive,
            binary_ops_floor: 4,
            unary_ops_floor: 1,
            exception: "two inhabitants: every ordered pair and every value is \
                        four cases and two cases, so sampling 4096 would only \
                        repeat them",
            corpus: || vec![NodeStorage::LitBool(false), NodeStorage::LitBool(true)],
            binary: bool_binary,
            unary: bool_unary,
            edges: Vec::new(),
        },
    ]
}

/// `Div` and `Mod` against a zero divisor, for a width that totalizes them.
fn divisor_edges<T: Copy>(
    lefts: &[T],
    zero: T,
    literal: fn(T) -> NodeStorage,
) -> Vec<(BinOp, NodeStorage, NodeStorage)> {
    let mut edges = Vec::with_capacity(lefts.len() * 2);
    for &left in lefts {
        for op in [BinOp::Div, BinOp::Mod] {
            edges.push((op, literal(left), literal(zero)));
        }
    }
    edges
}

/// The signed cases the target text leaves undefined.
fn signed_division_edges() -> Vec<(BinOp, NodeStorage, NodeStorage)> {
    let mut edges = Vec::new();
    for (left, right) in [
        (0, 0),
        (1, 0),
        (-1, 0),
        (i32::MIN, 0),
        (i32::MAX, 0),
        (i32::MIN, -1),
    ] {
        for op in [BinOp::Div, BinOp::Mod] {
            edges.push((op, NodeStorage::LitI32(left), NodeStorage::LitI32(right)));
        }
    }
    edges
}

/// Float division by zero: defined, and not the same answer for every numerator.
fn float_divisor_edges() -> Vec<(BinOp, NodeStorage, NodeStorage)> {
    [1.0f32, -1.0, 0.0, -0.0, f32::INFINITY, f32::NAN]
        .into_iter()
        .map(|left| {
            (
                BinOp::Div,
                NodeStorage::LitF32(left),
                NodeStorage::LitF32(0.0),
            )
        })
        .collect()
}

fn u32_binary(op: BinOp, left: &NodeStorage, right: &NodeStorage) -> Option<Expected> {
    let (NodeStorage::LitU32(left), NodeStorage::LitU32(right)) = (left, right) else {
        return None;
    };
    let (left, right) = (*left, *right);
    let value = match op {
        BinOp::Add | BinOp::WrappingAdd => IrValue::U32(left.wrapping_add(right)),
        BinOp::Sub | BinOp::WrappingSub => IrValue::U32(left.wrapping_sub(right)),
        BinOp::Mul => IrValue::U32(left.wrapping_mul(right)),
        BinOp::Div => IrValue::U32(left.checked_div(right).unwrap_or(u32::MAX)),
        BinOp::Mod => IrValue::U32(left.checked_rem(right).unwrap_or(0)),
        BinOp::BitAnd => IrValue::U32(left & right),
        BinOp::BitOr => IrValue::U32(left | right),
        BinOp::BitXor => IrValue::U32(left ^ right),
        BinOp::Shl => IrValue::U32(left.wrapping_shl(right & 31)),
        BinOp::Shr => IrValue::U32(left.wrapping_shr(right & 31)),
        BinOp::Eq => IrValue::Bool(left == right),
        BinOp::Ne => IrValue::Bool(left != right),
        BinOp::Lt => IrValue::Bool(left < right),
        BinOp::Le => IrValue::Bool(left <= right),
        BinOp::Gt => IrValue::Bool(left > right),
        BinOp::Ge => IrValue::Bool(left >= right),
        BinOp::Min => IrValue::U32(left.min(right)),
        BinOp::Max => IrValue::U32(left.max(right)),
        BinOp::SaturatingAdd => IrValue::U32(left.saturating_add(right)),
        BinOp::SaturatingSub => IrValue::U32(left.saturating_sub(right)),
        BinOp::SaturatingMul => IrValue::U32(left.saturating_mul(right)),
        BinOp::AbsDiff => IrValue::U32(left.abs_diff(right)),
        BinOp::RotateLeft => IrValue::U32(left.rotate_left(right & 31)),
        BinOp::RotateRight => IrValue::U32(left.rotate_right(right & 31)),
        BinOp::MulHigh => {
            IrValue::U32((u64::from(left).wrapping_mul(u64::from(right)) >> 32) as u32)
        }
        BinOp::And => IrValue::Bool(left != 0 && right != 0),
        BinOp::Or => IrValue::Bool(left != 0 || right != 0),
        _ => return None,
    };
    Some(Expected::Value(value))
}

fn u32_unary(op: &UnOp, operand: &NodeStorage) -> Option<Expected> {
    let NodeStorage::LitU32(value) = operand else {
        return None;
    };
    let value = *value;
    let result = match op {
        UnOp::BitNot => IrValue::U32(!value),
        UnOp::LogicalNot => IrValue::Bool(value == 0),
        UnOp::Popcount => IrValue::U32(value.count_ones()),
        UnOp::Clz => IrValue::U32(value.leading_zeros()),
        UnOp::Ctz => IrValue::U32(value.trailing_zeros()),
        UnOp::ReverseBits => IrValue::U32(value.reverse_bits()),
        _ => return None,
    };
    Some(Expected::Value(result))
}

fn u64_binary(op: BinOp, left: &NodeStorage, right: &NodeStorage) -> Option<Expected> {
    let (NodeStorage::LitU64(left), NodeStorage::LitU64(right)) = (left, right) else {
        return None;
    };
    let (left, right) = (*left, *right);
    let value = match op {
        BinOp::Add | BinOp::WrappingAdd => IrValue::U64(left.wrapping_add(right)),
        BinOp::Sub | BinOp::WrappingSub => IrValue::U64(left.wrapping_sub(right)),
        BinOp::Mul => IrValue::U64(left.wrapping_mul(right)),
        BinOp::Div => IrValue::U64(left.checked_div(right).unwrap_or(u64::MAX)),
        BinOp::Mod => IrValue::U64(left.checked_rem(right).unwrap_or(0)),
        BinOp::BitAnd => IrValue::U64(left & right),
        BinOp::BitOr => IrValue::U64(left | right),
        BinOp::BitXor => IrValue::U64(left ^ right),
        BinOp::Shl => IrValue::U64(left.wrapping_shl((right & 63) as u32)),
        BinOp::Shr => IrValue::U64(left.wrapping_shr((right & 63) as u32)),
        BinOp::Eq => IrValue::Bool(left == right),
        BinOp::Ne => IrValue::Bool(left != right),
        BinOp::Lt => IrValue::Bool(left < right),
        BinOp::Le => IrValue::Bool(left <= right),
        BinOp::Gt => IrValue::Bool(left > right),
        BinOp::Ge => IrValue::Bool(left >= right),
        BinOp::Min => IrValue::U64(left.min(right)),
        BinOp::Max => IrValue::U64(left.max(right)),
        BinOp::SaturatingAdd => IrValue::U64(left.saturating_add(right)),
        BinOp::SaturatingSub => IrValue::U64(left.saturating_sub(right)),
        BinOp::SaturatingMul => IrValue::U64(left.saturating_mul(right)),
        BinOp::AbsDiff => IrValue::U64(left.abs_diff(right)),
        BinOp::MulHigh => {
            IrValue::U64((u128::from(left).wrapping_mul(u128::from(right)) >> 64) as u64)
        }
        BinOp::And => IrValue::Bool(left != 0 && right != 0),
        BinOp::Or => IrValue::Bool(left != 0 || right != 0),
        _ => return None,
    };
    Some(Expected::Value(value))
}

fn u64_unary(op: &UnOp, operand: &NodeStorage) -> Option<Expected> {
    let NodeStorage::LitU64(value) = operand else {
        return None;
    };
    let value = *value;
    let result = match op {
        UnOp::Negate => IrValue::U64(0u64.wrapping_sub(value)),
        UnOp::BitNot => IrValue::U64(!value),
        UnOp::LogicalNot => IrValue::Bool(value == 0),
        UnOp::Popcount => IrValue::U64(u64::from(value.count_ones())),
        UnOp::Clz => IrValue::U64(u64::from(value.leading_zeros())),
        UnOp::Ctz => IrValue::U64(u64::from(value.trailing_zeros())),
        UnOp::ReverseBits => IrValue::U64(value.reverse_bits()),
        _ => return None,
    };
    Some(Expected::Value(result))
}

fn i32_binary(op: BinOp, left: &NodeStorage, right: &NodeStorage) -> Option<Expected> {
    let (NodeStorage::LitI32(left), NodeStorage::LitI32(right)) = (left, right) else {
        return None;
    };
    let (left, right) = (*left, *right);
    let undefined = right == 0 || (left == i32::MIN && right == -1);
    let value = match op {
        BinOp::Add | BinOp::WrappingAdd => IrValue::I32(left.wrapping_add(right)),
        BinOp::Sub | BinOp::WrappingSub => IrValue::I32(left.wrapping_sub(right)),
        BinOp::Mul => IrValue::I32(left.wrapping_mul(right)),
        BinOp::Div if undefined => return Some(Expected::Refused(UNDEFINED_SIGNED_DIVISION)),
        BinOp::Mod if undefined => return Some(Expected::Refused(UNDEFINED_SIGNED_DIVISION)),
        BinOp::Div => IrValue::I32(left / right),
        BinOp::Mod => IrValue::I32(left % right),
        BinOp::BitAnd => IrValue::I32(left & right),
        BinOp::BitOr => IrValue::I32(left | right),
        BinOp::BitXor => IrValue::I32(left ^ right),
        BinOp::Shl => IrValue::I32(left.wrapping_shl(u32::from_ne_bytes(right.to_ne_bytes()) & 31)),
        BinOp::Shr => IrValue::I32(left.wrapping_shr(u32::from_ne_bytes(right.to_ne_bytes()) & 31)),
        BinOp::Eq => IrValue::Bool(left == right),
        BinOp::Ne => IrValue::Bool(left != right),
        BinOp::Lt => IrValue::Bool(left < right),
        BinOp::Le => IrValue::Bool(left <= right),
        BinOp::Gt => IrValue::Bool(left > right),
        BinOp::Ge => IrValue::Bool(left >= right),
        BinOp::Min => IrValue::I32(left.min(right)),
        BinOp::Max => IrValue::I32(left.max(right)),
        BinOp::SaturatingAdd => IrValue::I32(left.saturating_add(right)),
        BinOp::SaturatingSub => IrValue::I32(left.saturating_sub(right)),
        BinOp::SaturatingMul => IrValue::I32(left.saturating_mul(right)),
        _ => return None,
    };
    Some(Expected::Value(value))
}

fn i32_unary(op: &UnOp, operand: &NodeStorage) -> Option<Expected> {
    let NodeStorage::LitI32(value) = operand else {
        return None;
    };
    match op {
        UnOp::Negate => Some(Expected::Value(IrValue::I32(value.wrapping_neg()))),
        _ => None,
    }
}

fn f32_binary(op: BinOp, left: &NodeStorage, right: &NodeStorage) -> Option<Expected> {
    let (NodeStorage::LitF32(left), NodeStorage::LitF32(right)) = (left, right) else {
        return None;
    };
    let (left, right) = (canonical_f32(*left), canonical_f32(*right));
    let value = match op {
        BinOp::Add => IrValue::F32(canonical_f32(left + right)),
        BinOp::Sub => IrValue::F32(canonical_f32(left - right)),
        BinOp::Mul => IrValue::F32(canonical_f32(left * right)),
        BinOp::Div => IrValue::F32(canonical_f32(left / right)),
        BinOp::Eq => IrValue::Bool(
            left.partial_cmp(&right)
                .is_some_and(std::cmp::Ordering::is_eq),
        ),
        BinOp::Ne => IrValue::Bool(
            left.partial_cmp(&right)
                .is_none_or(|ordering| !ordering.is_eq()),
        ),
        BinOp::Lt => IrValue::Bool(left < right),
        BinOp::Le => IrValue::Bool(left <= right),
        BinOp::Gt => IrValue::Bool(left > right),
        BinOp::Ge => IrValue::Bool(left >= right),
        BinOp::Min => IrValue::F32(canonical_f32(left.min(right))),
        BinOp::Max => IrValue::F32(canonical_f32(left.max(right))),
        _ => return None,
    };
    Some(Expected::Value(value))
}

fn f32_unary(op: &UnOp, operand: &NodeStorage) -> Option<Expected> {
    let NodeStorage::LitF32(value) = operand else {
        return None;
    };
    let value = canonical_f32(*value);
    let result = match op {
        UnOp::Negate => IrValue::F32(canonical_f32(-value)),
        UnOp::InverseSqrt => IrValue::F32(canonical_f32(1.0 / value.sqrt())),
        UnOp::Reciprocal => IrValue::F32(canonical_f32(1.0 / value)),
        _ => return None,
    };
    Some(Expected::Value(result))
}

fn bool_binary(op: BinOp, left: &NodeStorage, right: &NodeStorage) -> Option<Expected> {
    let (NodeStorage::LitBool(left), NodeStorage::LitBool(right)) = (left, right) else {
        return None;
    };
    let (left, right) = (*left, *right);
    let value = match op {
        BinOp::And => left && right,
        BinOp::Or => left || right,
        BinOp::Eq => left == right,
        BinOp::Ne => left != right,
        _ => return None,
    };
    Some(Expected::Value(IrValue::Bool(value)))
}

fn bool_unary(op: &UnOp, operand: &NodeStorage) -> Option<Expected> {
    let NodeStorage::LitBool(value) = operand else {
        return None;
    };
    match op {
        UnOp::LogicalNot => Some(Expected::Value(IrValue::Bool(!*value))),
        _ => None,
    }
}

fn binary_result(op: BinOp, left: &NodeStorage, right: &NodeStorage) -> ReferenceResult {
    let graph = vec![
        (NodeId(0), left.clone()),
        (NodeId(1), right.clone()),
        (
            NodeId(2),
            NodeStorage::BinOp {
                op,
                left: NodeId(0),
                right: NodeId(1),
            },
        ),
    ];
    run_storage_graph(&graph, &[NodeId(2)]).map(|values| values[0])
}

fn unary_result(op: &UnOp, operand: &NodeStorage) -> ReferenceResult {
    let graph = vec![
        (NodeId(0), operand.clone()),
        (
            NodeId(1),
            NodeStorage::UnOp {
                op: op.clone(),
                operand: NodeId(0),
            },
        ),
    ];
    run_storage_graph(&graph, &[NodeId(1)]).map(|values| values[0])
}

type ReferenceResult = Result<IrValue, ReferenceError>;

/// Compare one case against its declared expectation.
///
/// `context` is only rendered on failure: it runs once per case at sweep depth,
/// and formatting it eagerly costs more than the evaluation it describes.
fn assert_case(actual: ReferenceResult, expected: &Expected, context: impl Fn() -> String) {
    match (actual, expected) {
        (Ok(actual), Expected::Value(expected)) => match (actual, *expected) {
            (IrValue::F32(actual), IrValue::F32(expected)) => assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "Fix: f32 result bits differ for {}",
                context()
            ),
            (actual, expected) => {
                assert_eq!(actual, expected, "Fix: result differs for {}", context());
            }
        },
        (Ok(actual), Expected::Refused(marker)) => panic!(
            "Fix: {} must be refused naming `{marker}`; it returned {actual:?}.",
            context()
        ),
        (Err(error), Expected::Refused(marker)) => assert!(
            error.to_string().contains(marker),
            "Fix: {} must be refused naming `{marker}`: {error}",
            context()
        ),
        (Err(error), Expected::Value(expected)) => {
            panic!("Fix: {} must evaluate to {expected:?}: {error}", context())
        }
    }
}

#[test]
fn every_frozen_scalar_literal_has_a_matrix_row() {
    let frozen = frozen_literal_variants();
    let declared: BTreeSet<&str> = widths().iter().map(|row| row.literal).collect();

    let missing: Vec<&String> = frozen
        .iter()
        .filter(|variant| !declared.contains(variant.as_str()))
        .collect();
    assert!(
        missing.is_empty(),
        "Fix: the scalar matrix has no row for NodeStorage variant(s) {missing:?}. \
         Add a row declaring that width's corpus, operations and case counts; \
         until then the oracle's semantics for it are unswept."
    );

    let unknown: Vec<&&str> = declared
        .iter()
        .filter(|variant| !frozen.contains(**variant))
        .collect();
    assert!(
        unknown.is_empty(),
        "Fix: the scalar matrix declares row(s) {unknown:?} that the frozen public \
         surface has no NodeStorage variant for. Remove the row, or refresh the \
         snapshot with scripts/check_public_api_snapshot.sh --refresh vyre-foundation."
    );
}

#[test]
fn every_row_sweeps_the_full_depth_or_declares_why_not() {
    let mut short = Vec::new();
    for row in widths() {
        let corpus = (row.corpus)();
        let len = corpus.len();
        assert!(
            len > 1,
            "Fix: the {} row has a corpus of {len} value(s); a sweep needs at least two.",
            row.literal
        );

        assert_depth(
            &row,
            row.binary_draw,
            row.binary_cases,
            len,
            len * len,
            BINARY_DEPTH,
            "binary",
        );
        assert_depth(
            &row,
            row.unary_draw,
            row.unary_cases,
            len,
            len,
            UNARY_DEPTH,
            "unary",
        );

        let sampled = matches!(row.binary_draw, Draw::Strided { .. })
            && matches!(row.unary_draw, Draw::Strided { .. });
        assert_eq!(
            sampled,
            row.exception.is_empty(),
            "Fix: the {} row must carry an exception reason exactly when it stops sampling. \
             A row that samples the full depth has nothing to explain, and a row that does \
             not must say why.",
            row.literal
        );

        let binary = row.binary_ops(&corpus).len();
        let unary = row.unary_ops(&corpus).len();
        if binary < row.binary_ops_floor || unary < row.unary_ops_floor {
            short.push(format!(
                "{} declares {binary} binary and {unary} unary against a floor of {} and {}",
                row.literal, row.binary_ops_floor, row.unary_ops_floor
            ));
        }
    }
    assert!(
        short.is_empty(),
        "Fix: {short:?}. Coverage this sweep already had must not be dropped; lower a floor \
         only together with the reason the oracle stopped defining the operation."
    );
}

/// Check one arity's case count against the row's draw.
fn assert_depth(
    row: &Width,
    draw: Draw,
    cases: usize,
    len: usize,
    exhaustive_cases: usize,
    depth: usize,
    arity: &str,
) {
    match draw {
        Draw::Strided { multiplier, .. } => {
            assert_eq!(
                cases, depth,
                "Fix: the {} row samples {cases} {arity} cases against a depth of {depth}. \
                 Raise it, or enumerate the corpus with an exhaustive draw and say why.",
                row.literal
            );
            assert_eq!(
                gcd(multiplier, len),
                1,
                "Fix: the {} row draws {arity} operands with stride {multiplier} over a corpus \
                 of {len}; a stride sharing a factor with the length revisits a subset of it.",
                row.literal
            );
            assert!(
                cases >= len,
                "Fix: the {} row runs {cases} {arity} cases over a corpus of {len}; a sampled \
                 sweep must be at least as long as the corpus to reach every value.",
                row.literal
            );
        }
        Draw::Exhaustive => assert_eq!(
            cases, exhaustive_cases,
            "Fix: the {} row enumerates {cases} {arity} cases where its corpus has \
             {exhaustive_cases} to enumerate.",
            row.literal
        ),
    }
}

fn gcd(left: usize, right: usize) -> usize {
    if right == 0 {
        left
    } else {
        gcd(right, left % right)
    }
}

#[test]
fn declared_binary_semantics_match_the_storage_graph_oracle() {
    for row in widths() {
        let corpus = (row.corpus)();
        let ops = row.binary_ops(&corpus);
        let mut checked = 0usize;
        for case in 0..row.binary_cases {
            let (left, right) = row.binary_draw.pair(case, corpus.len());
            let (left, right) = (&corpus[left], &corpus[right]);
            for &op in &ops {
                let Some(expected) = (row.binary)(op, left, right) else {
                    continue;
                };
                assert_case(binary_result(op, left, right), &expected, || {
                    format!(
                        "{} {op:?} case {case} left={left:?} right={right:?}",
                        row.literal
                    )
                });
                checked += 1;
            }
        }
        assert_eq!(
            checked,
            row.binary_cases * ops.len(),
            "Fix: the {} row swept {checked} binary cases for {} operations over {} cases; \
             an operation stopped being declared part way through the sweep.",
            row.literal,
            ops.len(),
            row.binary_cases
        );
    }
}

#[test]
fn declared_unary_semantics_match_the_storage_graph_oracle() {
    for row in widths() {
        let corpus = (row.corpus)();
        let ops = row.unary_ops(&corpus);
        let mut checked = 0usize;
        for case in 0..row.unary_cases {
            let operand = &corpus[row.unary_draw.single(case, corpus.len())];
            for op in &ops {
                let Some(expected) = (row.unary)(op, operand) else {
                    continue;
                };
                assert_case(unary_result(op, operand), &expected, || {
                    format!("{} {op:?} case {case} value={operand:?}", row.literal)
                });
                checked += 1;
            }
        }
        assert_eq!(
            checked,
            row.unary_cases * ops.len(),
            "Fix: the {} row swept {checked} unary cases for {} operations over {} cases; \
             an operation stopped being declared part way through the sweep.",
            row.literal,
            ops.len(),
            row.unary_cases
        );
    }
}

#[test]
fn declared_edge_cases_match_the_storage_graph_oracle() {
    for row in widths() {
        for (op, left, right) in &row.edges {
            let Some(expected) = (row.binary)(*op, left, right) else {
                panic!(
                    "Fix: the {} row lists an edge case for {op:?}, which it does not declare.",
                    row.literal
                )
            };
            assert_case(binary_result(*op, left, right), &expected, || {
                format!("{} edge {op:?} left={left:?} right={right:?}", row.literal)
            });
        }
    }
}

#[test]
fn operations_a_width_does_not_declare_are_refused_by_name() {
    for row in widths() {
        let corpus = (row.corpus)();
        let (left, right) = row.probe(&corpus);
        let declared = row.binary_ops(&corpus);
        for op in builtin_bin_ops() {
            if declared.contains(&op) {
                continue;
            }
            assert_refused(
                binary_result(op, left, right),
                row.diagnostic,
                "binary",
                || format!("{} {op:?}", row.literal),
            );
        }

        let declared = row.unary_ops(&corpus);
        for op in builtin_un_ops() {
            if declared.contains(&op) {
                continue;
            }
            assert_refused(unary_result(&op, left), row.diagnostic, "unary", || {
                format!("{} {op:?}", row.literal)
            });
        }
    }
}

/// An operation outside a width's declared surface must fail, and the failure
/// must name the width and the arity so the reader knows which row to widen.
fn assert_refused(
    actual: ReferenceResult,
    diagnostic: &str,
    arity: &str,
    context: impl Fn() -> String,
) {
    match actual {
        Ok(value) => panic!(
            "Fix: {} now evaluates to {value:?}. The oracle gained semantics this matrix \
             does not declare: add the expectation to that width's row.",
            context()
        ),
        Err(error) => {
            let message = error.to_string();
            let expected = format!("{UNSUPPORTED} {diagnostic} {arity} operation");
            assert!(
                message.contains(&expected),
                "Fix: {} must be refused with `{expected}`: {message}",
                context()
            );
        }
    }
}

#[test]
fn operands_of_different_widths_are_refused_as_a_type_mismatch() {
    let rows = widths();
    let representatives: Vec<(&str, NodeStorage)> = rows
        .iter()
        .map(|row| (row.literal, (row.corpus)()[0].clone()))
        .collect();

    let mut checked = 0usize;
    for (left_name, left) in &representatives {
        for (right_name, right) in &representatives {
            if left_name == right_name {
                continue;
            }
            let error = binary_result(BinOp::Add, left, right).err().unwrap_or_else(|| {
                panic!("Fix: {left_name} added to {right_name} must be refused as a type mismatch.")
            });
            assert!(
                error.to_string().contains(TYPE_MISMATCH),
                "Fix: {left_name} added to {right_name} must name `{TYPE_MISMATCH}`: {error}"
            );
            checked += 1;
        }
    }
    let rows = representatives.len();
    assert_eq!(
        checked,
        rows * (rows - 1),
        "Fix: every ordered pair of distinct widths must be checked for the mismatch refusal."
    );
}

/// The `NodeStorage` literal variants the frozen public-API snapshot records.
///
/// `scripts/check_public_api_snapshot.sh` regenerates the snapshot from rustdoc
/// and a byte-stability gate holds it equal to the crate's real surface, so a new
/// scalar literal reaches this matrix through the gate that already forces a
/// snapshot refresh.
fn frozen_literal_variants() -> BTreeSet<String> {
    let path = foundation_api_snapshot();
    let snapshot = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "Fix: the public-API snapshot at {} must be readable to enumerate the scalar literals: {error}",
            path.display()
        )
    });
    let names: BTreeSet<String> = snapshot
        .lines()
        .filter_map(|line| line.strip_prefix("pub vyre_foundation::ir::NodeStorage::Lit"))
        .filter_map(|rest| rest.split('(').next())
        .filter(|name| !name.is_empty() && !name.contains(':'))
        .map(|name| format!("Lit{name}"))
        .collect();
    assert!(
        !names.is_empty(),
        "Fix: the public-API snapshot at {} lists no NodeStorage literal variants. Refresh it \
         with scripts/check_public_api_snapshot.sh --refresh vyre-foundation.",
        path.display()
    );
    names
}

fn foundation_api_snapshot() -> PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest
        .ancestors()
        .map(|directory| directory.join("docs/public-api/vyre-foundation.txt"))
        .find(|candidate| candidate.is_file())
        .unwrap_or_else(|| {
            panic!(
                "Fix: no docs/public-api/vyre-foundation.txt above {}. The scalar matrix enumerates \
                 the frozen literal surface from that snapshot.",
                manifest.display()
            )
        })
}
