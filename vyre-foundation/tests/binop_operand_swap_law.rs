//! One operator law answers every operand-reordering question.
//!
//! WHY this suite exists: operand-swap legality was decided three times. The
//! wire canonicalizer in `ir_inner::model::program::canonical` reordered 15
//! operators and sorted the non-literal operands of 11. The canonicalize pass
//! reordered 9 and sorted 7. The CSE key reordered 11. The three sets
//! disagreed on `WrappingAdd`, `Min`, `Max`, `AbsDiff`, `MulHigh`,
//! `SaturatingAdd` and `SaturatingMul`, so:
//!
//! - a program the wire canonicalizer had already reordered was not the
//!   program the canonicalize pass produced, and the pass reported canonical
//!   form on IR that the canonicalizer still rewrote;
//! - CSE missed `min(a, b)` against `min(b, a)` although the canonical wire
//!   form of the two was byte-identical, which is a redundant expression
//!   surviving the pass whose job is to remove it.
//!
//! `BinOp::operand_swap` is now the single answer and this suite holds all
//! three consumers to it.
//!
//! # Why the expected classification is written out here
//!
//! [`EXPECTED_SWAP`] states the law independently of the code that implements
//! it. Asserting a consumer against `op.commutes()` would compare the code to
//! itself: a classification that called `Sub` commutative would make CSE merge
//! `sub(a, b)` with `sub(b, a)` and the assertion would agree with both. The
//! table is held to the declared variant set at run time, so it cannot go
//! stale in silence.
//!
//! The operator set comes from [`vyre_test_support::bin_op_variants`], which
//! reads the `pub enum BinOp` declaration at run time, so a new operator turns
//! this suite RED until a fixture and a row exist for it. `operand_swap`
//! itself is an exhaustive match with no catch-all arm, so a new operator
//! cannot inherit a classification either.
//!
//! What this does NOT catch: whether the classification of a given operator is
//! mathematically right. `MulHigh` swapping bit-exactly is a claim about
//! unsigned multiply-high, not something a rewrite can verify.

use std::collections::BTreeMap;

use vyre::ir::{BinOp, BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::optimizer::passes::algebraic::canonicalize_engine;
use vyre_foundation::optimizer::passes::fusion_cse::cse::cse;
use vyre_foundation::transform::grid_sync_split::entry_sequence;
use vyre_spec::OperandSwap;
use vyre_test_support::bin_op_variants::{
    assert_covers_every_bin_op_variant, bin_op_variant_samples, variant_name,
};

/// What swapping each operator's operands does to its result.
///
/// `BitPreserving` is the exact-bits claim: integer and boolean operators, plus
/// `AbsDiff` and `MulHigh`, whose operands are interchangeable at the bit
/// level. `ValuePreserving` is the numeric operators whose float form can
/// differ in NaN payload after a swap while computing the same value.
/// Everything else is `Ordered`, extension operators included, because an
/// extension declares its own semantics and no law is derived for it.
const EXPECTED_SWAP: &[(&str, OperandSwap)] = &[
    ("Add", OperandSwap::ValuePreserving),
    ("Mul", OperandSwap::ValuePreserving),
    ("Min", OperandSwap::ValuePreserving),
    ("Max", OperandSwap::ValuePreserving),
    ("WrappingAdd", OperandSwap::BitPreserving),
    ("SaturatingAdd", OperandSwap::BitPreserving),
    ("SaturatingMul", OperandSwap::BitPreserving),
    ("MulHigh", OperandSwap::BitPreserving),
    ("AbsDiff", OperandSwap::BitPreserving),
    ("BitAnd", OperandSwap::BitPreserving),
    ("BitOr", OperandSwap::BitPreserving),
    ("BitXor", OperandSwap::BitPreserving),
    ("Eq", OperandSwap::BitPreserving),
    ("Ne", OperandSwap::BitPreserving),
    ("And", OperandSwap::BitPreserving),
    ("Or", OperandSwap::BitPreserving),
    ("Sub", OperandSwap::Ordered),
    ("Div", OperandSwap::Ordered),
    ("Mod", OperandSwap::Ordered),
    ("WrappingSub", OperandSwap::Ordered),
    ("SaturatingSub", OperandSwap::Ordered),
    ("Shl", OperandSwap::Ordered),
    ("Shr", OperandSwap::Ordered),
    ("RotateLeft", OperandSwap::Ordered),
    ("RotateRight", OperandSwap::Ordered),
    ("Lt", OperandSwap::Ordered),
    ("Gt", OperandSwap::Ordered),
    ("Le", OperandSwap::Ordered),
    ("Ge", OperandSwap::Ordered),
    ("Shuffle", OperandSwap::Ordered),
    ("Ballot", OperandSwap::Ordered),
    ("WaveReduce", OperandSwap::Ordered),
    ("WaveBroadcast", OperandSwap::Ordered),
    ("Opaque", OperandSwap::Ordered),
];

/// The stated classification of `op`, by declared variant name.
fn expected_swap(op: BinOp) -> OperandSwap {
    let name = variant_name(op);
    EXPECTED_SWAP
        .iter()
        .find(|(row, _)| *row == name)
        .map(|(_, swap)| *swap)
        .unwrap_or_else(|| {
            panic!("Fix: EXPECTED_SWAP has no row for {name}; state the law for the new operator")
        })
}

/// Whether the stated law lets a consumer swap `op`'s operands.
fn expected_commutes(op: BinOp) -> bool {
    !matches!(expected_swap(op), OperandSwap::Ordered)
}

/// Operators the probe programs can carry as a scalar binary expression.
///
/// A subgroup operator reads its result from the invocation's lane neighbours
/// rather than from its two operands, and an extension operator has no builtin
/// semantics, so neither shape is a rewrite probe. Both are still covered by
/// [`the_declared_law_matches_the_stated_law`].
fn swap_probe_operators() -> Vec<BinOp> {
    bin_op_variant_samples()
        .into_iter()
        .filter(|op| {
            !matches!(
                op,
                BinOp::Shuffle
                    | BinOp::Ballot
                    | BinOp::WaveReduce
                    | BinOp::WaveBroadcast
                    | BinOp::Opaque(_)
            )
        })
        .collect()
}

/// `op(left, right)`. `Expr` has no operator-parameterised binary
/// constructor, only the per-operator builders.
fn bin(op: BinOp, left: Expr, right: Expr) -> Expr {
    Expr::BinOp {
        op,
        left: Box::new(left),
        right: Box::new(right),
    }
}

/// `out[0] = op(a[0], 7)` with the two operands in the requested order.
fn literal_probe(op: BinOp, literal_first: bool) -> Program {
    let load = Expr::load("a", Expr::u32(0));
    let literal = Expr::u32(7);
    let (left, right) = if literal_first {
        (literal, load)
    } else {
        (load, literal)
    };
    probe_program(bin(op, left, right))
}

/// `out[0] = op(a[0], b[0])` with the two loads in the requested order.
fn load_probe(op: BinOp, reversed: bool) -> Program {
    let a = Expr::load("a", Expr::u32(0));
    let b = Expr::load("b", Expr::u32(0));
    let (left, right) = if reversed { (b, a) } else { (a, b) };
    probe_program(bin(op, left, right))
}

fn probe_program(value: Expr) -> Program {
    Program::wrapped(
        probe_buffers(),
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), value)],
    )
}

fn probe_buffers() -> Vec<BufferDecl> {
    vec![
        BufferDecl::read("a", 0, DataType::U32).with_count(64),
        BufferDecl::read("b", 1, DataType::U32).with_count(64),
        BufferDecl::read_write("out", 2, DataType::U32).with_count(64),
    ]
}

/// The fixture set is the declared operator set.
#[test]
fn the_probe_set_covers_every_declared_operator() {
    assert_covers_every_bin_op_variant(&bin_op_variant_samples());
}

/// The implemented classification is the stated one, in both directions.
#[test]
fn the_declared_law_matches_the_stated_law() {
    let stated: BTreeMap<&str, OperandSwap> = EXPECTED_SWAP.iter().copied().collect();
    assert_eq!(
        stated.len(),
        EXPECTED_SWAP.len(),
        "Fix: EXPECTED_SWAP names an operator twice"
    );

    for op in bin_op_variant_samples() {
        assert_eq!(
            op.operand_swap(),
            expected_swap(op),
            "Fix: BinOp::operand_swap disagrees with the law stated for {}",
            variant_name(op)
        );
    }

    let declared: Vec<String> = bin_op_variant_samples()
        .into_iter()
        .map(variant_name)
        .collect();
    let extra: Vec<&&str> = stated
        .keys()
        .filter(|name| !declared.iter().any(|declared| declared == *name))
        .collect();
    assert!(
        extra.is_empty(),
        "Fix: EXPECTED_SWAP states a law for operators vyre-spec no longer declares: {extra:?}"
    );
}

/// Bit-exact swap is the stronger claim of the two axes.
///
/// A classification that answered `commutes_bit_exactly` without answering
/// `commutes` would let the CSE key merge operands the wire form keeps
/// ordered.
#[test]
fn bit_exact_swap_implies_value_preserving_swap() {
    for op in bin_op_variant_samples() {
        assert!(
            !op.commutes_bit_exactly() || op.commutes(),
            "Fix: {} swaps bit-exactly but is not value-commutative; \
             BinOp::operand_swap classifies it as {:?}, which no consumer can act on",
            variant_name(op),
            op.operand_swap()
        );
    }
}

/// The canonicalize pass leaves the program in wire-canonical form.
///
/// This is the defect the three tables produced: the pass reported canonical
/// form while `Program::canonicalized` still had a reordering to make. The
/// comparison is structural because `Program::fingerprint` canonicalizes
/// first, so a fingerprint comparison would hold whatever the pass did.
#[test]
fn pass_output_is_already_wire_canonical() {
    for op in swap_probe_operators() {
        for (label, program) in [
            ("literal left", literal_probe(op, true)),
            ("literal right", literal_probe(op, false)),
            ("loads in order", load_probe(op, false)),
            ("loads reversed", load_probe(op, true)),
        ] {
            let after_pass = canonicalize_engine::run(program);
            let canonical = after_pass.canonicalized();
            assert_eq!(
                after_pass.entry(),
                canonical.entry(),
                "Fix: the canonicalize pass left {} ({label}) in a form the wire canonicalizer \
                 still rewrites; the two disagree about operand order",
                variant_name(op)
            );
        }
    }
}

/// A commuting operator's two operand orders reach one program.
#[test]
fn a_commuting_operator_has_one_canonical_operand_order() {
    for op in swap_probe_operators() {
        let literal_left = canonicalize_engine::run(literal_probe(op, true));
        let literal_right = canonicalize_engine::run(literal_probe(op, false));
        let merged = literal_left.entry() == literal_right.entry();
        assert_eq!(
            merged,
            expected_commutes(op),
            "Fix: {} is stated {:?}, but the canonicalize pass {} the two operand orders",
            variant_name(op),
            expected_swap(op),
            if merged { "merged" } else { "kept apart" }
        );
    }
}

/// Non-literal operands are sorted only where the swap is bit-exact.
///
/// A value-preserving swap over floats changes the NaN payload the program
/// produces, so sorting `b + a` into `a + b` is a rewrite the canonical form
/// declines. The comparison is against the wire canonicalizer, which is the
/// owner of the form the pass must land in.
#[test]
fn non_literal_operands_sort_only_under_a_bit_exact_swap() {
    for op in swap_probe_operators() {
        let in_order = canonicalize_engine::run(load_probe(op, false));
        let reversed = canonicalize_engine::run(load_probe(op, true));
        let sorted = in_order.entry() == reversed.entry();
        assert_eq!(
            sorted,
            matches!(expected_swap(op), OperandSwap::BitPreserving),
            "Fix: {} is stated {:?}, but the canonicalize pass {} its two non-literal operands",
            variant_name(op),
            expected_swap(op),
            if sorted { "sorted" } else { "left" }
        );
    }
}

/// CSE merges swapped operands exactly when the operator commutes.
///
/// `min(a, b)` and `min(b, a)` are one value, and before the law had one owner
/// the CSE key kept them apart while the wire form declared them identical.
#[test]
fn cse_merges_swapped_operands_for_a_commuting_operator() {
    for op in swap_probe_operators() {
        let program = Program::wrapped(
            probe_buffers(),
            [1, 1, 1],
            vec![
                Node::let_bind(
                    "forward",
                    bin(
                        op,
                        Expr::load("a", Expr::u32(0)),
                        Expr::load("b", Expr::u32(0)),
                    ),
                ),
                Node::let_bind(
                    "reversed",
                    bin(
                        op,
                        Expr::load("b", Expr::u32(0)),
                        Expr::load("a", Expr::u32(0)),
                    ),
                ),
                Node::store(
                    "out",
                    Expr::u32(0),
                    bin(BinOp::Add, Expr::var("forward"), Expr::var("reversed")),
                ),
            ],
        );
        let optimized = cse(program);
        let body = entry_sequence(&optimized);
        let Some(Node::Let { value, .. }) = body.get(1) else {
            panic!("Fix: the second entry statement is no longer the `reversed` binding: {body:?}");
        };
        let merged = matches!(value, Expr::Var(_));
        assert_eq!(
            merged,
            expected_commutes(op),
            "Fix: {} is stated {:?}, but CSE {} the swapped pair; the second binding is {value:?}",
            variant_name(op),
            expected_swap(op),
            if merged { "merged" } else { "kept" }
        );
    }
}
