//! A schedule may reorder the combines of a program only when every combine it
//! performs is associative and commutative.
//!
//! Reordering is what a work queue, a spatial partition, and an overlapped
//! pipeline all do to the order invocations reach a shared accumulator. Applying
//! one to a rounding accumulation returns a different number from the program
//! that was submitted, and the difference is data-dependent, so it shows up as
//! an accuracy report and not as a failure.
//!
//! The operator vocabularies are enumerated here through their frozen wire tags
//! rather than listed, so an operator added to a vocabulary is covered by these
//! cases without an edit. Which operator applies which combine, and which IR
//! variant combines at all, are exhaustive matches with no catch-all arm in
//! `vyre-spec` and `vyre-foundation/src/visit`, so a new operator or variant
//! fails to compile there instead of defaulting to reorderable here.

#[path = "support/opaque_echo_extension.rs"]
mod opaque_echo_extension;

use std::sync::Arc;

use opaque_echo_extension::{EchoExpr, EchoNode};

use vyre_foundation::algebraic_reordering::{reordering_class, ReorderingClass};
use vyre_foundation::ir::{
    AtomicOp, BufferAccess, BufferDecl, CollectiveOp, CommGroup, DataType, Expr, Ident,
    MemoryOrdering, Node, Program, SubgroupReduceOp,
};

/// Every element type an accumulation is tested over, paired with whether its
/// arithmetic is exact.
const ELEMENT_TYPES: [(DataType, bool); 4] = [
    (DataType::U32, true),
    (DataType::I32, true),
    (DataType::F32, false),
    (DataType::F16, false),
];

/// A program whose one accumulator buffer has the given element type.
fn program(element: &DataType, entry: Vec<Node>) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("acc", 0, BufferAccess::ReadWrite, element.clone()).with_count(4),
            BufferDecl::output("out", 1, element.clone()).with_count(4),
        ],
        [64, 1, 1],
        entry,
    )
}

fn atomic(op: AtomicOp, element: &DataType) -> Program {
    program(
        element,
        vec![
            Node::let_bind(
                "old",
                Expr::Atomic {
                    op,
                    buffer: Ident::from("acc"),
                    index: Box::new(Expr::u32(0)),
                    expected: None,
                    value: Box::new(Expr::u32(1)),
                    ordering: MemoryOrdering::Relaxed,
                },
            ),
            Node::Return,
        ],
    )
}

fn subgroup_reduce(op: SubgroupReduceOp, element: &DataType) -> Program {
    program(
        element,
        vec![
            Node::let_bind(
                "total",
                Expr::SubgroupReduce {
                    op,
                    value: Box::new(Expr::load("acc", Expr::u32(0))),
                },
            ),
            Node::Return,
        ],
    )
}

fn all_reduce(op: CollectiveOp, element: &DataType) -> Program {
    program(
        element,
        vec![
            Node::AllReduce {
                buffer: Ident::from("acc"),
                op,
                group: CommGroup::WORLD,
            },
            Node::Return,
        ],
    )
}

/// Every builtin subgroup reduction, recovered from the frozen wire tags rather
/// than listed, so a reduction added to the vocabulary appears here.
fn every_subgroup_reduce_op() -> Vec<SubgroupReduceOp> {
    let ops: Vec<SubgroupReduceOp> = (0..=u8::MAX)
        .filter_map(|tag| SubgroupReduceOp::from_wire_tag(tag).ok())
        .collect();
    assert!(
        ops.len() >= 7,
        "the subgroup reduction vocabulary lost operators: {ops:?}"
    );
    ops
}

/// Every builtin collective reduction, recovered the same way.
fn every_collective_op() -> Vec<CollectiveOp> {
    let ops: Vec<CollectiveOp> = (0..=u8::MAX)
        .filter_map(|tag| CollectiveOp::from_wire_tag(tag).ok())
        .collect();
    assert!(
        ops.len() >= 6,
        "the collective vocabulary lost operators: {ops:?}"
    );
    ops
}

#[test]
fn an_exact_accumulation_reorders_and_a_rounding_one_does_not() {
    for (element, exact) in &ELEMENT_TYPES {
        let class = reordering_class(&atomic(AtomicOp::Add, element));
        let expected = if *exact {
            ReorderingClass::Reassociable
        } else {
            ReorderingClass::Ordered
        };
        assert_eq!(
            class, expected,
            "atomic add over {element:?} classified {class:?}"
        );
    }
}

#[test]
fn every_subgroup_reduction_reorders_exactly_when_its_combine_is_exact() {
    for op in every_subgroup_reduce_op() {
        for (element, exact) in &ELEMENT_TYPES {
            let class = reordering_class(&subgroup_reduce(op, element));
            let permitted = *exact || op.combine().is_bitwise();
            assert_eq!(
                class.permits_reordering(),
                permitted,
                "{op:?} over {element:?} classified {class:?}"
            );
        }
    }
}

#[test]
fn every_collective_reduction_reorders_exactly_when_its_combine_is_exact() {
    for op in every_collective_op() {
        for (element, exact) in &ELEMENT_TYPES {
            let class = reordering_class(&all_reduce(op, element));
            let permitted = *exact || op.combine().is_bitwise();
            assert_eq!(
                class.permits_reordering(),
                permitted,
                "{op:?} over {element:?} classified {class:?}"
            );
        }
    }
}

#[test]
fn an_atomic_whose_result_depends_on_the_displaced_value_is_ordered() {
    for op in [
        AtomicOp::Exchange,
        AtomicOp::CompareExchange,
        AtomicOp::CompareExchangeWeak,
        AtomicOp::FetchNand,
        AtomicOp::LruUpdate,
    ] {
        for (element, _) in &ELEMENT_TYPES {
            let class = reordering_class(&atomic(op, element));
            assert_eq!(
                class,
                ReorderingClass::Ordered,
                "{op:?} over {element:?} classified {class:?}"
            );
        }
    }
}

#[test]
fn a_program_that_combines_nothing_carries_no_reordering_constraint() {
    let program = program(
        &DataType::F32,
        vec![
            Node::store("out", Expr::u32(0), Expr::load("acc", Expr::u32(0))),
            Node::Return,
        ],
    );

    assert_eq!(reordering_class(&program), ReorderingClass::NoCombine);
}

#[test]
fn one_rounding_accumulation_orders_a_program_whose_other_combines_are_exact() {
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("counter", 0, BufferAccess::ReadWrite, DataType::U32).with_count(4),
            BufferDecl::storage("acc", 1, BufferAccess::ReadWrite, DataType::F32).with_count(4),
            BufferDecl::output("out", 2, DataType::U32).with_count(4),
        ],
        [64, 1, 1],
        vec![
            Node::let_bind(
                "hits",
                Expr::Atomic {
                    op: AtomicOp::Add,
                    buffer: Ident::from("counter"),
                    index: Box::new(Expr::u32(0)),
                    expected: None,
                    value: Box::new(Expr::u32(1)),
                    ordering: MemoryOrdering::Relaxed,
                },
            ),
            Node::let_bind(
                "sum",
                Expr::Atomic {
                    op: AtomicOp::Add,
                    buffer: Ident::from("acc"),
                    index: Box::new(Expr::u32(0)),
                    expected: None,
                    value: Box::new(Expr::f32(1.0)),
                    ordering: MemoryOrdering::Relaxed,
                },
            ),
            Node::Return,
        ],
    );

    assert_eq!(reordering_class(&program), ReorderingClass::Ordered);
}

#[test]
fn an_opaque_expression_is_ordered_because_its_combine_is_unknown() {
    let program = program(
        &DataType::U32,
        vec![
            Node::store(
                "out",
                Expr::u32(0),
                Expr::Opaque(Arc::new(EchoExpr {
                    payload: b"unknown-combine".to_vec(),
                })),
            ),
            Node::Return,
        ],
    );

    assert_eq!(reordering_class(&program), ReorderingClass::Ordered);
}

#[test]
fn an_opaque_statement_is_ordered_because_its_combine_is_unknown() {
    let program = program(
        &DataType::U32,
        vec![
            Node::Opaque(Arc::new(EchoNode {
                payload: b"unknown-combine".to_vec(),
            })),
            Node::Return,
        ],
    );

    assert_eq!(reordering_class(&program), ReorderingClass::Ordered);
}

#[test]
fn a_combine_nested_in_a_loop_body_is_seen() {
    let inner = atomic(AtomicOp::Add, &DataType::F32);
    let nested = program(
        &DataType::F32,
        vec![
            Node::Loop {
                var: Ident::from("i"),
                from: Expr::u32(0),
                to: Expr::u32(4),
                body: inner.entry().to_vec(),
            },
            Node::Return,
        ],
    );

    assert_eq!(reordering_class(&nested), ReorderingClass::Ordered);
}
