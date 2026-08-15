//! Adversarial invariants for graph_view, canonicalize, and algebraic_law_registry.
//!
//! The programs come from `contract_cases::optimizer_program_corpus`, which
//! also owns the fixed-point-and-semantics scaffold. What canonicalize owes as
//! an optimizer entry point is asserted once, in the entry-point table in
//! `optimizer_idempotence_proptest`; what is left here is the part that is
//! about graph_view and the law registry rather than about canonicalize.

use proptest::prelude::*;
use vyre_foundation::algebraic_law_registry::{
    has_law, is_commutative, laws_for_op, AlgebraicLaw, AlgebraicLawRegistration,
};
use vyre_foundation::graph_view::{
    from_graph, to_graph, DataflowKind, GraphValidateError, NodeGraph,
};
use vyre_foundation::ir::{BinOp, Expr, Node, Program};
use vyre_foundation::optimizer::passes::algebraic::canonicalize_engine as canonicalize;
use vyre_foundation::transform::visit::for_each_node;

#[path = "contract_cases/optimizer_program_corpus.rs"]
mod corpus;

use corpus::{output_only_store, program_strategy, program_with_body, test_output_buffer};

inventory::submit! {
    AlgebraicLawRegistration::new("test::binop::add", AlgebraicLaw::Commutative)
}
inventory::submit! {
    AlgebraicLawRegistration::new("test::duplicate::commutative", AlgebraicLaw::Commutative)
}
inventory::submit! {
    AlgebraicLawRegistration::new("test::duplicate::commutative", AlgebraicLaw::Commutative)
}

fn raw_program_with_body(body: Vec<Node>) -> Program {
    Program::from_raw_parts(vec![test_output_buffer()], [1, 1, 1], body)
}

fn canonicalized_store_value(expr: Expr) -> Expr {
    let canonical = canonicalize::run(output_only_store(expr));
    let first = canonical
        .entry()
        .first()
        .expect("Fix: store_program always produces one root region");
    let store = match first {
        Node::Region { body, .. } => body
            .first()
            .expect("Fix: store_program root region must contain one store node"),
        other => other,
    };
    match store {
        Node::Store { value, .. } => value.clone(),
        other => panic!("Fix: expected canonicalized store node, got {other:?}"),
    }
}

/// BinOps where canonicalize may sort non-literal operands without
/// changing semantics. Add/Mul are excluded because IEEE-754 float
/// NaN payload propagation makes them non-commutative at the bit level.
fn safe_to_sort_nonliterals_binops() -> [BinOp; 7] {
    [
        BinOp::BitAnd,
        BinOp::BitOr,
        BinOp::BitXor,
        BinOp::Eq,
        BinOp::Ne,
        BinOp::And,
        BinOp::Or,
    ]
}

fn operand_ordered_binops() -> [BinOp; 4] {
    [BinOp::Sub, BinOp::Div, BinOp::Shl, BinOp::Shr]
}

fn malformed_graph_with_cycle() -> NodeGraph {
    let program = raw_program_with_body(vec![
        Node::store("out", Expr::u32(0), Expr::u32(1)),
        Node::store("out", Expr::u32(0), Expr::u32(2)),
    ]);
    let mut graph = to_graph(&program);
    // Add a backward edge to create a real cycle: 0->1 (original) + 1->0 (new).
    graph.edges.push(vyre_foundation::graph_view::DataEdge::new(
        1,
        0,
        vyre_foundation::graph_view::EdgeKind::Ordering,
    ));
    graph
}

fn malformed_graph_with_dangling_edge() -> NodeGraph {
    let program = raw_program_with_body(vec![
        Node::store("out", Expr::u32(0), Expr::u32(1)),
        Node::store("out", Expr::u32(0), Expr::u32(2)),
    ]);
    let mut graph = to_graph(&program);
    graph.edges[0].to = 999;
    graph
}

fn malformed_graph_with_orphan_phi() -> NodeGraph {
    let program = raw_program_with_body(vec![
        Node::store("out", Expr::u32(0), Expr::u32(1)),
        Node::store("out", Expr::u32(0), Expr::u32(2)),
    ]);
    let mut graph = to_graph(&program);
    graph.nodes[1].kind = DataflowKind::Phi(Vec::new());
    graph
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 100, .. ProptestConfig::default() })]

    #[test]
    fn graph_round_trip_is_byte_identical_under_canonicalize(program in program_strategy()) {
        let round_tripped = from_graph(to_graph(&program)).unwrap();
        let original = canonicalize::run(program);
        let lowered = canonicalize::run(round_tripped);
        prop_assert_eq!(
            original.to_wire().expect("Fix: generated Program must serialize"),
            lowered.to_wire().expect("Fix: graph round-trip Program must serialize"),
        );
    }
}

#[test]
fn malformed_cycle_graph_is_rejected_without_panic() {
    let result = from_graph(malformed_graph_with_cycle());
    assert!(
        matches!(result, Err(GraphValidateError::Cycle { .. })),
        "Fix: from_graph must return Result::Err for cyclic graph ids, not panic"
    );
}

#[test]
fn malformed_dangling_edge_graph_is_rejected_without_panic() {
    let result = from_graph(malformed_graph_with_dangling_edge());
    assert!(
        matches!(result, Err(GraphValidateError::DanglingEdge { .. })),
        "Fix: from_graph must return Result::Err for dangling graph edges, not panic"
    );
}

#[test]
fn malformed_orphan_phi_graph_is_rejected_without_panic() {
    let result = from_graph(malformed_graph_with_orphan_phi());
    assert!(
        matches!(result, Err(GraphValidateError::OrphanPhi { .. })),
        "Fix: from_graph must return Result::Err for orphan Phi nodes, not panic"
    );
}

#[test]
fn phi_chain_is_dropped_on_lowering() {
    let body = (0..=50)
        .map(|index| Node::store("out", Expr::u32(0), Expr::u32(index)))
        .collect::<Vec<_>>();
    let mut graph = to_graph(&program_with_body(body));
    for index in 1..graph.nodes.len() {
        let previous = graph.nodes[index - 1].id;
        graph.nodes[index].kind = DataflowKind::Phi(vec![previous]);
    }
    graph.edges.clear();
    let lowered = from_graph(graph).unwrap();

    assert_eq!(
        lowered.entry().len(),
        1,
        "Fix: lowering must drop every synthetic Phi node in the chain"
    );
}

/// Node descent comes from `transform::visit::for_each_node`, so a statement
/// that survives lowering inside a nesting variant this file does not
/// enumerate is still counted.
fn count_stores(nodes: &[Node]) -> usize {
    let mut stores = 0;
    for_each_node(nodes, |node| {
        stores += usize::from(matches!(node, Node::Store { .. }));
    });
    stores
}

/// A deep linear graph (one ordering edge per node) must lower without a
/// stack overflow. The cycle check is an explicit-stack DFS; a recursive DFS
/// would recurse once per node and panic on a chain this long
/// (FINDING-FOUNDATION-1: reject/accept topology *without panicking* at scale).
#[test]
fn deep_linear_chain_lowers_without_stack_overflow() {
    const DEPTH: usize = 200_000;
    let body = (0..DEPTH)
        .map(|index| Node::store("out", Expr::u32(0), Expr::u32((index % 7) as u32)))
        .collect::<Vec<_>>();
    let graph = to_graph(&raw_program_with_body(body));
    // to_graph emits a 0->1->2->... ordering chain DEPTH nodes deep.
    assert!(
        graph.nodes.len() >= DEPTH,
        "expected a chain of >= DEPTH nodes"
    );

    let lowered = from_graph(graph).expect("a deep acyclic chain is well-formed and must lower");
    // from_graph wraps the lowered statements in one root Region, so assert the
    // real invariant: every Statement node survives lowering (none dropped).
    assert_eq!(
        count_stores(lowered.entry()),
        DEPTH,
        "every Statement node in the chain must lower to one statement"
    );
}

/// A cycle buried at the end of a very deep chain must still be detected and
/// rejected, and must do so without a stack overflow.
#[test]
fn deep_cycle_is_detected_without_stack_overflow() {
    const DEPTH: usize = 200_000;
    let body = (0..DEPTH)
        .map(|index| Node::store("out", Expr::u32(0), Expr::u32((index % 7) as u32)))
        .collect::<Vec<_>>();
    let mut graph = to_graph(&raw_program_with_body(body));
    let last = (graph.nodes.len() - 1) as u32;
    // Back edge from the deepest node to the root closes a giant cycle.
    graph.edges.push(vyre_foundation::graph_view::DataEdge::new(
        last,
        0,
        vyre_foundation::graph_view::EdgeKind::Ordering,
    ));

    let result = from_graph(graph);
    assert!(
        matches!(result, Err(GraphValidateError::Cycle { .. })),
        "Fix: a cycle in a deep chain must be rejected (not lowered, not a panic), got {result:?}"
    );
}

#[test]
fn commutative_binops_canonicalize_to_the_same_operand_order() {
    for op in safe_to_sort_nonliterals_binops() {
        let lhs = Expr::var("z");
        let rhs = Expr::var("a");
        let forward = canonicalized_store_value(Expr::BinOp {
            op,
            left: Box::new(lhs.clone()),
            right: Box::new(rhs.clone()),
        });
        let reversed = canonicalized_store_value(Expr::BinOp {
            op,
            left: Box::new(rhs),
            right: Box::new(lhs),
        });
        assert_eq!(
            forward, reversed,
            "Fix: canonicalize must sort operands for bitwise/boolean commutative BinOps; failed for {op:?}"
        );
    }
}

#[test]
fn add_mul_preserve_nonliteral_operand_order() {
    // IEEE-754 NaN payload propagation is not commutative for float
    // Add/Mul, so canonicalize must not reorder non-literal operands.
    for op in [BinOp::Add, BinOp::Mul] {
        let lhs = Expr::var("z");
        let rhs = Expr::var("a");
        let forward = canonicalized_store_value(Expr::BinOp {
            op,
            left: Box::new(lhs.clone()),
            right: Box::new(rhs.clone()),
        });
        let reversed = canonicalized_store_value(Expr::BinOp {
            op,
            left: Box::new(rhs),
            right: Box::new(lhs),
        });
        assert_ne!(
            forward, reversed,
            "Fix: canonicalize must NOT sort non-literal operands for {op:?} because IEEE-754 NaN payloads are not commutative"
        );
    }
}

#[test]
fn literals_are_hoisted_right_for_commutative_add() {
    let canonical = canonicalized_store_value(Expr::add(Expr::u32(1), Expr::var("x")));
    match canonical {
        Expr::BinOp {
            op: BinOp::Add,
            left,
            right,
        } => {
            assert!(
                !matches!(&*left, Expr::LitU32(_)),
                "Fix: canonicalize must hoist literal Add operands to the right"
            );
            assert!(
                matches!(&*right, Expr::LitU32(1)),
                "Fix: canonicalize must preserve the literal payload when hoisting it right"
            );
        }
        other => panic!("Fix: canonicalize(Add) must remain a BinOp, got {other:?}"),
    }
}

#[test]
fn non_commutative_binops_preserve_operand_order() {
    for op in operand_ordered_binops() {
        let canonical = canonicalized_store_value(Expr::BinOp {
            op,
            left: Box::new(Expr::var("lhs")),
            right: Box::new(Expr::var("rhs")),
        });
        match canonical {
            Expr::BinOp {
                op: actual,
                left,
                right,
            } => {
                assert_eq!(actual, op);
                assert_eq!(&*left, &Expr::var("lhs"));
                assert_eq!(&*right, &Expr::var("rhs"));
            }
            other => panic!("Fix: canonicalize({op:?}) must remain a BinOp, got {other:?}"),
        }
    }
}

#[test]
fn laws_for_unknown_op_returns_empty_vec() {
    assert!(
        laws_for_op("test::missing::law").is_empty(),
        "Fix: querying an unknown op id must return an empty law set"
    );
}

#[test]
fn has_law_is_idempotent_under_duplicate_registration() {
    let once = has_law("test::duplicate::commutative", |law| {
        matches!(law, AlgebraicLaw::Commutative)
    });
    let twice = has_law("test::duplicate::commutative", |law| {
        matches!(law, AlgebraicLaw::Commutative)
    });
    assert!(
        once,
        "Fix: duplicate registration must still satisfy has_law"
    );
    assert_eq!(
        once, twice,
        "Fix: duplicate registration must not make has_law nondeterministic"
    );
}

#[test]
fn is_commutative_distinguishes_add_from_sub() {
    assert!(
        is_commutative("test::binop::add"),
        "Fix: Add-style op ids registered as commutative must query true"
    );
    assert!(
        !is_commutative("test::binop::sub"),
        "Fix: unregistered Sub-style op ids must not be reported commutative"
    );
}
