//! Cross-scope CSE hoists a repeated top-level expression after semantic
//! execution of the two CSE analysis graphs.

use vyre_driver_reference::ReferenceSemanticExecutor;
use vyre_foundation::ir::{BinOp, BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_megakernel::{Digest, SearchBudget, SemanticExecutionPolicy};
use vyre_pass_engine::optimizer::cse_via_encoded::{apply_cross_scope_cse, gpu_cse_canonicals};

/// `src[0] + src[1]`, the repeated operand the hoist is supposed to find.
fn repeated() -> Expr {
    Expr::BinOp {
        op: BinOp::Add,
        left: Box::new(Expr::load("src", Expr::u32(0))),
        right: Box::new(Expr::load("src", Expr::u32(1))),
    }
}

fn program(entry: Vec<Node>) -> Program {
    Program::wrapped(
        vec![
            BufferDecl::storage("src", 0, BufferAccess::ReadOnly, DataType::U32).with_count(4),
            BufferDecl::storage("out", 1, BufferAccess::ReadWrite, DataType::U32).with_count(4),
        ],
        [1, 1, 1],
        entry,
    )
}

/// The entry scope, descending through the wrapping `Region`.
fn entry_scope(program: &Program) -> Vec<Node> {
    match program.entry() {
        [Node::Region { body, .. }] => body.as_ref().clone(),
        entry => entry.to_vec(),
    }
}

fn policy() -> SemanticExecutionPolicy {
    vyre_test_support::semantic_requests::unknown_policy(
        Digest([0; 32]),
        SearchBudget::new(8, 64, 0, 0, 1_000),
        1_000_000,
    )
}

fn hoisted(program: &Program) -> Program {
    let (arena, canonical) = gpu_cse_canonicals(program, &ReferenceSemanticExecutor, &policy())
        .expect("the CSE kernels execute through the reference semantic executor");
    apply_cross_scope_cse(program, &arena, &canonical)
}

#[test]
fn a_repeated_top_level_expr_becomes_one_binding_and_two_var_reads() {
    let after = hoisted(&program(vec![
        Node::store("out", Expr::u32(0), repeated()),
        Node::store("out", Expr::u32(1), repeated()),
    ]));

    let scope = entry_scope(&after);
    let bindings: Vec<(String, Expr)> = scope
        .iter()
        .filter_map(|node| match node {
            Node::Let { name, value } => Some((name.to_string(), value.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(
        bindings.len(),
        1,
        "cross-scope CSE hoisted {} bindings for one repeated Expr, so it stopped firing: {scope:?}",
        bindings.len()
    );
    let (name, value) = &bindings[0];
    assert_eq!(
        value,
        &repeated(),
        "the hoisted binding does not hold the repeated Expr"
    );

    let stored: Vec<Expr> = scope
        .iter()
        .filter_map(|node| match node {
            Node::Store { value, .. } => Some(value.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        stored,
        vec![Expr::var(name.as_str()), Expr::var(name.as_str())],
        "both stores should read the hoisted binding"
    );
}

#[test]
fn a_single_occurrence_is_left_where_it_is() {
    let after = hoisted(&program(vec![
        Node::store("out", Expr::u32(0), repeated()),
        Node::store("out", Expr::u32(1), Expr::u32(9)),
    ]));

    let scope = entry_scope(&after);
    assert!(
        !scope.iter().any(|node| matches!(node, Node::Let { .. })),
        "cross-scope CSE hoisted an Expr that occurs once, paying a Var indirection for nothing"
    );
}
