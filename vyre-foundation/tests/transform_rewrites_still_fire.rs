//! Every host-side IR rewrite in `vyre_foundation::transform` still fires.
//!
//! WHY: these rewrites are driven only from the pass-engine resident pipeline,
//! and the end-to-end suites that exercise that pipeline need a live CUDA or
//! WGPU backend. A rewrite that silently stopped rewriting anything therefore
//! left every runnable gate green. Each case below builds the smallest Program
//! the rewrite is supposed to change, and asserts the specific change, so a
//! rewrite that turns into an identity function fails here.
//!
//! The case table is checked against `transform::HOST_REWRITES` at run time: a
//! rewrite the resident pipeline runs with no case here fails
//! `every_rewrite_the_resident_pipeline_runs_has_a_firing_case`, rather than
//! shipping uncovered.
//!
//! Not covered: rewrite quality. These tests pin that the rewrite fires and
//! that it declines the adversarial case next to it, not that its output is
//! optimal.

use std::collections::BTreeSet;

use vyre_foundation::ir::{BinOp, BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::transform::{const_prop, dead_branch, licm, HOST_REWRITES};
use vyre_foundation::visit::child_bodies;

/// A program whose entry is exactly `entry`, with the given buffers.
fn program(buffers: Vec<BufferDecl>, entry: Vec<Node>) -> Program {
    Program::wrapped(buffers, [1, 1, 1], entry)
}

/// Every node reachable from `entry`, in walk order, including `entry` itself.
fn flatten(entry: &[Node]) -> Vec<Node> {
    let mut out = Vec::new();
    let mut stack: Vec<&Node> = entry.iter().rev().collect();
    while let Some(node) = stack.pop() {
        out.push(node.clone());
        for group in child_bodies(node) {
            stack.extend(group.iter().rev());
        }
    }
    out
}

/// The nodes of the entry scope, descending through the wrapping `Region` that
/// `Program::wrapped` adds.
fn entry_scope(program: &Program) -> Vec<Node> {
    match program.entry() {
        [Node::Region { body, .. }] => body.as_ref().clone(),
        entry => entry.to_vec(),
    }
}

/// Names of the `Node::Let` bindings in `nodes`, in order.
fn let_names(nodes: &[Node]) -> Vec<String> {
    nodes
        .iter()
        .filter_map(|node| match node {
            Node::Let { name, .. } => Some(name.to_string()),
            _ => None,
        })
        .collect()
}

#[test]
fn const_prop_substitutes_a_let_bound_literal_into_its_use() {
    let before = program(
        vec![BufferDecl::storage(
            "out",
            0,
            BufferAccess::ReadWrite,
            DataType::U32,
        )],
        vec![
            Node::let_bind("k", Expr::u32(7)),
            Node::store("out", Expr::u32(0), Expr::Var("k".into())),
        ],
    );

    let after = const_prop::apply_const_prop(&before);

    let stored = flatten(&entry_scope(&after))
        .into_iter()
        .find_map(|node| match node {
            Node::Store { value, .. } => Some(value),
            _ => None,
        })
        .expect("the store survives const propagation");
    assert_eq!(
        stored,
        Expr::u32(7),
        "const propagation left Var(k) in the store, so it stopped firing"
    );
}

#[test]
fn dead_branch_collapses_a_literal_condition_to_the_surviving_branch() {
    let before = program(
        vec![BufferDecl::storage(
            "out",
            0,
            BufferAccess::ReadWrite,
            DataType::U32,
        )],
        vec![Node::if_then_else(
            Expr::LitBool(true),
            vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
            vec![Node::store("out", Expr::u32(0), Expr::u32(2))],
        )],
    );

    let after = dead_branch::apply_dead_branch(&before);

    let nodes = flatten(&entry_scope(&after));
    assert!(
        !nodes.iter().any(|node| matches!(node, Node::If { .. })),
        "dead-branch elimination left the constant If in place, so it stopped firing"
    );
    let stored: Vec<Expr> = nodes
        .into_iter()
        .filter_map(|node| match node {
            Node::Store { value, .. } => Some(value),
            _ => None,
        })
        .collect();
    assert_eq!(
        stored,
        vec![Expr::u32(1)],
        "the wrong branch survived the collapse"
    );
}

#[test]
fn dead_branch_keeps_an_if_whose_condition_is_not_a_literal() {
    let before = program(
        vec![
            BufferDecl::storage("flag", 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage("out", 1, BufferAccess::ReadWrite, DataType::U32),
        ],
        vec![Node::if_then_else(
            Expr::load("flag", Expr::u32(0)),
            vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
            Vec::new(),
        )],
    );

    let after = dead_branch::apply_dead_branch(&before);

    assert!(
        flatten(&entry_scope(&after))
            .iter()
            .any(|node| matches!(node, Node::If { .. })),
        "dead-branch elimination dropped an If whose condition is a runtime load"
    );
}

#[test]
fn licm_hoists_an_invariant_let_out_of_the_loop_body() {
    let before = program(
        vec![BufferDecl::storage(
            "out",
            0,
            BufferAccess::ReadWrite,
            DataType::U32,
        )],
        vec![Node::loop_for(
            "i",
            Expr::u32(0),
            Expr::u32(4),
            vec![
                Node::let_bind(
                    "invariant",
                    Expr::BinOp {
                        op: BinOp::Add,
                        left: Box::new(Expr::u32(2)),
                        right: Box::new(Expr::u32(3)),
                    },
                ),
                Node::store("out", Expr::Var("i".into()), Expr::Var("invariant".into())),
            ],
        )],
    );

    let after = licm::apply_licm(&before);

    let scope = entry_scope(&after);
    assert_eq!(
        let_names(&scope),
        vec!["invariant".to_string()],
        "LICM did not hoist the invariant let into the loop's parent scope"
    );
    let body = scope
        .iter()
        .find_map(|node| match node {
            Node::Loop { body, .. } => Some(body.to_vec()),
            _ => None,
        })
        .expect("the loop survives LICM");
    assert!(
        let_names(&body).is_empty(),
        "the hoisted let is still bound inside the loop body as well"
    );
}

#[test]
fn licm_hoists_a_load_from_a_read_only_buffer() {
    let before = program(
        vec![
            BufferDecl::storage("src", 0, BufferAccess::ReadOnly, DataType::U32),
            BufferDecl::storage("out", 1, BufferAccess::ReadWrite, DataType::U32),
        ],
        vec![Node::loop_for(
            "i",
            Expr::u32(0),
            Expr::u32(4),
            vec![
                Node::let_bind("base", Expr::load("src", Expr::u32(0))),
                Node::store("out", Expr::Var("i".into()), Expr::Var("base".into())),
            ],
        )],
    );

    let after = licm::apply_licm(&before);

    assert_eq!(
        let_names(&entry_scope(&after)),
        vec!["base".to_string()],
        "LICM stopped hoisting a Load from a ReadOnly buffer, which is the case that made it worth running"
    );
}

#[test]
fn licm_declines_a_load_from_a_writable_buffer() {
    let before = program(
        vec![
            BufferDecl::storage("scratch", 0, BufferAccess::ReadWrite, DataType::U32),
            BufferDecl::storage("out", 1, BufferAccess::ReadWrite, DataType::U32),
        ],
        vec![Node::loop_for(
            "i",
            Expr::u32(0),
            Expr::u32(4),
            vec![
                Node::let_bind("base", Expr::load("scratch", Expr::u32(0))),
                Node::store("out", Expr::Var("i".into()), Expr::Var("base".into())),
            ],
        )],
    );

    let after = licm::apply_licm(&before);

    assert!(
        let_names(&entry_scope(&after)).is_empty(),
        "LICM hoisted a Load from a buffer the program may write, which reorders it past the Store"
    );
}

/// Rewrites with a firing case above, keyed by the name the registry gives them.
const CASES: &[&str] = &["const_prop", "dead_branch", "licm"];

#[test]
fn every_rewrite_the_resident_pipeline_runs_has_a_firing_case() {
    let registered: BTreeSet<&str> = HOST_REWRITES.iter().map(|rewrite| rewrite.name).collect();
    assert!(
        !registered.is_empty(),
        "the rewrite registry is empty, so the resident pipeline applies nothing"
    );
    assert_eq!(
        registered,
        CASES.iter().copied().collect::<BTreeSet<&str>>(),
        "the rewrites vyre_foundation::transform::HOST_REWRITES declares no longer match the cases \
         in this file; add a firing case for each new rewrite and delete the case for each removed one"
    );
}
