//! The class closed here: a rewriting walk that skips a position, and a
//! rewriting walk that deep-clones a subtree it never touched.
//!
//! # What used to stand here
//!
//! Four passes carried their own `match node { .. }` that rebuilt every
//! variant: induction-variable substitution, fusion alpha-renaming, cache-key
//! canonicalization, and the pass engine's encoded-order rewrite. Nothing tied
//! them to each other or to `visit::child_bodies`, so each one was a separate
//! chance to miss a position, and three of the four rebuilt unconditionally: a
//! substitution for a variable that does not occur allocated a full copy of the
//! program and returned it.
//!
//! # The property that replaced it
//!
//! `transform::rewrite_walk::rewrite_node` is the only rewriting enumeration of
//! `Node`, and it is held to the read-only enumeration in `transform::visit`:
//!
//! - COVERAGE. Every expression `walk_exprs` reaches for a node is offered to
//!   the rewrite's `operand` hook, and every body `child_bodies` enumerates is
//!   offered to its `body` hook. The two walks are independent matches over the
//!   same enum, so one that drifts from the other fails here.
//! - COVERAGE IS CHECKED PER VARIANT. The fixtures come from
//!   `vyre_test_support::ir_variants`, which must name every entry of
//!   `NODE_VARIANT_NAMES`, so a new variant fails these suites until somebody
//!   records what a rewrite owes it.
//! - NO CLONE ON THE UNCHANGED PATH. A policy that changes nothing must report
//!   `None` for every fixture, and a real caller
//!   (`Program::canonicalized`) must hand back the SAME body allocation, proved
//!   by pointer identity rather than by equality. An equality assertion cannot
//!   see a clone.
//!
//! The mutation that proves the last one: make the `Node::Region` arm of
//! `rewrite_node` build `Some(Node::Region { body: Arc::new(...) })`
//! unconditionally. Equality still holds everywhere and
//! `no_op_policy_reports_no_change_for_every_variant` plus
//! `canonicalizing_an_already_canonical_program_reuses_the_body_allocation` go
//! red.

use std::sync::Arc;

use vyre_foundation::ir::{Expr, Ident, Node, Program};
use vyre_foundation::transform::rewrite_walk::{self, NodeRewrite};
use vyre_foundation::transform::visit::{child_bodies, node_shape, walk_exprs};
use vyre_foundation::MemoryOrdering;
use vyre_test_support::ir_variants::{
    node_body_slot_samples, node_operand_samples, node_variant_samples, NodeSample,
};

fn program_of(nodes: Vec<Node>) -> Program {
    Program::wrapped(Vec::new(), [1, 1, 1], nodes)
}

/// Every fixture the closure gate knows about: bare variants, plus one with a
/// marker in each body slot and each operand slot.
fn every_fixture() -> Vec<NodeSample> {
    let marker_node = Node::barrier_with_ordering(MemoryOrdering::SeqCst);
    let marker_expr = Expr::var("vyre_fixture_marker_operand");
    let mut all = node_variant_samples();
    all.extend(node_body_slot_samples(&marker_node));
    all.extend(node_operand_samples(&marker_expr));
    all
}

/// A policy that changes nothing and records what it was offered.
#[derive(Default)]
struct Observe {
    operands: Vec<Expr>,
    idents: Vec<Ident>,
    bodies: Vec<Vec<Node>>,
}

impl NodeRewrite for Observe {
    fn operand(&mut self, expr: &Expr) -> Option<Expr> {
        self.operands.push(expr.clone());
        None
    }

    fn ident(&mut self, name: &Ident) -> Option<Ident> {
        self.idents.push(name.clone());
        None
    }

    fn body(&mut self, _parent: &Node, body: &[Node]) -> Option<Vec<Node>> {
        self.bodies.push(body.to_vec());
        rewrite_walk::rewrite_body(body, self)
    }
}

/// An observing policy that does not descend, so it records exactly the bodies
/// of one node rather than of its whole subtree.
#[derive(Default)]
struct ObserveShallow {
    bodies: Vec<Vec<Node>>,
}

impl NodeRewrite for ObserveShallow {
    fn operand(&mut self, _expr: &Expr) -> Option<Expr> {
        None
    }

    fn body(&mut self, _parent: &Node, body: &[Node]) -> Option<Vec<Node>> {
        self.bodies.push(body.to_vec());
        None
    }
}

/// A policy that replaces one expression wherever it is offered.
struct ReplaceOperand {
    from: Expr,
    to: Expr,
}

impl NodeRewrite for ReplaceOperand {
    fn operand(&mut self, expr: &Expr) -> Option<Expr> {
        (*expr == self.from).then(|| self.to.clone())
    }
}

/// A policy that replaces one identifier wherever it is offered.
struct ReplaceIdent {
    from: Ident,
    to: Ident,
}

impl NodeRewrite for ReplaceIdent {
    fn operand(&mut self, _expr: &Expr) -> Option<Expr> {
        None
    }

    fn ident(&mut self, name: &Ident) -> Option<Ident> {
        (*name == self.from).then(|| self.to.clone())
    }
}

fn sorted_debug<T: std::fmt::Debug>(items: &[T]) -> Vec<String> {
    let mut out: Vec<String> = items.iter().map(|item| format!("{item:?}")).collect();
    out.sort();
    out
}

/// Every expression the read-only walk reaches is offered to the rewrite.
///
/// `walk_exprs` and `rewrite_node` are independent exhaustive matches over the
/// same enum. This is what stops them drifting: an operand position added to
/// one and forgotten in the other shows up as a missing expression here, which
/// for a substitution would be a stale variable reference left in the IR.
#[test]
fn rewrite_walk_offers_every_expression_the_read_only_walk_reaches() {
    for sample in every_fixture() {
        let mut observed = Observe::default();
        assert_eq!(
            rewrite_walk::rewrite_node(&sample.node, &mut observed),
            None,
            "{}: an observing policy changes nothing, so the node must not be rebuilt",
            sample.label()
        );

        // The rewrite is offered whole operands; the read-only walk yields
        // every sub-expression. Expand the offered operands the same way.
        let expanded = program_of(
            observed
                .operands
                .iter()
                .map(|expr| Node::let_bind("vyre_fixture_probe", expr.clone()))
                .collect(),
        );
        let mut offered = Vec::new();
        walk_exprs(&expanded, |expr| offered.push(expr.clone()));
        // Drop the probe bindings' own operands, which are the offered roots
        // and are already present.
        let mut reached = Vec::new();
        walk_exprs(&program_of(vec![sample.node.clone()]), |expr| {
            reached.push(expr.clone());
        });

        assert_eq!(
            sorted_debug(&offered),
            sorted_debug(&reached),
            "{}: the rewriting walk and walk_exprs disagree about this variant's \
             expressions",
            sample.label()
        );
    }
}

/// Every body the read-only enumeration exposes is offered to the rewrite.
///
/// `child_bodies` always returns two groups and only `Node::If` fills both, so
/// the comparison is over their contents. The per-slot half of this, that a
/// rewrite reaches inside each individual slot, is
/// [`a_rewritten_operand_reaches_inside_every_body_slot`].
#[test]
fn rewrite_walk_offers_every_body_child_bodies_enumerates() {
    for sample in every_fixture() {
        let mut observed = ObserveShallow::default();
        rewrite_walk::rewrite_node(&sample.node, &mut observed);

        let expected: Vec<&Node> = child_bodies(&sample.node).into_iter().flatten().collect();
        let offered: Vec<&Node> = observed.bodies.iter().flatten().collect();
        assert_eq!(
            offered,
            expected,
            "{}: child_bodies exposes bodies the rewriting walk never offered",
            sample.label()
        );
        assert_eq!(
            node_shape(&sample.node).nests_nodes,
            !observed.bodies.is_empty(),
            "{}: node_shape and the rewriting walk disagree about whether this \
             variant nests bodies",
            sample.label()
        );
    }
}

/// A policy that changes nothing rebuilds nothing, for every variant.
///
/// This is the borrow-preserving contract in its strongest form: it fails on a
/// walk that rebuilds any variant unconditionally, which equality assertions
/// downstream cannot see.
#[test]
fn no_op_policy_reports_no_change_for_every_variant() {
    for sample in every_fixture() {
        let mut observed = Observe::default();
        assert_eq!(
            rewrite_walk::rewrite_node(&sample.node, &mut observed),
            None,
            "{}: nothing was rewritten, so the walk must return the original",
            sample.label()
        );

        let body = vec![sample.node.clone()];
        assert_eq!(
            rewrite_walk::rewrite_body(&body, &mut Observe::default()),
            None,
            "{}: an unchanged body must not be reallocated",
            sample.label()
        );
    }
}

/// A rewritten operand reaches every operand-carrying variant.
#[test]
fn a_rewritten_operand_lands_in_every_operand_slot() {
    let marker = Expr::var("vyre_fixture_marker_operand");
    let replacement = Expr::u32(0xDEAD_BEEF);
    for sample in node_operand_samples(&marker) {
        let mut policy = ReplaceOperand {
            from: marker.clone(),
            to: replacement.clone(),
        };
        let rewritten = rewrite_walk::rewrite_node(&sample.node, &mut policy)
            .unwrap_or_else(|| panic!("{}: the planted operand was never offered", sample.label()));

        let mut found = false;
        walk_exprs(&program_of(vec![rewritten]), |expr| {
            if *expr == replacement {
                found = true;
            }
            assert_ne!(
                *expr,
                marker,
                "{}: the planted operand survived the rewrite",
                sample.label()
            );
        });
        assert!(found, "{}: the replacement is missing", sample.label());
    }
}

/// A rewritten operand reaches a node nested inside every body slot.
#[test]
fn a_rewritten_operand_reaches_inside_every_body_slot() {
    let marker = Expr::var("vyre_fixture_marker_operand");
    let replacement = Expr::u32(0xDEAD_BEEF);
    let planted = Node::let_bind("vyre_fixture_inner", marker.clone());
    for sample in node_body_slot_samples(&planted) {
        let mut policy = ReplaceOperand {
            from: marker.clone(),
            to: replacement.clone(),
        };
        let rewritten =
            rewrite_walk::rewrite_node(&sample.node, &mut policy).unwrap_or_else(|| {
                panic!(
                    "{}: the walk never descended into this slot",
                    sample.label()
                )
            });
        let mut found = false;
        walk_exprs(&program_of(vec![rewritten]), |expr| {
            if *expr == replacement {
                found = true;
            }
        });
        assert!(
            found,
            "{}: a rewrite must reach a node nested in this body slot",
            sample.label()
        );
    }
}

/// Binding and tag identifiers are a rewritable position, and only the value
/// namespace is one.
///
/// Alpha-renaming depends on both halves: a `Let` target it fails to rename
/// desyncs from its uses, and a buffer name it does rename points at a
/// declaration that no longer exists.
#[test]
fn ident_positions_cover_the_value_namespace_and_stop_there() {
    let from = Ident::from("fixture_v");
    let to = Ident::from("vyre_fixture_renamed");
    let renamed: Vec<String> = every_fixture()
        .into_iter()
        .filter_map(|sample| {
            let mut policy = ReplaceIdent {
                from: from.clone(),
                to: to.clone(),
            };
            rewrite_walk::rewrite_node(&sample.node, &mut policy).map(|_| sample.label())
        })
        .collect();
    assert!(
        renamed.iter().any(|label| label.starts_with("Node::Let")),
        "a Let target must be a rewritable identifier: {renamed:?}"
    );
    assert!(
        renamed
            .iter()
            .any(|label| label.starts_with("Node::Assign")),
        "an Assign target must be a rewritable identifier: {renamed:?}"
    );

    let buffer = Ident::from("fixture_buffer");
    for sample in every_fixture() {
        let mut policy = ReplaceIdent {
            from: buffer.clone(),
            to: to.clone(),
        };
        assert_eq!(
            rewrite_walk::rewrite_node(&sample.node, &mut policy),
            None,
            "{}: buffer names are declared in the program's buffer table, not bound \
             by a node, so the node walk must leave them alone",
            sample.label()
        );
    }
}

/// The real caller keeps the allocation when nothing is canonicalized.
///
/// `Program::canonicalized` runs on every security-sensitive cache key, so a
/// walk that rebuilt unconditionally copied the whole program on every lookup.
/// Pointer identity is the only assertion that can see that; the two programs
/// compare equal either way.
#[test]
fn canonicalizing_an_already_canonical_program_reuses_the_body_allocation() {
    let program = program_of(vec![
        Node::let_bind("x", Expr::u32(1)),
        Node::store("out", Expr::u32(0), Expr::var("x")),
    ]);
    let canonical = program.canonicalized();

    let before = region_body(program.entry());
    let after = region_body(canonical.entry());
    assert!(
        Arc::ptr_eq(before, after),
        "an already-canonical body must be the same allocation, not an equal copy"
    );
}

/// Canonicalization still rewrites when there is something to rewrite, so the
/// test above cannot pass by doing nothing.
#[test]
fn canonicalizing_a_swappable_operand_rebuilds_the_body() {
    let program = program_of(vec![Node::let_bind(
        "x",
        Expr::BinOp {
            op: vyre_foundation::ir::BinOp::Add,
            left: Box::new(Expr::u32(1)),
            right: Box::new(Expr::var("y")),
        },
    )]);
    let canonical = program.canonicalized();
    assert!(
        !Arc::ptr_eq(region_body(program.entry()), region_body(canonical.entry())),
        "a literal-first commutative operand must be normalized, which rebuilds the body"
    );
    assert_ne!(program.entry(), canonical.entry());
}

fn region_body(entry: &[Node]) -> &Arc<Vec<Node>> {
    match entry {
        [Node::Region { body, .. }] => body,
        other => panic!("Fix: Program::wrapped must produce one root region, got {other:?}"),
    }
}
