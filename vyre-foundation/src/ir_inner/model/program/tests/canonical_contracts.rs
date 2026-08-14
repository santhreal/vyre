//! Canonicalization reports what it changed instead of rebuilding the tree.
//!
//! The optimizer fingerprints the program after every pass, and by then the
//! program is normally canonical already, so `Program::canonical_form` returns
//! a borrow of the original. These contracts pin both directions of that
//! decision, because either one being wrong is a miscompile of the cache key:
//!
//! * borrowing when a rewrite was owed would publish non-canonical bytes under
//!   the canonical fingerprint, so two programs that differ only in authoring
//!   order would get different keys;
//! * rebuilding when nothing was owed is only slow, and is pinned so the fast
//!   path cannot silently regress into an unconditional deep clone.
//!
//! The byte-level check is a differential one: an independently-written
//! perturbation (swap the operands of every commutative literal/non-literal
//! pair, wrap every body in a transparent `Block`, reverse the buffer table)
//! must land on the same canonical bytes as the original program. That is the
//! choke point every node and expression shape passes through.

use std::borrow::Cow;
use std::sync::Arc;

use crate::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use crate::ir_inner::model::spec_types::{BinOp, CollectiveOp, CommGroup};
use crate::memory_model::MemoryOrdering;
use crate::optimizer::rewrite::{rewrite_node_slices, rewrite_nodes_cow};

/// Declared in canonical key order, so a fixture is canonical as authored and
/// the reversed-buffer perturbation is the only thing that forces a re-sort.
fn buffers() -> Vec<BufferDecl> {
    vec![
        BufferDecl::storage("in", 0, BufferAccess::ReadOnly, DataType::U32).with_count(8),
        BufferDecl::output("out", 2, DataType::U32).with_count(8),
        BufferDecl::storage("scratch", 1, BufferAccess::ReadWrite, DataType::U32).with_count(8),
    ]
}

fn program_with(body: Vec<Node>) -> Program {
    Program::wrapped(buffers(), [1, 1, 1], body)
}

/// One fixture per canonicalization-relevant node shape. Every entry is
/// already canonical: commutative literals sit on the right and no
/// transparent `Block` wraps a body.
fn canonical_fixtures() -> Vec<(&'static str, Program)> {
    vec![
        (
            "let-assign-store",
            program_with(vec![
                Node::let_bind("value", Expr::add(Expr::var("x"), Expr::u32(3))),
                Node::assign("value", Expr::mul(Expr::var("value"), Expr::u32(5))),
                Node::store("out", Expr::u32(0), Expr::var("value")),
            ]),
        ),
        (
            "if-else",
            program_with(vec![Node::if_then_else(
                Expr::lt(Expr::var("x"), Expr::u32(4)),
                vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
                vec![Node::store("out", Expr::u32(1), Expr::u32(2))],
            )]),
        ),
        (
            "loop",
            program_with(vec![Node::loop_for(
                "i",
                Expr::u32(0),
                Expr::u32(4),
                vec![Node::store(
                    "out",
                    Expr::var("i"),
                    Expr::add(Expr::var("i"), Expr::u32(7)),
                )],
            )]),
        ),
        (
            "block-with-binding",
            program_with(vec![Node::block(vec![
                Node::let_bind("kept", Expr::add(Expr::var("x"), Expr::u32(1))),
                Node::store("out", Expr::u32(0), Expr::var("kept")),
            ])]),
        ),
        (
            "async-copies",
            program_with(vec![
                Node::async_load_gpu_driven(
                    "in",
                    "scratch",
                    Expr::add(Expr::var("x"), Expr::u32(2)),
                    Expr::u32(16),
                    "stage",
                ),
                Node::async_store(
                    "scratch",
                    "out",
                    Expr::add(Expr::var("x"), Expr::u32(4)),
                    Expr::u32(16),
                    "stage",
                ),
                Node::AsyncWait { tag: "stage".into() },
                Node::Resume { tag: "stage".into() },
            ]),
        ),
        (
            "trap-barrier-return",
            program_with(vec![
                Node::Trap {
                    address: Box::new(Expr::add(Expr::var("x"), Expr::u32(9))),
                    tag: "oob".into(),
                },
                Node::barrier_with_ordering(MemoryOrdering::AcqRel),
                Node::Return,
            ]),
        ),
        (
            "indirect-dispatch",
            program_with(vec![Node::IndirectDispatch {
                count_buffer: "in".into(),
                count_offset: 0,
            }]),
        ),
        (
            "collectives",
            program_with(vec![
                Node::AllReduce {
                    buffer: "scratch".into(),
                    op: CollectiveOp::Sum,
                    group: CommGroup::WORLD,
                },
                Node::Broadcast {
                    buffer: "scratch".into(),
                    root: 0,
                    group: CommGroup::WORLD,
                },
            ]),
        ),
    ]
}

/// Node kinds the fixtures must keep exercising: every shape that carries an
/// expression or a nested body (so canonicalization has something to rebuild),
/// plus one leaf shape per inert family. Named by the wire-format op id, so a
/// renamed node kind fails here instead of quietly dropping coverage.
const REQUIRED_NODE_KINDS: [&str; 15] = [
    "vyre.node.let",
    "vyre.node.assign",
    "vyre.node.store",
    "vyre.node.if",
    "vyre.node.loop",
    "vyre.node.block",
    "vyre.node.region",
    "vyre.node.async_load",
    "vyre.node.async_store",
    "vyre.node.async_wait",
    "vyre.node.trap",
    "vyre.node.resume",
    "vyre.node.barrier",
    "vyre.node.return",
    "vyre.node.indirect_dispatch",
];

fn node_kinds_in(nodes: &[Node], found: &mut Vec<&'static str>) {
    for node in nodes {
        let kind = crate::ir_inner::model::node::node_op_id(node);
        if !found.contains(&kind) {
            found.push(kind);
        }
        match node {
            Node::If {
                then, otherwise, ..
            } => {
                node_kinds_in(then, found);
                node_kinds_in(otherwise, found);
            }
            Node::Loop { body, .. } | Node::Block(body) => node_kinds_in(body, found),
            Node::Region { body, .. } => node_kinds_in(body, found),
            _ => {}
        }
    }
}

/// Swap the operands of every commutative `BinOp` that pairs a literal with a
/// non-literal. Written against the IR, not against the canonicalizer: the
/// canonical form of the result must equal the canonical form of the input.
fn swap_literal_operands(program: &Program) -> Program {
    let entry = rewrite_nodes_cow(program.entry(), &mut |candidate| {
        let Expr::BinOp { op, left, right } = candidate else {
            return None;
        };
        let commutative = matches!(
            op,
            BinOp::Add
                | BinOp::WrappingAdd
                | BinOp::SaturatingAdd
                | BinOp::Mul
                | BinOp::SaturatingMul
                | BinOp::BitAnd
                | BinOp::BitOr
                | BinOp::BitXor
                | BinOp::Eq
                | BinOp::Ne
                | BinOp::And
                | BinOp::Or
                | BinOp::Min
                | BinOp::Max
                | BinOp::AbsDiff
        );
        let literal_pair = matches!(
            (is_literal(left), is_literal(right)),
            (true, false) | (false, true)
        );
        if !commutative || !literal_pair {
            return None;
        }
        Some(Expr::BinOp {
            op: *op,
            left: right.clone(),
            right: left.clone(),
        })
    });
    program.with_rewritten_entry(entry.into_owned())
}

fn is_literal(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::LitU32(_) | Expr::LitI32(_) | Expr::LitF32(_) | Expr::LitBool(_)
    )
}

/// Wrap every statement that is not a binding in a `Block`. Such a block owns
/// no local name, so it is semantically transparent and canonicalization must
/// flatten all of them back out. A `Block` around a `Let` would change scoping,
/// so those stay unwrapped.
fn wrap_in_transparent_blocks(program: &Program) -> Program {
    fn wrap(nodes: &[Node]) -> Cow<'_, [Node]> {
        rewrite_node_slices(nodes, |node| {
            let inner = match node {
                Node::Let { .. } => return Cow::Borrowed(std::slice::from_ref(node)),
                Node::If {
                    cond,
                    then,
                    otherwise,
                } => Node::if_then_else(
                    cond.clone(),
                    wrap(then).into_owned(),
                    wrap(otherwise).into_owned(),
                ),
                Node::Loop {
                    var,
                    from,
                    to,
                    body,
                } => Node::loop_for(var.clone(), from.clone(), to.clone(), wrap(body).into_owned()),
                Node::Region {
                    generator,
                    source_region,
                    body,
                } => Node::Region {
                    generator: generator.clone(),
                    source_region: source_region.clone(),
                    body: Arc::new(wrap(body).into_owned()),
                },
                other => other.clone(),
            };
            Cow::Owned(vec![Node::block(vec![inner])])
        })
    }
    program.with_rewritten_entry(wrap(program.entry()).into_owned())
}

fn reverse_buffers(program: &Program) -> Program {
    let mut buffers = program.buffers().to_vec();
    buffers.reverse();
    program.with_rewritten_buffers(buffers)
}

#[test]
fn fixtures_exercise_every_required_node_kind() {
    let mut found = Vec::new();
    for (_, program) in canonical_fixtures() {
        node_kinds_in(program.entry(), &mut found);
    }
    let missing: Vec<&str> = REQUIRED_NODE_KINDS
        .into_iter()
        .filter(|kind| !found.contains(kind))
        .collect();
    assert!(
        missing.is_empty(),
        "canonicalization fixtures no longer exercise {missing:?}. Fix: add a fixture carrying those node kinds so the borrow/rebuild decision stays covered for them."
    );
}

#[test]
fn already_canonical_programs_are_reused_not_rebuilt() {
    for (name, program) in canonical_fixtures() {
        let canonical = program.canonicalized();
        assert!(
            Arc::ptr_eq(&program.entry, &canonical.entry),
            "fixture `{name}` is already canonical, so canonicalization must reuse its entry body instead of rebuilding it"
        );
        assert!(
            Arc::ptr_eq(&program.buffers, &canonical.buffers),
            "fixture `{name}` declares buffers in canonical order, so canonicalization must reuse the buffer table"
        );
    }
}

#[test]
fn perturbed_programs_are_rebuilt_into_the_same_canonical_bytes() {
    for (name, program) in canonical_fixtures() {
        let expected = program
            .canonical_wire_bytes()
            .expect("fixture must encode to canonical wire bytes");
        for (perturbation, perturbed) in [
            ("swapped-literal-operands", swap_literal_operands(&program)),
            ("transparent-blocks", wrap_in_transparent_blocks(&program)),
            ("reversed-buffers", reverse_buffers(&program)),
        ] {
            let actual = perturbed
                .canonical_wire_bytes()
                .expect("perturbed program must encode to canonical wire bytes");
            assert_eq!(
                actual, expected,
                "fixture `{name}` under `{perturbation}` must canonicalize back to the same wire bytes"
            );
        }
    }
}

#[test]
fn canonicalization_reaches_a_fixed_point_in_one_application() {
    for (name, program) in canonical_fixtures() {
        for (perturbation, perturbed) in [
            ("swapped-literal-operands", swap_literal_operands(&program)),
            ("transparent-blocks", wrap_in_transparent_blocks(&program)),
            ("reversed-buffers", reverse_buffers(&program)),
        ] {
            let once = perturbed.canonicalized();
            let twice = once.canonicalized();
            assert!(
                Arc::ptr_eq(&once.entry, &twice.entry),
                "fixture `{name}` under `{perturbation}` must be canonical after one application, so the second must reuse the entry body"
            );
            assert!(
                Arc::ptr_eq(&once.buffers, &twice.buffers),
                "fixture `{name}` under `{perturbation}` must have a canonical buffer table after one application"
            );
        }
    }
}

#[test]
fn transparent_blocks_are_flattened_but_binding_blocks_survive() {
    let transparent = program_with(vec![Node::block(vec![Node::store(
        "out",
        Expr::u32(0),
        Expr::u32(1),
    )])]);
    let canonical = transparent.canonicalized();
    assert!(
        !Arc::ptr_eq(&transparent.entry, &canonical.entry),
        "a transparent Block must be flattened, which is a rebuild"
    );
    let owning = program_with(vec![Node::block(vec![
        Node::let_bind("kept", Expr::u32(1)),
        Node::store("out", Expr::u32(0), Expr::var("kept")),
    ])]);
    assert!(
        Arc::ptr_eq(&owning.entry, &owning.canonicalized().entry),
        "a Block that owns a binding is not transparent, so canonicalization must leave it alone"
    );
}
