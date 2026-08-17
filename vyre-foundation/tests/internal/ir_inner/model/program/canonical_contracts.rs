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
use crate::ir_inner::model::node::node_op_id;
use crate::ir_inner::model::op_signature::{BinOp, CollectiveOp, CommGroup};
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
                Node::AsyncWait {
                    tag: "stage".into(),
                },
                Node::Resume {
                    tag: "stage".into(),
                },
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

/// The nested statement lists `node` owns, in the order the splice walk must
/// visit them; empty for a statement that owns none.
///
/// This is the run-time half of the splice-walk exhaustiveness contract. The
/// match is total and carries no wildcard arm, so a new `Node` variant does not
/// compile until it declares whether it owns a body. Every check below that has
/// to know which shapes own bodies reads it from here instead of from a list
/// that would silently stop covering a variant the IR gained.
fn nested_bodies(node: &Node) -> Vec<&[Node]> {
    match node {
        Node::Block(body) => vec![body.as_slice()],
        Node::Loop { body, .. } => vec![body.as_slice()],
        Node::Region { body, .. } => vec![body.as_slice()],
        Node::If {
            then, otherwise, ..
        } => vec![then.as_slice(), otherwise.as_slice()],
        Node::Let { .. }
        | Node::Assign { .. }
        | Node::Store { .. }
        | Node::IndirectDispatch { .. }
        | Node::AsyncLoad { .. }
        | Node::AsyncStore { .. }
        | Node::AsyncWait { .. }
        | Node::Trap { .. }
        | Node::Resume { .. }
        | Node::AllReduce { .. }
        | Node::AllGather { .. }
        | Node::ReduceScatter { .. }
        | Node::Broadcast { .. }
        | Node::Return
        | Node::Barrier { .. }
        | Node::TileLoad { .. }
        | Node::TileStore { .. }
        | Node::TileMatmul { .. }
        | Node::TileReduce { .. }
        | Node::TileDecl { .. }
        | Node::Opaque(_) => Vec::new(),
        Node::TileElementwise { body, .. } => vec![body.as_slice()],
    }
}

/// Path of every transparent `Block` still present under `nodes`.
///
/// Descends through [`nested_bodies`], never through the walk under test, so a
/// body position the walk stepped over is still a body position this scan
/// enters.
fn transparent_blocks_remaining(nodes: &[Node], path: &str, found: &mut Vec<String>) {
    for (index, node) in nodes.iter().enumerate() {
        let here = format!("{path}/{index}:{}", node_op_id(node));
        if let Node::Block(children) = node {
            if children
                .iter()
                .all(|child| !matches!(child, Node::Let { .. }))
            {
                found.push(here.clone());
            }
        }
        for body in nested_bodies(node) {
            transparent_blocks_remaining(body, &here, found);
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
                } => Node::loop_for(
                    var.clone(),
                    from.clone(),
                    to.clone(),
                    wrap(body).into_owned(),
                ),
                Node::Region {
                    generator,
                    source_region,
                    body,
                } => Node::Region {
                    generator: generator.clone(),
                    source_region: source_region.clone(),
                    body: Arc::new(wrap(body).into_owned()),
                },
                Node::Block(body) => Node::block(wrap(body).into_owned()),
                // Listed rather than wildcarded, for the same reason the splice
                // walk lists them: a new node that owns a body must be routed
                // above, or the perturbation would never reach inside it and
                // the walk would go untested there.
                Node::Assign { .. }
                | Node::Store { .. }
                | Node::IndirectDispatch { .. }
                | Node::AsyncLoad { .. }
                | Node::AsyncStore { .. }
                | Node::AsyncWait { .. }
                | Node::Trap { .. }
                | Node::Resume { .. }
                | Node::AllReduce { .. }
                | Node::AllGather { .. }
                | Node::ReduceScatter { .. }
                | Node::Broadcast { .. }
                | Node::Return
                | Node::Barrier { .. }
                | Node::TileLoad { .. }
                | Node::TileStore { .. }
                | Node::TileMatmul { .. }
                | Node::TileReduce { .. }
                | Node::TileDecl { .. }
                | Node::Opaque(_) => node.clone(),
                Node::TileElementwise { out, inputs, body } => Node::TileElementwise {
                    out: out.clone(),
                    inputs: inputs.clone(),
                    body: wrap(body).into_owned(),
                },
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

/// A transparent `Block` in any body position must be flattened.
///
/// Why this exists: the splice walk lists every `Node` variant explicitly, so a
/// new body-carrying variant cannot compile until it is routed. What the
/// compiler cannot catch is an existing variant routed to the leaf arm by
/// mistake, because the walk then steps over its body and leaves transparent
/// blocks inside it, under a fingerprint that claims to be canonical. The scan
/// descends through `nested_bodies`, a second exhaustive match over the same
/// enum, so a body the walk skipped is still a body the scan enters and the
/// surviving block fails the assertion.
///
/// What it does not catch: a variant added to both matches as a leaf when it
/// really owns a body. Nothing short of the enum declaration can catch that.
#[test]
fn no_transparent_block_survives_in_any_body_position() {
    for (name, program) in canonical_fixtures() {
        let perturbed = wrap_in_transparent_blocks(&program);
        let mut introduced = Vec::new();
        transparent_blocks_remaining(perturbed.entry(), "", &mut introduced);
        assert!(
            !introduced.is_empty(),
            "fixture `{name}` gained no transparent block under the perturbation, so this case proves nothing. Fix: give the fixture a statement the perturbation wraps."
        );
        let canonical = perturbed.canonicalized();
        let mut surviving = Vec::new();
        transparent_blocks_remaining(canonical.entry(), "", &mut surviving);
        assert!(
            surviving.is_empty(),
            "fixture `{name}` still carries transparent blocks at {surviving:?} after canonicalization. Fix: route the owning node kind through the nested-body arms of splice_transparent_blocks."
        );
    }
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
