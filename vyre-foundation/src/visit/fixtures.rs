//! Shared IR-shape generators for the traversal proptests.
//!
//! The public visitor and the validator are exercised over the same corpus, so
//! the corpus has exactly one owner: this module. `crate::validate` drives it
//! through [`arb_program`].
//!
//! The corpus is the union of what both callers need. In particular
//! [`arb_expr`] emits argument-less `Expr::Call` leaves: the validator needs
//! them to reach `validate_call`, and the traversal proptests are strictly
//! better covered for having them, since `Expr::Call` is a node the walk
//! descends into.

use crate::composition::mark_self_exclusive_region;
use crate::ir::{
    AtomicOp, BinOp, BufferDecl, CollectiveOp, CommGroup, DataType, Expr, Ident, Node,
    NodeExtension, Program, SubgroupReduceOp, UnOp,
};
use crate::ir_inner::model::tile::{Layout, Residency, Tile};
use crate::ir_inner::model::expr::GeneratorRef;
use crate::memory_model::MemoryOrdering;
use proptest::prelude::*;
use std::sync::Arc;

vyre_test_support::test_node_extension! {
    WellFormedNodeExtension,
    kind: "test.node.wellformed",
    identity: "test.node.wellformed",
    fingerprint: 0x11,
}

// An extension whose `debug_identity` is empty, so `V031` is reachable from
// the corpus rather than only from a hand-written case.
vyre_test_support::test_node_extension! {
    AnonymousNodeExtension,
    kind: "test.node.anonymous",
    identity: "",
    fingerprint: 0x22,
}

pub(crate) fn arb_ident() -> BoxedStrategy<String> {
    prop::sample::select(&["x", "y", "idx", "i", "acc"][..])
        .prop_map(str::to_string)
        .boxed()
}

pub(crate) fn arb_buffer_name() -> BoxedStrategy<String> {
    prop::sample::select(&["out", "input", "rw", "counts", "scratch"][..])
        .prop_map(str::to_string)
        .boxed()
}

pub(crate) fn arb_call_op() -> BoxedStrategy<String> {
    prop::sample::select(
        &[
            "test.noop",
            "test.add.u32",
            "test.mul.f32",
            "test.unknown_op",
        ][..],
    )
    .prop_map(str::to_string)
    .boxed()
}

pub(crate) fn arb_expr() -> BoxedStrategy<Expr> {
    let leaf = prop_oneof![
        any::<u32>().prop_map(Expr::LitU32),
        any::<i32>().prop_map(Expr::LitI32),
        any::<bool>().prop_map(Expr::LitBool),
        arb_ident().prop_map(Expr::var),
        arb_buffer_name().prop_map(Expr::buf_len),
        arb_call_op().prop_map(|op| Expr::call(op, vec![])),
    ];

    leaf.prop_recursive(3, 48, 3, |inner| {
        prop_oneof![
            (arb_buffer_name(), inner.clone()).prop_map(|(buffer, index)| Expr::Load {
                buffer: buffer.into(),
                index: Box::new(index),
            }),
            (inner.clone(), inner.clone()).prop_map(|(left, right)| Expr::BinOp {
                op: BinOp::Add,
                left: Box::new(left),
                right: Box::new(right),
            }),
            (inner.clone(), inner.clone()).prop_map(|(left, right)| Expr::BinOp {
                op: BinOp::Sub,
                left: Box::new(left),
                right: Box::new(right),
            }),
            inner.clone().prop_map(|operand| Expr::UnOp {
                op: UnOp::Negate,
                operand: Box::new(operand),
            }),
            (inner.clone(), inner.clone(), inner.clone()).prop_map(
                |(cond, true_val, false_val)| Expr::Select {
                    cond: Box::new(cond),
                    true_val: Box::new(true_val),
                    false_val: Box::new(false_val),
                }
            ),
            inner.clone().prop_map(|value| Expr::Cast {
                target: DataType::U32,
                value: Box::new(value),
            }),
            (
                arb_buffer_name(),
                inner.clone(),
                proptest::option::of(inner.clone()),
                inner.clone(),
            )
                .prop_map(|(buffer, index, expected, value)| Expr::Atomic {
                    op: AtomicOp::Add,
                    buffer: buffer.into(),
                    index: Box::new(index),
                    expected: expected.map(Box::new),
                    value: Box::new(value),
                    ordering: MemoryOrdering::SeqCst,
                }),
        ]
    })
    .boxed()
}

pub(crate) fn arb_async_tag() -> BoxedStrategy<String> {
    // The empty tag is the `V128` case; it belongs in the corpus because
    // both walks must report it at the same end of the transfer.
    prop::sample::select(&["stream0", "stream1", ""][..])
        .prop_map(str::to_string)
        .boxed()
}

pub(crate) fn arb_generator() -> BoxedStrategy<String> {
    prop_oneof![
        prop::sample::select(&["region.a", "region.b"][..]).prop_map(str::to_string),
        prop::sample::select(&["excl.a", "excl.b"][..]).prop_map(mark_self_exclusive_region),
    ]
    .boxed()
}

pub(crate) fn arb_collective_op() -> BoxedStrategy<CollectiveOp> {
    prop::sample::select(&[CollectiveOp::Sum, CollectiveOp::Max][..]).boxed()
}

pub(crate) fn arb_comm_group() -> BoxedStrategy<CommGroup> {
    prop::sample::select(&[CommGroup::WORLD, CommGroup(1)][..]).boxed()
}

pub(crate) fn arb_node() -> BoxedStrategy<Node> {
    arb_node_with_depth(3)
}

/// The IR-shape corpus every walk-equivalence property runs on.
///
/// Covers every variant `Node` declares. `corpus_generates_every_node_variant`
/// is the gate: it samples this strategy, names what came out with
/// `crate::ir::node_variant_name`, and compares that against
/// `crate::ir::NODE_VARIANT_NAMES`, which the AST registry macro emits from
/// the enum body. A variant added to `Node` and not added here turns that
/// gate RED, so the corpus cannot silently stop covering the enum.
pub(crate) fn arb_node_with_depth(depth: u32) -> BoxedStrategy<Node> {
    let leaf = prop::strategy::Union::new(vec![
        (arb_ident(), arb_expr())
            .prop_map(|(name, value)| Node::Let {
                name: name.into(),
                value,
            })
            .boxed(),
        (arb_ident(), arb_expr())
            .prop_map(|(name, value)| Node::Assign {
                name: name.into(),
                value,
            })
            .boxed(),
        (arb_buffer_name(), arb_expr(), arb_expr())
            .prop_map(|(buffer, index, value)| Node::Store {
                buffer: buffer.into(),
                index,
                value,
            })
            .boxed(),
        (arb_buffer_name(), 0u64..=8)
            .prop_map(|(count_buffer, count_offset)| Node::IndirectDispatch {
                count_buffer: count_buffer.into(),
                count_offset,
            })
            .boxed(),
        (
            arb_buffer_name(),
            arb_buffer_name(),
            arb_expr(),
            arb_expr(),
            arb_async_tag(),
        )
            .prop_map(|(source, destination, offset, size, tag)| Node::AsyncLoad {
                source: source.into(),
                destination: destination.into(),
                offset: Box::new(offset),
                size: Box::new(size),
                tag: tag.into(),
            })
            .boxed(),
        (
            arb_buffer_name(),
            arb_buffer_name(),
            arb_expr(),
            arb_expr(),
            arb_async_tag(),
        )
            .prop_map(
                |(source, destination, offset, size, tag)| Node::AsyncStore {
                    source: source.into(),
                    destination: destination.into(),
                    offset: Box::new(offset),
                    size: Box::new(size),
                    tag: tag.into(),
                },
            )
            .boxed(),
        arb_async_tag()
            .prop_map(|tag| Node::AsyncWait { tag: tag.into() })
            .boxed(),
        (arb_expr(), arb_async_tag())
            .prop_map(|(address, tag)| Node::Trap {
                address: Box::new(address),
                tag: tag.into(),
            })
            .boxed(),
        arb_async_tag()
            .prop_map(|tag| Node::Resume { tag: tag.into() })
            .boxed(),
        (arb_buffer_name(), arb_collective_op(), arb_comm_group())
            .prop_map(|(buffer, op, group)| Node::AllReduce {
                buffer: buffer.into(),
                op,
                group,
            })
            .boxed(),
        (arb_buffer_name(), arb_buffer_name(), arb_comm_group())
            .prop_map(|(input, output, group)| Node::AllGather {
                input: input.into(),
                output: output.into(),
                group,
            })
            .boxed(),
        (
            arb_buffer_name(),
            arb_buffer_name(),
            arb_collective_op(),
            arb_comm_group(),
        )
            .prop_map(|(input, output, op, group)| Node::ReduceScatter {
                input: input.into(),
                output: output.into(),
                op,
                group,
            })
            .boxed(),
        (arb_buffer_name(), 0u32..=2, arb_comm_group())
            .prop_map(|(buffer, root, group)| Node::Broadcast {
                buffer: buffer.into(),
                root,
                group,
            })
            .boxed(),
        Just(Node::Return).boxed(),
        Just(Node::barrier()).boxed(),
        prop::bool::ANY
            .prop_map(|well_formed| {
                let extension: Arc<dyn NodeExtension> = if well_formed {
                    Arc::new(WellFormedNodeExtension)
                } else {
                    Arc::new(AnonymousNodeExtension)
                };
                Node::Opaque(extension)
            })
            .boxed(),
        // Tile leaf nodes (no body).
        (arb_ident(), arb_buffer_name(), arb_expr())
            .prop_map(|(tile, buffer, origin)| Node::tile_load(
                tile.clone(),
                Tile::new(DataType::F32, vec![16, 16], Layout::RowMajor, Residency::Register),
                buffer,
                vec![origin],
                Layout::RowMajor,
            ))
            .boxed(),
        (arb_buffer_name(), arb_expr(), arb_ident())
            .prop_map(|(buffer, origin, tile)| Node::tile_store(buffer, vec![origin], tile))
            .boxed(),
        (arb_ident(), arb_ident(), arb_ident())
            .prop_map(|(acc, a, b)| Node::tile_matmul(acc, a, b))
            .boxed(),
        (arb_ident(), arb_ident())
            .prop_map(|(out, tile)| Node::tile_reduce(out, tile, SubgroupReduceOp::Add, 0))
            .boxed(),
        arb_ident()
            .prop_map(|name| Node::tile_decl(
                name,
                Tile::new(DataType::F32, vec![16, 16], Layout::RowMajor, Residency::Register),
            ))
            .boxed(),
    ]);

    if depth == 0 {
        return leaf.boxed();
    }

    leaf.prop_recursive(2, 48, 2, move |inner| {
        prop_oneof![
            (
                arb_expr(),
                prop::collection::vec(inner.clone(), 0..=3),
                prop::collection::vec(inner.clone(), 0..=3),
            )
                .prop_map(|(cond, then, otherwise)| Node::If {
                    cond,
                    then,
                    otherwise,
                }),
            (
                arb_ident(),
                arb_expr(),
                arb_expr(),
                prop::collection::vec(inner.clone(), 0..=3),
            )
                .prop_map(|(var, from, to, body)| Node::Loop {
                    var: var.into(),
                    from,
                    to,
                    body,
                }),
            prop::collection::vec(inner.clone(), 0..=3).prop_map(Node::Block),
            (
                arb_generator(),
                proptest::option::of(arb_generator()),
                prop::collection::vec(inner.clone(), 0..=3),
            )
                .prop_map(|(generator, source_region, body)| Node::Region {
                    generator: Ident::from(generator),
                    source_region: source_region.map(|name| GeneratorRef { name }),
                    body: Arc::new(body),
                }),
            (
                arb_ident(),
                prop::collection::vec(arb_ident().prop_map(Ident::from), 1..=2),
                prop::collection::vec(inner, 0..=3),
            )
                .prop_map(|(out, inputs, body)| Node::TileElementwise {
                    out: out.into(),
                    inputs,
                    body,
                }),
        ]
    })
    .boxed()
}

pub(crate) fn arb_program() -> BoxedStrategy<Program> {
    prop::collection::vec(arb_node(), 0..=8)
        .prop_map(|entry| {
            Program::wrapped(
                vec![
                    BufferDecl::output("out", 0, DataType::U32)
                        .with_count(8)
                        .with_output_byte_range(0..16),
                    BufferDecl::read("input", 1, DataType::U32).with_count(8),
                    BufferDecl::read_write("rw", 2, DataType::U32).with_count(8),
                    BufferDecl::read("counts", 3, DataType::U32).with_count(8),
                    BufferDecl::workgroup("scratch", 4, DataType::U32),
                ],
                [1, 1, 1],
                entry,
            )
        })
        .boxed()
}

/// `Node` variants the corpus deliberately does not generate, each with the
/// reason.
///
/// Empty: every declared variant is generated. An entry here is a recorded
/// decision, not a shortcut, and `corpus_generates_every_node_variant`
/// accepts only variants named here as absent.
pub(crate) const CORPUS_EXCLUDED_NODE_VARIANTS: &[(&str, &str)] = &[];
