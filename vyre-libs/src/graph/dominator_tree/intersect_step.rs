//! One Cooper-Harvey-Kennedy relaxation sweep over the predecessor lists.
//!
//! For every node but the entry, the sweep intersects the reachable
//! predecessors on the current idom forest and writes the result back when it
//! moved. Intersection is the parent-pointer LCA: walk the deeper of the two
//! sides up one edge until both sides meet.
//!
//! The sweep is its own query. It reads the predecessor CSR, the forest, and
//! the depths that forest implies, and it reports whether anything moved. The
//! fixpoint that composes it owns only the repetition and the convergence test.
//!
//! # Wire shape
//!
//! ```text
//! pred_offsets : u32[node_count + 1]   // predecessor CSR
//! pred_targets : u32[pred_edge_count]  // predecessor CSR
//! idom         : u32[node_count]       // parent pointer, updated in place
//! depth        : u32[node_count]       // depth of the forest on entry
//! changed      : u32[1]                // 1 when the sweep moved a parent
//! ```
//!
//! `depth` is read, never written: it describes the forest the sweep started
//! from. Recomputing it is `dominator_tree_depth`'s query, and the fixpoint
//! runs that one first.

use vyre_foundation::composition::{wrap_anonymous_region, wrap_child_region};
use vyre_foundation::ir::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use super::program::IDOM_NONE;

/// Canonical op id for one predecessor-intersection sweep.
pub const OP_ID: &str = "vyre-libs::graph::dominator_tree_intersect_step";

/// Stable op id for dominator tree LCA intersection.
pub const DOMINATOR_TREE_LCA_OP_ID: &str = "vyre-libs::graph::dominator_tree_lca";

/// Buffer name the standalone operation reports its movement through.
const CHANGED_BUFFER: &str = "dt_changed";

/// Binding the standalone operation carries `changed` on.
const CHANGED_BINDING: u32 = 4;

/// Build one relaxation sweep over `node_count` nodes.
///
/// `changed` is a name the caller binds, not a buffer: the sweep assigns 1 into
/// it when it moves a parent and leaves it alone otherwise, so a caller can
/// accumulate movement across several sweeps in one binding.
#[must_use]
pub fn dominator_tree_intersect_step_body(
    node_count: u32,
    idom: &str,
    depth: &str,
    changed: &str,
) -> Vec<Node> {
    vec![Node::loop_for(
        "v",
        Expr::u32(0),
        Expr::u32(node_count),
        vec![Node::if_then(
            Expr::ne(Expr::var("v"), Expr::u32(0)),
            vec![
                Node::let_bind("new_idom", Expr::u32(IDOM_NONE)),
                Node::let_bind("p_start", Expr::load("pred_offsets", Expr::var("v"))),
                Node::let_bind(
                    "p_end",
                    Expr::load("pred_offsets", Expr::add(Expr::var("v"), Expr::u32(1))),
                ),
                Node::loop_for(
                    "p_idx",
                    Expr::var("p_start"),
                    Expr::var("p_end"),
                    vec![
                        Node::let_bind("p", Expr::load("pred_targets", Expr::var("p_idx"))),
                        Node::if_then(
                            Expr::ne(Expr::load(idom, Expr::var("p")), Expr::u32(IDOM_NONE)),
                            vec![Node::if_then_else(
                                Expr::eq(Expr::var("new_idom"), Expr::u32(IDOM_NONE)),
                                vec![Node::assign("new_idom", Expr::var("p"))],
                                vec![
                                    Node::let_bind("a", Expr::var("new_idom")),
                                    wrap_child_region(
                                        DOMINATOR_TREE_LCA_OP_ID,
                                        Ident::from(OP_ID),
                                        dominator_tree_lca_body(
                                            node_count,
                                            idom,
                                            depth,
                                            Expr::var("p"),
                                        ),
                                    ),
                                    Node::assign("new_idom", Expr::var("a")),
                                ],
                            )],
                        ),
                    ],
                ),
                Node::if_then(
                    Expr::and(
                        Expr::ne(Expr::var("new_idom"), Expr::u32(IDOM_NONE)),
                        Expr::ne(Expr::var("new_idom"), Expr::load(idom, Expr::var("v"))),
                    ),
                    vec![
                        Node::store(idom, Expr::var("v"), Expr::var("new_idom")),
                        Node::assign(changed, Expr::u32(1)),
                    ],
                ),
            ],
        )],
    )]
}

/// The relaxation sweep as a child region of `parent_op_id`.
///
/// The body is the one in [`dominator_tree_intersect_step_body`], so the region
/// names an operation that reads exactly the buffers the body touches.
#[must_use]
pub fn dominator_tree_intersect_step_child(
    parent_op_id: &str,
    node_count: u32,
    idom: &str,
    depth: &str,
    changed: &str,
) -> Node {
    wrap_child_region(
        OP_ID,
        Ident::from(parent_op_id),
        dominator_tree_intersect_step_body(node_count, idom, depth, changed),
    )
}

/// Body of the dominator tree LCA intersection.
#[must_use]
pub fn dominator_tree_lca_body(
    node_count: u32,
    idom: &str,
    depth: &str,
    b_init: Expr,
) -> Vec<Node> {
    vec![
        Node::let_bind("b", b_init),
        Node::loop_for(
            "lca_step",
            Expr::u32(0),
            Expr::u32(node_count),
            vec![Node::if_then(
                Expr::ne(Expr::var("a"), Expr::var("b")),
                vec![
                    Node::let_bind("da", Expr::load(depth, Expr::var("a"))),
                    Node::let_bind("db", Expr::load(depth, Expr::var("b"))),
                    Node::if_then_else(
                        Expr::gt(Expr::var("da"), Expr::var("db")),
                        vec![Node::assign("a", Expr::load(idom, Expr::var("a")))],
                        vec![Node::assign("b", Expr::load(idom, Expr::var("b")))],
                    ),
                ],
            )],
        ),
    ]
}

/// Build the standalone LCA intersection sub-operation.
#[must_use]
pub fn dominator_tree_lca_program(node_count: u32) -> Program {
    let count = node_count.max(1);
    let mut body = vec![Node::let_bind("a", Expr::u32(1))];
    body.extend(dominator_tree_lca_body(
        node_count,
        "idom",
        "depth",
        Expr::u32(2),
    ));
    body.push(Node::store("out_lca", Expr::u32(0), Expr::var("a")));
    let guarded = vec![Node::if_then(
        Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0)),
        body,
    )];
    Program::wrapped(
        vec![
            BufferDecl::storage("idom", 0, BufferAccess::ReadOnly, DataType::U32).with_count(count),
            BufferDecl::storage("depth", 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(count),
            BufferDecl::output("out_lca", 2, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![wrap_anonymous_region(DOMINATOR_TREE_LCA_OP_ID, guarded)],
    )
}

/// The least common ancestor of the fixture's two deepest nodes.
const EXPECTED_DOMINATOR_TREE_LCA_BYTES: [u8; 4] = [3, 0, 0, 0];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        DOMINATOR_TREE_LCA_OP_ID,
        || dominator_tree_lca_program(4),
        Some(|| vec![vec![
            vyre_primitives::wire::pack_u32_slice(&[0, 3, 3, 0]),
            vyre_primitives::wire::pack_u32_slice(&[0, 2, 2, 1]),
        ]]),
        Some(|| vec![vec![EXPECTED_DOMINATOR_TREE_LCA_BYTES.to_vec()]]),
    )
}

/// The registered operation on its own: one sweep, with movement reported on
/// the `dt_changed` output buffer.
///
/// Serial on lane 0, like the fixpoint that composes it: the sweep reads the
/// forest it also writes, so a second lane would race it.
#[must_use]
pub fn dominator_tree_intersect_step(
    node_count: u32,
    pred_edge_count: u32,
    idom: &str,
    depth: &str,
) -> Program {
    let count = node_count.max(1);
    let mut body = vec![Node::let_bind("changed", Expr::u32(0))];
    body.extend(dominator_tree_intersect_step_body(
        node_count, idom, depth, "changed",
    ));
    body.push(Node::store(
        CHANGED_BUFFER,
        Expr::u32(0),
        Expr::var("changed"),
    ));
    Program::wrapped(
        vec![
            BufferDecl::storage("pred_offsets", 0, BufferAccess::ReadOnly, DataType::U32)
                .with_count(node_count.saturating_add(1)),
            BufferDecl::storage("pred_targets", 1, BufferAccess::ReadOnly, DataType::U32)
                .with_count(pred_edge_count.max(1)),
            BufferDecl::storage(idom, 2, BufferAccess::ReadWrite, DataType::U32).with_count(count),
            BufferDecl::storage(depth, 3, BufferAccess::ReadOnly, DataType::U32).with_count(count),
            BufferDecl::storage(
                CHANGED_BUFFER,
                CHANGED_BINDING,
                BufferAccess::ReadWrite,
                DataType::U32,
            )
            .with_count(1),
        ],
        [1, 1, 1],
        vec![wrap_anonymous_region(
            OP_ID,
            vec![Node::if_then(
                Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0)),
                body,
            )],
        )],
    )
}
