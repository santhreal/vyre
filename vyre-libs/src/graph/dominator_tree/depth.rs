//! One depth recomputation over a parent-pointer idom forest.
//!
//! Cooper-Harvey-Kennedy intersects two nodes by walking the deeper of the two
//! up the idom tree, so every relaxation sweep needs the depth of every node in
//! the forest the previous sweep left behind. The walk is its own query: the
//! forest goes in, one depth per node comes out, and nothing about the
//! predecessor lists or the fixpoint reaches into it.
//!
//! # Wire shape
//!
//! ```text
//! idom  : u32[node_count]   // parent pointer; entry is its own parent
//! depth : u32[node_count]   // idom-tree edges between the node and the entry
//! ```
//!
//! `IDOM_NONE` marks a node the fixpoint has not reached yet. Its depth is 0,
//! which is what the intersection needs: an unreached node never becomes the
//! deeper side of a comparison against a reached one.

use vyre_foundation::composition::{wrap_anonymous_region, wrap_child_region};
use vyre_foundation::ir::Ident;
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};

use super::program::IDOM_NONE;

/// Canonical op id for one idom-forest depth recomputation.
pub const OP_ID: &str = "vyre-libs::graph::dominator_tree_depth";

/// Build the depth recomputation over `node_count` nodes of `idom` into
/// `depth`.
///
/// The upward walk is bounded by `node_count` steps rather than by reaching the
/// entry, because a forest mid-fixpoint can still hold a parent chain that does
/// not terminate at the entry. A chain that runs out of steps keeps the depth
/// reached so far, and the next sweep corrects it.
#[must_use]
pub fn dominator_tree_depth_body(node_count: u32, idom: &str, depth: &str) -> Vec<Node> {
    vec![Node::loop_for(
        "v",
        Expr::u32(0),
        Expr::u32(node_count),
        vec![
            Node::let_bind("d", Expr::u32(0)),
            Node::let_bind("cur", Expr::var("v")),
            Node::loop_for(
                "depth_step",
                Expr::u32(0),
                Expr::u32(node_count),
                vec![Node::if_then(
                    Expr::ne(Expr::var("cur"), Expr::u32(0)),
                    vec![
                        Node::let_bind("parent", Expr::load(idom, Expr::var("cur"))),
                        Node::if_then(
                            Expr::and(
                                Expr::ne(Expr::var("parent"), Expr::var("cur")),
                                Expr::ne(Expr::var("parent"), Expr::u32(IDOM_NONE)),
                            ),
                            vec![
                                Node::assign("d", Expr::add(Expr::var("d"), Expr::u32(1))),
                                Node::assign("cur", Expr::var("parent")),
                            ],
                        ),
                    ],
                )],
            ),
            Node::store(depth, Expr::var("v"), Expr::var("d")),
        ],
    )]
}

/// The depth recomputation as a child region of `parent_op_id`.
///
/// The body is the one in [`dominator_tree_depth_body`], so the region names an
/// operation that reads exactly the two buffers the body touches.
#[must_use]
pub fn dominator_tree_depth_child(
    parent_op_id: &str,
    node_count: u32,
    idom: &str,
    depth: &str,
) -> Node {
    wrap_child_region(
        OP_ID,
        Ident::from(parent_op_id),
        dominator_tree_depth_body(node_count, idom, depth),
    )
}

/// The registered operation on its own: an idom forest in, one depth per node
/// out.
///
/// Serial on lane 0, like the fixpoint that composes it: the walk reads the
/// same forest every lane would write, so a second lane would race it.
#[must_use]
pub fn dominator_tree_depth(node_count: u32, idom: &str, depth: &str) -> Program {
    let count = node_count.max(1);
    Program::wrapped(
        vec![
            BufferDecl::storage(idom, 0, BufferAccess::ReadOnly, DataType::U32).with_count(count),
            BufferDecl::storage(depth, 1, BufferAccess::ReadWrite, DataType::U32).with_count(count),
        ],
        [1, 1, 1],
        vec![wrap_anonymous_region(
            OP_ID,
            vec![Node::if_then(
                Expr::eq(Expr::InvocationId { axis: 0 }, Expr::u32(0)),
                dominator_tree_depth_body(node_count, idom, depth),
            )],
        )],
    )
}
