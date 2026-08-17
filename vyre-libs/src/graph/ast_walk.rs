//! VAST tree walks for a tree encoded like [`vyre_foundation::vast::VastNode`]
//! (`first_child` plus `next_sibling`, rooted at node `0`).
//!
//! One walk, parameterized on [`VastWalkOrder`]. The IR uses parent links to
//! avoid an auxiliary stack: after visiting a node it descends to
//! `first_child` when present, otherwise it climbs parents until it finds a
//! valid `next_sibling`. Preorder emits a node before its descendants,
//! postorder after; the primitive owns both bodies behind one entry point, so
//! this dialect layer only tags the Region with the `vyre-libs` op id.

use crate::graph::vast_tree_walk;
use crate::graph::vast_tree_walk::VastWalkOrder;
use vyre_foundation::composition::{tag_program, wrap_anonymous_region};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_foundation::vast::{VastNode, NODE_STRIDE_U32, SENTINEL};

const PREORDER_OP_ID: &str = "vyre-libs::graph::ast_walk_preorder";
const POSTORDER_OP_ID: &str = "vyre-libs::graph::ast_walk_postorder";

fn op_id(order: VastWalkOrder) -> &'static str {
    match order {
        VastWalkOrder::Preorder => PREORDER_OP_ID,
        VastWalkOrder::Postorder => POSTORDER_OP_ID,
    }
}

/// Emit node indices for a VAST first-child / next-sibling tree in `order`.
///
/// # Panics
///
/// Panics on an invalid `node_count` or `out_cap`: an invalid launch shape must
/// not degrade to an inert kernel that walks nothing. Callers that need
/// structured handling use [`vast_tree_walk::try_ast_walk_order`].
#[must_use]
pub fn ast_walk(
    order: VastWalkOrder,
    nodes: &str,
    out: &str,
    node_count: u32,
    out_cap: u32,
) -> Program {
    tag_program(
        op_id(order),
        vast_tree_walk::try_ast_walk_order(order, nodes, out, node_count, out_cap)
            .unwrap_or_else(|error| panic!("{error}")),
    )
}

/// Emit preorder node indices for a VAST first-child / next-sibling tree.
///
/// # Panics
///
/// See [`ast_walk`].
#[must_use]
pub fn ast_walk_preorder(nodes: &str, out: &str, node_count: u32, out_cap: u32) -> Program {
    ast_walk(VastWalkOrder::Preorder, nodes, out, node_count, out_cap)
}

/// Emit postorder node indices for a general VAST first-child /
/// next-sibling tree rooted at node `0`.
///
/// # Panics
///
/// See [`ast_walk`].
#[must_use]
pub fn ast_walk_postorder_nodes(nodes: &str, out: &str, node_count: u32, out_cap: u32) -> Program {
    ast_walk(VastWalkOrder::Postorder, nodes, out, node_count, out_cap)
}

/// Emit `node_count - 1 - i` into `out[i]` for a spine postorder sequence.
///
/// This is the closed-form sequence for a degenerate single-child chain, not a
/// tree walk; general trees use [`ast_walk_postorder_nodes`].
#[must_use]
pub fn ast_walk_postorder(out: &str, node_count: u32) -> Program {
    let out_words = node_count.max(1);
    let body = vec![Node::loop_for(
        "i",
        Expr::u32(0),
        Expr::u32(node_count),
        vec![Node::store(
            out,
            Expr::var("i"),
            Expr::sub(Expr::u32(node_count.saturating_sub(1)), Expr::var("i")),
        )],
    )];

    Program::wrapped(
        vec![
            BufferDecl::storage(out, 0, BufferAccess::ReadWrite, DataType::U32)
                .with_count(out_words),
        ],
        [1, 1, 1],
        vec![wrap_anonymous_region(
            "vyre-libs::graph::ast_walk_postorder_spine",
            body,
        )],
    )
}

/// Pack a spine fixture: full VAST bytes plus the node-table slice.
///
/// Shared by both walk registrations and by the navigation contract tests, so
/// the tree under test is defined once.
#[cfg(test)]
#[must_use]
pub fn pack_spine_fixture(node_count: u32) -> (Vec<u8>, Vec<u8>) {
    let full = vyre_foundation::vast::pack_spine_vast(&vec![1u32; node_count as usize]);
    let node_len = (node_count as usize) * NODE_STRIDE_U32 * 4;
    let start = vyre_foundation::vast::HEADER_LEN;
    let region = full[start..start + node_len].to_vec();
    (full, region)
}

/// Pack a branching fixture:
///
/// ```text
/// 0
/// |- 1
/// |  `- 4
/// |- 2
/// `- 3
///    `- 5
/// ```
///
/// Preorder: `0, 1, 4, 2, 3, 5`; postorder: `4, 1, 2, 5, 3, 0`.
#[must_use]
pub fn pack_branching_fixture() -> Vec<u8> {
    let nodes = [
        VastNode {
            kind: 1,
            parent_idx: SENTINEL,
            first_child: 1,
            next_sibling: SENTINEL,
            src_file: 0,
            src_byte_off: 0,
            src_byte_len: 1,
            attr_off: 0,
            attr_len: 0,
            reserved: 0,
        },
        VastNode {
            kind: 1,
            parent_idx: 0,
            first_child: 4,
            next_sibling: 2,
            src_file: 0,
            src_byte_off: 1,
            src_byte_len: 1,
            attr_off: 0,
            attr_len: 0,
            reserved: 0,
        },
        VastNode {
            kind: 1,
            parent_idx: 0,
            first_child: SENTINEL,
            next_sibling: 3,
            src_file: 0,
            src_byte_off: 2,
            src_byte_len: 1,
            attr_off: 0,
            attr_len: 0,
            reserved: 0,
        },
        VastNode {
            kind: 1,
            parent_idx: 0,
            first_child: 5,
            next_sibling: SENTINEL,
            src_file: 0,
            src_byte_off: 3,
            src_byte_len: 1,
            attr_off: 0,
            attr_len: 0,
            reserved: 0,
        },
        VastNode {
            kind: 1,
            parent_idx: 1,
            first_child: SENTINEL,
            next_sibling: SENTINEL,
            src_file: 0,
            src_byte_off: 4,
            src_byte_len: 1,
            attr_off: 0,
            attr_len: 0,
            reserved: 0,
        },
        VastNode {
            kind: 1,
            parent_idx: 3,
            first_child: SENTINEL,
            next_sibling: SENTINEL,
            src_file: 0,
            src_byte_off: 5,
            src_byte_len: 1,
            attr_off: 0,
            attr_len: 0,
            reserved: 0,
        },
    ];
    let mut out = Vec::with_capacity(nodes.len() * NODE_STRIDE_U32 * 4);
    for node in nodes {
        out.extend_from_slice(&node.to_bytes());
    }
    out
}

fn harness_inputs() -> Vec<Vec<Vec<u8>>> {
    vec![vec![pack_branching_fixture(), vec![0u8; 32]]]
}

const EXPECTED_AST_WALK_PREORDER_OUTPUT_BYTES: [u8; 32] = [
    0, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];
const EXPECTED_AST_WALK_POSTORDER_OUTPUT_BYTES: [u8; 32] = [
    4, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 5, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        PREORDER_OP_ID,
        || ast_walk_preorder("nodes", "out", 6, 8),
        Some(harness_inputs),
        Some(|| vec![vec![EXPECTED_AST_WALK_PREORDER_OUTPUT_BYTES.to_vec()]]),
    )
    .with_category("graph")
}

inventory::submit! {
    vyre_foundation::operation::OperationRegistration::library(
        POSTORDER_OP_ID,
        || ast_walk_postorder_nodes("nodes", "out", 6, 8),
        Some(harness_inputs),
        Some(|| vec![vec![EXPECTED_AST_WALK_POSTORDER_OUTPUT_BYTES.to_vec()]]),
    )
    .with_category("graph")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Both entry points must route through the direction parameter, so the
    /// order argument alone decides which sequence the tagged Program emits.
    #[test]
    fn direction_parameter_selects_the_entry_point_program() {
        for (order, expected) in [
            (
                VastWalkOrder::Preorder,
                ast_walk_preorder("nodes", "out", 6, 8),
            ),
            (
                VastWalkOrder::Postorder,
                ast_walk_postorder_nodes("nodes", "out", 6, 8),
            ),
        ] {
            assert_eq!(
                ast_walk(order, "nodes", "out", 6, 8).fingerprint(),
                expected.fingerprint(),
                "{order:?} entry point must equal ast_walk in that direction"
            );
        }
        assert_ne!(
            ast_walk_preorder("nodes", "out", 6, 8).fingerprint(),
            ast_walk_postorder_nodes("nodes", "out", 6, 8).fingerprint(),
            "the two directions must not collapse to the same program"
        );
    }

    #[test]
    fn spine_preorder_is_the_identity_permutation() {
        let (_full, region) = pack_spine_fixture(3);
        assert_eq!(region.len(), 3 * NODE_STRIDE_U32 * 4);
        assert_eq!(
            vyre_foundation::vast::walk_preorder_indices(&region, 3, 16).unwrap(),
            vec![0u32, 1, 2],
            "spine preorder must be identity [0, 1, 2]"
        );
        assert!(
            vyre_foundation::validate::validate(&ast_walk_preorder("nodes", "out", 4, 8))
                .is_empty(),
            "ast_walk_preorder IR must pass the validator"
        );
    }

    #[test]
    fn spine_postorder_reverses_preorder() {
        let (_, region) = pack_spine_fixture(4);
        let pre = vyre_foundation::vast::walk_preorder_indices(&region, 4, 128).unwrap();
        let post = vyre_foundation::vast::walk_postorder_indices(&region, 4, 128).unwrap();
        assert_eq!(post, pre.iter().rev().copied().collect::<Vec<_>>());
        assert!(
            vyre_foundation::validate::validate(&ast_walk_postorder("out", 4)).is_empty(),
            "postorder spine program must validate"
        );
    }

    #[test]
    fn branching_tree_walks_both_directions() {
        let node_region = pack_branching_fixture();
        assert_eq!(
            vyre_foundation::vast::walk_preorder_indices(&node_region, 6, 128).unwrap(),
            vec![0, 1, 4, 2, 3, 5]
        );
        assert_eq!(
            vyre_foundation::vast::walk_postorder_indices(&node_region, 6, 128).unwrap(),
            vec![4, 1, 2, 5, 3, 0]
        );
        assert!(
            vyre_foundation::validate::validate(&ast_walk_preorder("nodes", "out", 6, 8))
                .is_empty()
        );
        assert!(
            vyre_foundation::validate::validate(&ast_walk_postorder_nodes("nodes", "out", 6, 8))
                .is_empty()
        );
    }
}
