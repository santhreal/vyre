//! Contracts for `vyre_runtime::resident_work_queue::advanced::hierarchical_atomics`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.
#![cfg(feature = "megakernel-batch")]

use vyre_foundation::ir::{Expr, Node};
use vyre_runtime::resident_work_queue::advanced::hierarchical_atomics::record_hit_to_ring_hierarchical;

#[test]
fn hierarchical_hit_writer_emits_real_ring_stores() {
    let nodes = record_hit_to_ring_hierarchical("is_hit");
    let store_count = count_stores(&nodes);
    assert_eq!(store_count, 4);
    assert!(contains_subgroup(&nodes));
    assert!(
        contains_subgroup_local_id(&nodes),
        "subgroup aggregation must elect one leader per subgroup, not only workgroup lane 0"
    );
}

fn count_stores(nodes: &[Node]) -> usize {
    let mut count = 0;
    vyre_foundation::visit::for_each_node(nodes, |node| {
        if matches!(node, Node::Store { .. }) {
            count += 1;
        }
    });
    count
}

fn contains_subgroup(nodes: &[Node]) -> bool {
    nodes.iter().any(|node| {
        matches!(
            node,
            Node::Let {
                value: Expr::SubgroupBallot { .. }
                    | Expr::SubgroupReduce { .. }
                    | Expr::SubgroupShuffle { .. },
                ..
            }
        )
    })
}

/// True when any expression anywhere in `nodes` reads the subgroup lane id.
///
/// Node descent, operand positions and sub-expression structure all come
/// from `vyre_foundation::visit`, the single owner of each. The
/// hand-written pair this replaces restated all three and ended each match
/// in a wildcard, so an operand position or expression kind added later
/// would have read as "no lane id" instead of failing to compile.
fn contains_subgroup_local_id(nodes: &[Node]) -> bool {
    let mut found = false;
    vyre_foundation::visit::for_each_expr(nodes, |expr| {
        found = found || matches!(expr, Expr::SubgroupLocalId);
    });
    found
}
