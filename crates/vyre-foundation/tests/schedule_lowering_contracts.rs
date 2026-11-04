//! Selected-schedule lowering closes every schedule-free execution marker.
//!
//! WHY: composed library IR must not encode physical launch geometry. The
//! selected schedule introduces physical identifiers and barriers at one
//! boundary, and unchanged physical programs retain their entry allocation.

use vyre_foundation::ir::{Expr, MemoryOrdering, Node, Program};
use vyre_foundation::transform::schedule_lowering::lower_logical_schedule;
use vyre_foundation::visit::{for_each_expr, for_each_node};

fn every_ordering() -> Vec<MemoryOrdering> {
    (0..=u8::MAX)
        .filter_map(|tag| MemoryOrdering::from_wire_tag(tag).ok())
        .collect()
}

#[test]
fn every_logical_identity_axis_and_barrier_ordering_lowers_physically() {
    let mut nodes = Vec::new();
    for axis in 0..=2 {
        nodes.push(Node::let_bind(
            format!("global_{axis}"),
            Expr::logical_index(axis),
        ));
        nodes.push(Node::let_bind(
            format!("tile_{axis}"),
            Expr::logical_tile_index(axis),
        ));
        nodes.push(Node::let_bind(
            format!("within_{axis}"),
            Expr::logical_within_tile_index(axis),
        ));
    }
    let orderings = every_ordering();
    assert!(!orderings.is_empty());
    nodes.extend(orderings.iter().copied().map(Node::logical_barrier));

    let original = Program::wrapped(Vec::new(), [8, 4, 2], nodes);
    let (lowered, changed) = lower_logical_schedule(original.clone());
    assert!(changed);
    assert_eq!(lowered.workgroup_size(), original.workgroup_size());
    assert_eq!(lowered.buffers(), original.buffers());

    let mut logical = 0usize;
    let mut physical_axes = [0usize; 3];
    for_each_expr(lowered.entry(), |expr| match expr {
        Expr::LogicalIndex { .. }
        | Expr::LogicalTileId { .. }
        | Expr::LogicalWithinTileId { .. } => logical += 1,
        Expr::InvocationId { .. } => physical_axes[0] += 1,
        Expr::WorkgroupId { .. } => physical_axes[1] += 1,
        Expr::LocalId { .. } => physical_axes[2] += 1,
        _ => {}
    });
    let mut logical_barriers = 0usize;
    let mut physical_orderings = Vec::new();
    for_each_node(lowered.entry(), |node| match node {
        Node::LogicalBarrier { .. } => logical_barriers += 1,
        Node::Barrier { ordering } => physical_orderings.push(*ordering),
        _ => {}
    });

    assert_eq!(logical, 0);
    assert_eq!(logical_barriers, 0);
    assert_eq!(physical_axes, [3, 3, 3]);
    assert_eq!(physical_orderings, orderings);
}

#[test]
fn physical_program_is_not_rebuilt_or_reclassified() {
    let original = Program::wrapped(
        Vec::new(),
        [1, 1, 1],
        vec![
            Node::let_bind("gid", Expr::InvocationId { axis: 0 }),
            Node::barrier_with_ordering(MemoryOrdering::SeqCst),
        ],
    );
    let entry = original.entry().as_ptr();
    let (unchanged, changed) = lower_logical_schedule(original);
    assert!(!changed);
    assert_eq!(unchanged.entry().as_ptr(), entry);
}
