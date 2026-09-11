//! Execution markers a lowered program body still carries.
//!
//! Lowering resolves every logical execution marker into a physical one, and a
//! test proves that by counting what survived: a logical index or a logical
//! barrier that reaches an emitter is the defect, and the physical axis profile
//! states which physical marker each logical one became. The counting walk is
//! the same question wherever it is asked, so it is one function and the
//! per-suite part is the expected census.

use vyre_foundation::ir::{Expr, MemoryOrdering, Node};
use vyre_foundation::visit::{for_each_expr, for_each_node};

/// The execution markers one program body states.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct MarkerCensus {
    /// Logical index, tile, and within-tile expressions.
    pub logical: usize,
    /// Physical invocation, workgroup, and local expressions, in that order.
    pub physical_axes: [usize; 3],
    /// Logical barrier statements.
    pub logical_barriers: usize,
    /// The ordering of every physical barrier, in body order.
    pub physical_orderings: Vec<MemoryOrdering>,
}

/// Count the execution markers `body` carries.
#[must_use]
pub fn census(body: &[Node]) -> MarkerCensus {
    let mut out = MarkerCensus::default();
    for_each_expr(body, |expr| match expr {
        Expr::LogicalIndex { .. }
        | Expr::LogicalTileId { .. }
        | Expr::LogicalWithinTileId { .. } => {
            out.logical += 1;
        }
        Expr::InvocationId { .. } => out.physical_axes[0] += 1,
        Expr::WorkgroupId { .. } => out.physical_axes[1] += 1,
        Expr::LocalId { .. } => out.physical_axes[2] += 1,
        _ => {}
    });
    for_each_node(body, |node| match node {
        Node::LogicalBarrier { .. } => out.logical_barriers += 1,
        Node::Barrier { ordering } => out.physical_orderings.push(*ordering),
        _ => {}
    });
    out
}

/// One expression naming all three logical position markers, one axis each.
///
/// A body built from this reaches lowering with every logical position form
/// present, so a census over the result reports each form's physical
/// replacement rather than one form's.
#[must_use]
pub fn logical_marker_sum() -> Expr {
    Expr::add(
        Expr::logical_index(0),
        Expr::add(
            Expr::logical_tile_index(1),
            Expr::logical_within_tile_index(2),
        ),
    )
}
