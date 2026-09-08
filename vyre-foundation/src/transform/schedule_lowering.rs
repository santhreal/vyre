//! Lower schedule-free logical execution markers into physical IR.
//!
//! Library programs contain logical domain, tile, and within-tile coordinates
//! plus logical synchronization boundaries. This transform is the only owner
//! that introduces physical invocation identifiers and barriers for composed
//! programs, after schedule selection and before descriptor construction.

use std::borrow::Cow;

use crate::ir::{Expr, Node, Program};
use crate::optimizer::rewrite::rewrite_expr;
use crate::transform::rewrite_walk::{rewrite_body, NodeRewrite};

struct ScheduleLowering;

impl NodeRewrite for ScheduleLowering {
    fn whole_node(&mut self, node: &Node) -> Option<Node> {
        match node {
            Node::LogicalBarrier { ordering } => Some(Node::Barrier {
                ordering: *ordering,
            }),
            _ => None,
        }
    }

    fn operand(&mut self, expr: &Expr) -> Option<Expr> {
        match rewrite_expr(expr, &mut |candidate| match candidate {
            Expr::LogicalIndex { axis } => Some(Expr::InvocationId { axis: *axis }),
            Expr::LogicalTileId { axis } => Some(Expr::WorkgroupId { axis: *axis }),
            Expr::LogicalWithinTileId { axis } => Some(Expr::LocalId { axis: *axis }),
            _ => None,
        }) {
            Cow::Borrowed(_) => None,
            Cow::Owned(rewritten) => Some(rewritten),
        }
    }
}

/// Apply a selected schedule's physical mapping to a borrowed program.
///
/// Returns `None` when `program` carries no logical execution marker, so a
/// caller that only needs the physical form pays no allocation for a program
/// that is already physical.
#[must_use]
pub fn lower_logical_schedule_borrowed(program: &Program) -> Option<Program> {
    let entry = rewrite_body(program.entry(), &mut ScheduleLowering)?;
    Some(program.with_rewritten_entry(entry))
}

/// Apply a selected schedule's physical identity and synchronization mapping.
///
/// Returns the original `Program` allocation when it contains no logical
/// execution markers. A changed program preserves buffers, metadata, and
/// workgroup policy while replacing only entry nodes.
#[must_use]
pub fn lower_logical_schedule(program: Program) -> (Program, bool) {
    match lower_logical_schedule_borrowed(&program) {
        Some(lowered) => (lowered, true),
        None => (program, false),
    }
}
