//! Structural queries over a built `Program`, shared by the suites that pin a
//! builder's emitted shape rather than its values.
//!
//! A shape assertion is only as trustworthy as the walker behind it: a walker
//! that forgets an `Expr` arm silently answers "no" and turns the assertion
//! green. Keeping one walker per question means a new IR node is fixed in one
//! place for every suite that asks.
//!
//! Every question is answered through [`shape_of`], one record per program.
//! A suite that asked one question directly left the other walkers with no
//! caller in that binary, so the shape is read whole and each suite states the
//! facts it pins.
//!
//! One copy of this module lived in each consuming crate's own `tests/` tree.
//! They drifted: one walked children through
//! [`vyre_foundation::visit::any_descendant`], the other hand-rolled the
//! descent behind a catch-all arm that answered "no" for every node kind it
//! did not list.

use std::fmt;
use vyre_foundation::visit::{any_descendant, any_expr_in, any_subexpr, for_each_node};

/// The structural facts a suite pins about a built program.
pub struct ProgramShape {
    /// Contains a loop at any nesting depth.
    ///
    /// A builder that lowers to a serial loop where the contract promises a
    /// parallel multi-block chain reports `true`.
    pub loops: bool,
    /// Reads `invocation_id` anywhere.
    ///
    /// A parallel builder must; a builder that lost its lane indexing reads
    /// none and would otherwise still pass a value comparison on a one-element
    /// input.
    pub reads_invocation_id: bool,
    /// Gates work behind `invocation_id.x == 0`.
    ///
    /// That gate serializes a dispatch onto one lane, so a builder that claims
    /// to expose parallel work must not contain one.
    pub gates_on_invocation_zero: bool,
    /// Number of grid-wide barriers.
    ///
    /// The multi-block scan chain is exactly Pass-A / Pass-B / Pass-C, so it
    /// needs exactly two. A dropped barrier reads as a lost cross-block
    /// dependency.
    pub grid_sync_barriers: usize,
}

impl fmt::Display for ProgramShape {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "loops={} invocation_id={} invocation_zero_gate={} grid_sync_barriers={}",
            self.loops,
            self.reads_invocation_id,
            self.gates_on_invocation_zero,
            self.grid_sync_barriers
        )
    }
}

/// Read every structural fact of `program` in one walk per question.
pub fn shape_of(program: &vyre_foundation::ir::Program) -> ProgramShape {
    let entry = program.entry();
    ProgramShape {
        loops: entry.iter().any(node_contains_loop),
        reads_invocation_id: any_expr_in(entry, &mut |expr| {
            matches!(expr, vyre_foundation::ir::Expr::LogicalIndex { .. })
        }),
        gates_on_invocation_zero: entry.iter().any(node_contains_invocation_zero_gate),
        grid_sync_barriers: entry.iter().map(node_grid_sync_barrier_count).sum(),
    }
}

fn node_contains_loop(node: &vyre_foundation::ir::Node) -> bool {
    use vyre_foundation::ir::Node;
    any_descendant(node, &mut |current| matches!(current, Node::Loop { .. }))
}

fn node_contains_invocation_zero_gate(node: &vyre_foundation::ir::Node) -> bool {
    use vyre_foundation::ir::Node;
    any_descendant(
        node,
        &mut |current| matches!(current, Node::If { cond, .. } if expr_is_invocation_zero(cond)),
    )
}

/// True when `expr` compares logical axis 0 against zero anywhere below it.
///
/// Sub-expressions come from the AST registry, so a new operand-carrying
/// variant is searched without an edit here. A hand-rolled descent answered
/// `false` for every variant it did not list, which turns a "does not gate on
/// invocation zero" assertion green for the programs it cannot see into.
fn expr_is_invocation_zero(expr: &vyre_foundation::ir::Expr) -> bool {
    use vyre_foundation::ir::{BinOp, Expr};
    any_subexpr(expr, &mut |current| {
        matches!(
            current,
            Expr::BinOp { op: BinOp::Eq, left, right }
                if matches!(
                    (&**left, &**right),
                    (Expr::LogicalIndex { axis: 0 }, Expr::LitU32(0))
                        | (Expr::LitU32(0), Expr::LogicalIndex { axis: 0 })
                )
        )
    })
}

fn node_grid_sync_barrier_count(node: &vyre_foundation::ir::Node) -> usize {
    use vyre_foundation::ir::MemoryOrdering;
    use vyre_foundation::ir::Node;
    let mut count = 0;
    for_each_node(std::slice::from_ref(node), |current| {
        if matches!(
            current,
            Node::LogicalBarrier {
                ordering: MemoryOrdering::GridSync
            }
        ) {
            count += 1;
        }
    });
    count
}
