//! Shared IR fixtures for the loop restructuring passes' tests.
//!
//! `loop_fusion` and `legality` assert against the same shape: two adjacent
//! `Node::Loop` siblings over one literal range, each storing to its own
//! buffer. Written out, one such loop is five lines and a pair is ten, so two
//! tests that differ in a single buffer name sat ten identical lines apart and
//! the pair of files reported a hundred duplicated lines of fixtures with no
//! duplicated logic behind them. This module owns the builders; a test names
//! only what it varies.

use crate::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use crate::transform::visit::child_bodies;

/// A read-write `u32` storage buffer with eight elements.
pub(super) fn buf(name: &str) -> BufferDecl {
    BufferDecl::storage(name, 0, BufferAccess::ReadWrite, DataType::U32).with_count(8)
}

/// `entry` wrapped in a program declaring buffers `a`, `b`, and `c`.
pub(super) fn program(entry: Vec<Node>) -> Program {
    Program::wrapped(vec![buf("a"), buf("b"), buf("c")], [1, 1, 1], entry)
}

/// `for var in 0..to { body }`.
pub(super) fn loop_over(var: &str, to: u32, body: Vec<Node>) -> Node {
    Node::loop_for(var, Expr::u32(0), Expr::u32(to), body)
}

/// `for var in 0..to { buffer[var] = value }`, the one-statement map loop both
/// passes are written against.
pub(super) fn store_loop(var: &str, to: u32, buffer: &str, value: u32) -> Node {
    loop_over(
        var,
        to,
        vec![Node::store(buffer, Expr::var(var), Expr::u32(value))],
    )
}

/// Every `Node::Loop` in `nodes`, at any depth, in source order.
///
/// Descent is [`child_bodies`], the one exhaustive owner of which variants nest.
/// The flattener this replaces ended in `_ => {}`, so a loop nested inside a
/// body-bearing variant it did not name was invisible to every assertion built
/// on it, and a pass that fused or split such a loop wrongly still read as
/// correct.
pub(super) fn loops_of(nodes: &[Node]) -> Vec<&Node> {
    let mut out = Vec::new();
    let mut stack: Vec<&Node> = nodes.iter().rev().collect();
    while let Some(node) = stack.pop() {
        if matches!(node, Node::Loop { .. }) {
            out.push(node);
        }
        for body in child_bodies(node).into_iter().rev() {
            stack.extend(body.iter().rev());
        }
    }
    out
}

/// How many `Node::Loop` nodes `nodes` holds, at any depth.
pub(super) fn count_loops(nodes: &[Node]) -> usize {
    loops_of(nodes).len()
}
