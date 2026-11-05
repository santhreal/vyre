//! Shared helpers for optimizer and transform unit tests.

use crate::ir::{Node, Program};

/// Return the effective entry body of a `Program`.
///
/// F-IR-29 invariant: `Program::wrapped` produces an entry whose first (and
/// usually only) top-level node is `Node::Region`. Most tests that inspect
/// the optimized IR need to look inside that Region. This helper hides the
/// unwrap so tests stay consistent even when the program has already been
/// through `region_inline` and the wrapper is gone.
pub(crate) fn region_body(program: &Program) -> &[Node] {
    match program.entry() {
        [Node::Region { body, .. }] => body,
        other => other,
    }
}

/// How many nodes in `nodes` and every nested body satisfy `pred`.
///
/// Nineteen pass test modules each carried their own recursive counter for this
/// (`count_loops`, `count_ifs`, `count_stores`, `count_barriers`, …), every one
/// restating which node variants nest. A counter that misses a nesting variant
/// under-reports, which makes the assertion it feeds pass on a program the pass
/// mishandled. Descent comes from `visit::for_each_node`, so there is
/// one answer to "which variants nest" for tests and production alike.
pub(crate) fn count_nodes(nodes: &[Node], mut pred: impl FnMut(&Node) -> bool) -> usize {
    let mut count = 0;
    crate::visit::for_each_node(nodes, |node| {
        if pred(node) {
            count += 1;
        }
    });
    count
}
