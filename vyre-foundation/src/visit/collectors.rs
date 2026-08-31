//! Program-level fact collectors and convenience wrappers.

use std::collections::HashSet;
use std::sync::Arc;

use crate::ir_inner::model::expr::{Expr, Ident};
use crate::ir_inner::model::program::Program;

use super::walk::walk_exprs;

/// This is a convenience wrapper around the visitor that extracts the set
/// of buffer identifiers actually used by the program. It is used by
/// validation and lowering to check that every declared buffer is
/// referenced and that no undeclared buffer is accessed.
///
/// The implementation uses a single combined traversal ([`super::walk::walk_nodes_and_exprs`])
/// instead of the previous two-pass approach.
///
/// # Examples
///
/// ```
/// use vyre::ir::Program;
/// use vyre_foundation::visit::referenced_buffers;
///
/// let program = Program::empty();
/// let buffers = referenced_buffers(&program);
/// assert!(buffers.is_empty());
/// ```
#[must_use]
#[inline]
pub fn referenced_buffers(program: &Program) -> HashSet<Ident> {
    // ProgramFacts::buffer_refs already enumerates every buffer-touching
    // node and expression in the program (Store/IndirectDispatch/AsyncLoad/
    // AsyncStore plus Load/BufLen/Atomic via the same SoA walk). Reuse the
    // OnceLock-cached facts instead of re-walking the entire tree with a
    // dedicated NodeSink + ExprSink pair.
    let facts = crate::optimizer::program_soa::ProgramFacts::build_cached(program);
    let mut names = HashSet::with_capacity(program.buffers().len());
    for (_, name, _) in facts.buffer_refs() {
        names.insert(name.clone());
    }
    names
}

/// Collect operation IDs from every [`Expr::Call`] in traversal order.
///
/// This helper is used by the inliner and the conform gate to discover
/// which operations a program depends on. The returned vector preserves
/// the order of first appearance.
///
/// # Examples
///
/// ```
/// use vyre::ir::{Expr, Node, Program};
/// use vyre_foundation::visit::collect_call_op_ids;
///
/// let program = Program::wrapped(
///     Vec::new(),
///     [1, 1, 1],
///     vec![Node::let_bind("x", Expr::call("primitive.math.add", vec![Expr::u32(1)]))],
/// );
/// assert_eq!(
///     collect_call_op_ids(&program)
///         .into_iter()
///         .map(|id| id.to_string())
///         .collect::<Vec<_>>(),
///     vec!["primitive.math.add".to_string()]
/// );
/// ```
#[must_use]
#[inline]
pub fn collect_call_op_ids(program: &Program) -> Vec<Arc<str>> {
    // Cached call_count is the exact number of Expr::Call sites in
    // the program. When it is zero, skip the entire expression walk.
    // When non-zero, pre-size the output to the exact count so we
    // never resize during the walk.
    let stats = program.stats();
    let call_count = stats.call_count as usize;
    if call_count == 0 {
        return Vec::new();
    }
    let mut op_ids = Vec::with_capacity(call_count);
    walk_exprs(program, |expr| {
        if let Expr::Call { op_id, .. } = expr {
            op_ids.push(op_id.shared_text());
        }
    });
    op_ids
}
