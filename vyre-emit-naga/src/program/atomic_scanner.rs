//! Pre-emission scan that walks a `Program` to identify which buffers
//! are the targets of `Expr::Atomic*` (so element types must wrap in
//! `atomic<...>`) and which buffers receive any write at all (so
//! `BufferAccess` can be auto-downgraded to `ReadOnly` when nobody
//! writes them).

use std::ops::ControlFlow::{self, Break, Continue};

use rustc_hash::FxHashSet;

use vyre_foundation::ir::{Expr, Ident, Node};
use vyre_foundation::transform::visit::{node_buffer_refs, node_operands, try_for_each_node};
use vyre_foundation::visit::{visit_preorder, ExprVisitor};

use super::extension_ops::{scan_registered_atomic_expr, scan_registered_atomic_node};
use super::LoweringError;

/// The two buffer sets Naga emission needs from one walk of a node tree.
///
/// Both halves come out together because taking one without the other is what
/// went wrong: the atomic half decides whether an element type wraps in
/// `atomic<...>`, the write half decides whether `BufferAccess::ReadWrite`
/// survives, and a buffer that is an atomic target but not a write target
/// emits an `atomic<u32>` inside `var<storage, read>`, which Naga rejects.
#[derive(Debug, Default)]
pub(super) struct BufferTargets {
    /// Buffers named by an `Expr::Atomic` or by a registered opaque extension
    /// that performs one. `add_buffer` wraps these element types in
    /// `atomic<...>`.
    pub atomic: FxHashSet<Ident>,
    /// Buffers the dispatch writes, including every atomic target. A
    /// `ReadWrite` buffer outside this set is auto-downgraded to `ReadOnly`.
    pub writes: FxHashSet<Ident>,
}

/// Collect both buffer sets from `nodes` and every nested body.
///
/// Descent, operand positions and per-node buffer direction all come from the
/// exhaustive owners in `vyre_foundation::transform::visit`
/// ([`try_for_each_node`] over `child_bodies`, [`node_operands`],
/// [`node_buffer_refs`]), so a new `Node` variant is a compile error there
/// rather than a buffer this scan quietly reports as untouched. The hand-rolled
/// descent this replaces ended in `_ => {}` and therefore recorded the four
/// collective variants as writing nothing.
///
/// # Errors
///
/// Returns a lowering error when an opaque node or expression extension has no
/// registered atomic scanner, because such a payload may perform an atomic this
/// scan cannot see.
pub(super) fn scan_buffer_targets(
    nodes: &[Node],
    out: &mut BufferTargets,
) -> Result<(), LoweringError> {
    match try_for_each_node(nodes, |node| scan_node(node, out)) {
        Continue(()) => {}
        Break(error) => return Err(error),
    }
    // Every atomic target is also written. The atomic half is collected from
    // operand expressions and opaque payloads, which `node_buffer_refs` cannot
    // see, so the union happens once here rather than per node.
    out.writes.extend(out.atomic.iter().cloned());
    Ok(())
}

fn scan_node(node: &Node, out: &mut BufferTargets) -> ControlFlow<LoweringError> {
    let refs = node_buffer_refs(node);
    out.writes
        .extend(refs.writes.into_iter().flatten().map(Ident::from));
    if let Node::Opaque(extension) = node {
        // `refs.complete` is false for exactly this variant: core cannot
        // enumerate an extension's buffers, so the Naga registry must, and an
        // unregistered payload has to fail closed rather than scan as empty.
        match scan_registered_atomic_node(extension.as_ref(), &mut out.atomic) {
            Ok(true) => {}
            Ok(false) => {
                return Break(LoweringError::invalid(format!(
                    "unsupported opaque node `{}` in atomic scan. Fix: register NagaProgramScanAtomicNode for this extension before lowering to Naga.",
                    extension.extension_kind()
                )))
            }
            Err(error) => return Break(error),
        }
    }
    let mut atomics = AtomicTargetScanner {
        out: &mut out.atomic,
    };
    for operand in node_operands(node).into_iter().flatten() {
        visit_preorder(&mut atomics, operand)?;
    }
    Continue(())
}

struct AtomicTargetScanner<'a> {
    out: &'a mut FxHashSet<Ident>,
}

impl ExprVisitor for AtomicTargetScanner<'_> {
    type Break = LoweringError;

    fn visit_atomic(
        &mut self,
        _expr: &Expr,
        _: &vyre_foundation::ir::AtomicOp,
        buffer: &vyre_foundation::ir::Ident,
        _: &Expr,
        _: Option<&Expr>,
        _: &Expr,
    ) -> ControlFlow<Self::Break> {
        self.out.insert(Ident::from(buffer));
        Continue(())
    }

    fn visit_opaque_expr(
        &mut self,
        _: &Expr,
        ext: &dyn vyre_foundation::ir::ExprNode,
    ) -> ControlFlow<Self::Break> {
        match scan_registered_atomic_expr(ext, self.out) {
            Ok(true) => Continue(()),
            Ok(false) => Break(LoweringError::invalid(format!(
                "unsupported opaque expression `{}` in atomic scan. Fix: register NagaProgramScanAtomicExpr for this extension or lower it before Naga atomic-target analysis.",
                ext.debug_identity()
            ))),
            Err(error) => Break(error),
        }
    }
}

#[cfg(test)]
mod tests;
