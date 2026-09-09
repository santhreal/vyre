//! Rename one buffer across a Program's declaration table and every reference
//! to it.
//!
//! Fusion unifies buffers by name: two arms that name one buffer share one
//! declaration, one binding slot, and one hazard record, and two arms that
//! name the same storage differently share nothing. A producer arm writing
//! `sum_out` and a consumer arm reading `s_in` therefore fuse into a module
//! with two buffers, no read-after-write barrier between the arms, and a
//! read-only declaration the launch demands caller bytes for. Renaming the
//! consumer's declaration to the producer's name before the merge is what
//! makes the value edge visible to that unification.
//!
//! The rename is refused rather than partially applied. A node or expression
//! carrying an out-of-tree payload states no buffer references core can
//! enumerate, so a rename over it cannot be proven complete, and a Program
//! that keeps one stale reference lowers to a load from a buffer it does not
//! declare.

use std::sync::Arc;

use crate::ir::{Expr, Ident, Node, Program};
use crate::visit::{
    expr_buffer_name_mut, node_buffer_names_mut, push_expr_children_mut, push_node_operands_mut,
    walk_nodes_mut, ExprBufferName, ExprStackMut,
};

use super::{FusionBufferRenameError, FusionError};

/// `program` with buffer `from` renamed to `to`, table and references together.
///
/// # Errors
///
/// Returns [`FusionError::BufferRename`] when `program` declares no buffer
/// named `from`, when it already declares a different buffer named `to`, or
/// when a node or expression carries an out-of-tree payload whose buffer
/// references cannot be enumerated.
pub fn rename_buffer(program: &Program, from: &str, to: &str) -> Result<Program, FusionError> {
    let refuse = |fix: &'static str| {
        Err(FusionError::BufferRename(FusionBufferRenameError {
            from: from.to_string(),
            to: to.to_string(),
            fix,
        }))
    };
    let mut declares_from = false;
    for buffer in program.buffers() {
        if buffer.name() == from {
            declares_from = true;
        } else if buffer.name() == to {
            return refuse(
                "give the fused arms one buffer name per value, or rename the colliding declaration first",
            );
        }
    }
    if !declares_from {
        return refuse("rename a buffer the Program declares");
    }

    let buffers = program
        .buffers()
        .iter()
        .map(|buffer| {
            if buffer.name() == from {
                let mut renamed = buffer.clone();
                renamed.name = Arc::from(to);
                renamed
            } else {
                buffer.clone()
            }
        })
        .collect();
    let mut renamed = program.with_rewritten_buffers(buffers);
    let replacement = Ident::from(to);
    let mut complete = true;
    walk_nodes_mut(&mut renamed, |node| {
        rename_node_buffer(node, from, &replacement, &mut complete);
    });
    if !complete {
        return refuse("split the fused arms rather than renaming through an opaque payload");
    }
    Ok(renamed)
}

/// Replace `from` with `replacement` at every buffer position `node` reaches,
/// clearing `complete` when a payload states references core cannot enumerate.
fn rename_node_buffer(node: &mut Node, from: &str, replacement: &Ident, complete: &mut bool) {
    let names = node_buffer_names_mut(node);
    if !names.is_complete() {
        *complete = false;
    }
    for name in names.into_names() {
        if name.as_str() == from {
            *name = replacement.duplicate_handle();
        }
    }
    let mut stack = ExprStackMut::new();
    push_node_operands_mut(node, &mut stack);
    while let Some(expr) = stack.pop() {
        rename_expr_buffer(expr, from, replacement, complete, &mut stack);
    }
}

fn rename_expr_buffer<'a>(
    expr: &'a mut Expr,
    from: &str,
    replacement: &Ident,
    complete: &mut bool,
    stack: &mut ExprStackMut<'a>,
) {
    match expr_buffer_name_mut(expr) {
        ExprBufferName::None => {}
        ExprBufferName::Named(name) => {
            if name.as_str() == from {
                *name = replacement.duplicate_handle();
            }
        }
        ExprBufferName::Unknown => *complete = false,
    }
    push_expr_children_mut(expr, stack);
}
