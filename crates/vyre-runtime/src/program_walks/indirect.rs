//! Indirect-dispatch discovery over backend-neutral IR.

use std::ops::ControlFlow;

use vyre_foundation::ir::{Ident, Node, Program};
use vyre_foundation::visit::try_for_each_node;

use crate::backend::BackendError;

/// Command-buffer indirect dispatch source.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndirectDispatch {
    /// Buffer containing the indirect x/y/z workgroup tuple.
    pub count_buffer: Ident,
    /// Byte offset of the tuple in the buffer.
    pub count_offset: u64,
}

/// Locates the single [`Node::IndirectDispatch`] in a program, if any.
///
/// Descent is [`try_for_each_node`], the shared-read owner in
/// `vyre-foundation`, so this scan does not restate which `Node` variants nest.
/// It used to implement `visit::NodeVisitor`, which is abstract-by-default:
/// answering a question about one variant cost a no-op body, with its full
/// signature, for the other fifteen. That block of stubs was a second
/// enumeration of `Node` inside this crate, and `Node` is `#[non_exhaustive]`,
/// so it could only ever be as complete as the day it was written.
///
/// # Errors
///
/// Returns when the program is inconsistent (e.g. multiple indirect
/// sources, or a misaligned offset).
pub fn find_indirect_dispatch(program: &Program) -> Result<Option<IndirectDispatch>, BackendError> {
    if !program.has_indirect_dispatch() {
        return Ok(None);
    }
    let mut found: Option<IndirectDispatch> = None;
    let walk = try_for_each_node(program.entry(), |node| {
        let Node::IndirectDispatch {
            count_buffer,
            count_offset,
        } = node
        else {
            return ControlFlow::Continue(());
        };
        if count_offset % 4 != 0 {
            return ControlFlow::Break(BackendError::new(format!(
                "indirect dispatch offset {count_offset} is not 4-byte aligned. Fix: use a u32-aligned dispatch tuple."
            )));
        }
        let next = IndirectDispatch {
            count_buffer: count_buffer.clone(),
            count_offset: *count_offset,
        };
        if found.replace(next).is_some() {
            return ControlFlow::Break(BackendError::new(
                "program declares more than one indirect dispatch source. Fix: keep exactly one Node::IndirectDispatch per Program.",
            ));
        }
        ControlFlow::Continue(())
    });
    match walk {
        ControlFlow::Break(err) => Err(err),
        ControlFlow::Continue(()) => Ok(found),
    }
}
