//! Whole-grid fence detection over a semantic Program.
//!
//! A `Node::Barrier { ordering: MemoryOrdering::GridSync }` synchronizes every
//! workgroup in the launch. Only a cooperative launch executes one inside a
//! single kernel, so the compiler must know a program carries one before it
//! selects a plan: on a device without cooperative launch the fence is a
//! compile failure, not an emit failure.
//!
//! This module is the one owner of the predicate. `vyre-driver` reads it
//! through [`requires_grid_sync`] rather than walking node bodies again.

use vyre_foundation::ir::{MemoryOrdering, Node, Program};
use vyre_foundation::visit::any_descendant;

/// True when `program` contains a whole-grid fence anywhere in its body.
///
/// The walk is deep. `validate::barrier` rejects only a barrier in divergent
/// control flow, so a fence inside a uniform `If` or a counted `Loop` is a
/// legal program that reaches here. Reporting such a program as fence-free
/// would admit it on a device that cannot run the fence, where the barrier
/// lowers to a workgroup barrier and the kernel runs unsynchronized.
#[must_use]
pub fn requires_grid_sync(program: &Program) -> bool {
    if !program.stats().has_node_barrier() {
        return false;
    }
    program.entry().iter().any(|node| {
        any_descendant(node, &mut |candidate| {
            matches!(
                candidate,
                Node::Barrier {
                    ordering: MemoryOrdering::GridSync,
                    ..
                }
            )
        })
    })
}
