//! The dispatch-boundary rule that makes a read-only binding declaration true.
//!
//! A `BufferAccess::ReadOnly` or `BufferAccess::Uniform` declaration states
//! that the kernel does not write that memory. Concrete backends compile such
//! a slot into a read-only load path whose cache is not coherent with stores
//! the same kernel issues, so a caller that binds one allocation to a
//! read-only slot and to a writable slot in the same dispatch makes the
//! declaration false and the compiled read stale. Nothing in the resource ABI
//! stops that binding on its own: a resident handle is `Copy` and may appear
//! at any number of descriptor slots.
//!
//! The check accumulates one slot at a time inside the descriptor walk a
//! resident dispatch already performs. It costs a linear scan of the handles
//! bound so far and allocates nothing for a program with eight or fewer
//! caller-owned slots.

use smallvec::SmallVec;
use vyre_foundation::ir::BufferAccess;

use crate::{BackendError, ResidentHandle};

/// One caller-owned allocation observed at a descriptor slot.
#[derive(Clone, Copy)]
struct BoundAllocation<'name> {
    handle: ResidentHandle,
    name: &'name str,
    writable: bool,
}

/// Accumulating refusal for a read-only slot that aliases a writable slot.
#[derive(Default)]
pub struct ReadOnlyAliasCheck<'name> {
    bound: SmallVec<[BoundAllocation<'name>; 8]>,
}

impl<'name> ReadOnlyAliasCheck<'name> {
    /// An empty check, before any descriptor slot has been observed.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bound: SmallVec::new(),
        }
    }

    /// Record one descriptor slot and refuse the first conflicting pair.
    ///
    /// `handle` is `None` for a slot the backend stages for this dispatch
    /// alone. A per-dispatch staging allocation is reachable from exactly one
    /// slot, so it cannot alias another.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError::InvalidProgram`] when `handle` was already
    /// observed at a slot of the opposite direction.
    pub fn observe(
        &mut self,
        name: &'name str,
        access: BufferAccess,
        handle: Option<ResidentHandle>,
    ) -> Result<(), BackendError> {
        let Some(handle) = handle else {
            return Ok(());
        };
        // An unrecognized access is neither silently writable nor silently
        // read-only: both defaults are wrong. A variant added to the frozen
        // `BufferAccess` contract has to state which side of this rule it is
        // on before a dispatch can bind it.
        let writable = match access {
            BufferAccess::ReadOnly | BufferAccess::Uniform => false,
            BufferAccess::ReadWrite | BufferAccess::WriteOnly => true,
            // Workgroup memory is allocated per launch by the backend and is
            // never bound to a caller-owned allocation.
            BufferAccess::Workgroup => return Ok(()),
        };
        for earlier in &self.bound {
            if earlier.handle != handle || earlier.writable == writable {
                continue;
            }
            let (read_only, written) = if writable {
                (earlier.name, name)
            } else {
                (name, earlier.name)
            };
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: resident buffer {handle} is bound to read-only binding `{read_only}` and to writable binding `{written}` in one dispatch. A read-only declaration states that the kernel does not write that memory, and backends compile it into a non-coherent read-only load path. Bind `{written}` to its own allocation, or declare `{read_only}` as BufferAccess::ReadWrite."
                ),
            });
        }
        self.bound.push(BoundAllocation {
            handle,
            name,
            writable,
        });
        Ok(())
    }
}
