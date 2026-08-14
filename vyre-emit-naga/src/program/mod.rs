//! Compatibility helpers for legacy Program-shaped Naga callers.
//!
//! Public entry points in this module route through `vyre-lower` and the
//! descriptor emitter. The descriptor path is the only production Naga lowering
//! truth.

mod async_resume_guard;
mod atomic_scanner;
mod entry;
mod extension_ops;
mod trap_collector;
mod trap_sidecar;

pub(crate) use vyre_foundation::lower::LoweringError;

/// Map a core IR memory kind to the bind-group index used by compatibility
/// helpers that still inspect Program buffers.
#[must_use]
pub fn bind_group_for(kind: vyre_foundation::ir::MemoryKind) -> u32 {
    match kind {
        vyre_foundation::ir::MemoryKind::Uniform | vyre_foundation::ir::MemoryKind::Push => 1,
        _ => 0,
    }
}

pub use entry::emit_prepared_module_with_capabilities;
pub use entry::{emit_module, emit_module_with_capabilities, prepared_program};

pub use entry::{trap_sidecar_decl, trap_tags};
pub use trap_sidecar::{TrapTag, TRAP_SIDECAR_NAME, TRAP_SIDECAR_WORDS};

#[cfg(test)]
mod tests;
