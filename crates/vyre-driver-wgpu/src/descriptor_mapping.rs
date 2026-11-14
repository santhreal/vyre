//! Shared descriptor-slot mapping for WGPU emission and pipeline metadata.
//!
//! `KernelDescriptor` lowering feeds both the offline emitter and the live WGPU
//! pipeline builder. Keeping bind-group, memory-kind, and access mapping here
//! prevents the two paths from drifting under new memory classes.

/// Return the WGPU bind group used for a lowered memory class.
pub(crate) fn descriptor_bind_group(
    memory_class: vyre_compiler::lower::MemoryClass,
) -> Option<u32> {
    match memory_class {
        vyre_compiler::lower::MemoryClass::Shared | vyre_compiler::lower::MemoryClass::Scratch => {
            None
        }
        vyre_compiler::lower::MemoryClass::Uniform => Some(1),
        vyre_compiler::lower::MemoryClass::Global | vyre_compiler::lower::MemoryClass::Constant => {
            Some(0)
        }
    }
}

/// Return the `(group, slot)` key for a lowered memory class and binding slot.
pub(crate) fn descriptor_binding_key(
    memory_class: vyre_compiler::lower::MemoryClass,
    slot: u32,
) -> Option<(u32, u32)> {
    descriptor_bind_group(memory_class).map(|group| (group, slot))
}

/// Convert lowered binding visibility into core IR buffer access.
pub(crate) fn descriptor_buffer_access(
    visibility: vyre_compiler::lower::BindingVisibility,
) -> vyre_foundation::ir::BufferAccess {
    match visibility {
        vyre_compiler::lower::BindingVisibility::ReadOnly => {
            vyre_foundation::ir::BufferAccess::ReadOnly
        }
        vyre_compiler::lower::BindingVisibility::WriteOnly => {
            vyre_foundation::ir::BufferAccess::WriteOnly
        }
        vyre_compiler::lower::BindingVisibility::ReadWrite => {
            vyre_foundation::ir::BufferAccess::ReadWrite
        }
    }
}

/// Convert lowered memory classes into core IR memory tiers.
pub(crate) fn descriptor_memory_kind(
    memory_class: vyre_compiler::lower::MemoryClass,
) -> vyre_foundation::ir::MemoryKind {
    match memory_class {
        vyre_compiler::lower::MemoryClass::Shared => vyre_foundation::ir::MemoryKind::Shared,
        vyre_compiler::lower::MemoryClass::Constant => vyre_foundation::ir::MemoryKind::Readonly,
        vyre_compiler::lower::MemoryClass::Uniform => vyre_foundation::ir::MemoryKind::Uniform,
        vyre_compiler::lower::MemoryClass::Global => vyre_foundation::ir::MemoryKind::Global,
        vyre_compiler::lower::MemoryClass::Scratch => vyre_foundation::ir::MemoryKind::Local,
    }
}
