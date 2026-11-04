//! Descriptor fixtures for the host-input binding contracts.
//!
//! `descriptor_buffer_bindings` and `host_input_slots` are checked from two
//! test modules in this crate, and each one built the same `BindingSlot` and
//! the same empty `KernelDescriptor` around it. The grid is here as well, so a
//! memory class or a visibility added to lowering widens every case that reads
//! it rather than the one whose copy someone remembered to update.

use vyre_lower::{
    BindingLayout, BindingSlot, BindingVisibility, Dispatch, GridIndexSpace, KernelBody,
    KernelDescriptor, MemoryClass,
};

/// One four-element `U32` binding at `index`.
pub(crate) fn slot(
    index: u32,
    name: &str,
    memory_class: MemoryClass,
    visibility: BindingVisibility,
) -> BindingSlot {
    BindingSlot {
        slot: index,
        element_type: vyre_foundation::ir::DataType::U32,
        element_count: Some(4),
        memory_class,
        visibility,
        name: name.to_owned(),
    }
}

/// A descriptor carrying `slots` and no body, which is all a binding-metadata
/// derivation reads.
pub(crate) fn descriptor_of(slots: Vec<BindingSlot>) -> KernelDescriptor {
    KernelDescriptor {
        id: String::from("host-input-fixture"),
        bindings: BindingLayout { slots },
        dispatch: Dispatch {
            workgroup_size: [1, 1, 1],
            grid_index: GridIndexSpace::default(),
        },
        body: KernelBody {
            ops: Vec::new(),
            child_bodies: Vec::new(),
            literals: Vec::new(),
        },
    }
}

/// Every `MemoryClass`. The match carries no catch-all arm, so a class added
/// to the lowering descriptor fails to compile here until someone records
/// whether a binding in it is fed from the host.
pub(crate) fn all_memory_classes() -> Vec<MemoryClass> {
    let classes = vec![
        MemoryClass::Global,
        MemoryClass::Shared,
        MemoryClass::Constant,
        MemoryClass::Uniform,
        MemoryClass::Scratch,
    ];
    for class in &classes {
        match class {
            MemoryClass::Global
            | MemoryClass::Shared
            | MemoryClass::Constant
            | MemoryClass::Uniform
            | MemoryClass::Scratch => {}
        }
    }
    classes
}

/// Every `BindingVisibility`, closed against additions the same way.
pub(crate) fn all_visibilities() -> Vec<BindingVisibility> {
    let visibilities = vec![
        BindingVisibility::ReadOnly,
        BindingVisibility::WriteOnly,
        BindingVisibility::ReadWrite,
    ];
    for visibility in &visibilities {
        match visibility {
            BindingVisibility::ReadOnly
            | BindingVisibility::WriteOnly
            | BindingVisibility::ReadWrite => {}
        }
    }
    visibilities
}

/// One slot per `MemoryClass` and `BindingVisibility` pair, so a rule that
/// holds for the read-only global case and fails for a uniform or a write-only
/// one cannot pass.
pub(crate) fn full_grid() -> Vec<BindingSlot> {
    let mut slots = Vec::new();
    let mut index = 0u32;
    for class in all_memory_classes() {
        for visibility in all_visibilities() {
            slots.push(slot(index, &format!("buf{index}"), class, visibility));
            index += 1;
        }
    }
    slots
}
