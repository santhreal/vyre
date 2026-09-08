//! KernelDescriptor-to-WGPU binding metadata.
//!
//! This module owns compile-time binding reflection: converting lowered
//! `KernelDescriptor` slots into stable `BufferBindingInfo`, bind-group layout
//! fingerprints, live WGPU bind-group layouts, and trap sidecar tags. The
//! parent `pipeline` module orchestrates compilation and dispatch only.

use std::sync::Arc;

use rustc_hash::FxHashSet;
use vyre_driver::{BackendError, BackendLayoutClass, BackendLayoutFingerprint, BackendLayoutSlot};
use vyre_emit_naga::program::TrapTag;
use vyre_lower::TRAP_SIDECAR_NAME;

use crate::descriptor_mapping::{
    descriptor_bind_group, descriptor_buffer_access, descriptor_memory_kind,
};
use crate::pipeline::element_size_bytes;

/// Metadata for one buffer binding derived from a `Program` at compile time.
#[derive(Clone, Debug)]
pub(crate) struct BufferBindingInfo {
    /// `group N` slot.
    pub group: u32,
    /// `binding slot N` slot.
    pub binding: u32,
    /// Buffer name referenced by IR loads/stores.
    pub name: Arc<str>,
    /// Access mode.
    pub access: vyre_foundation::ir::BufferAccess,
    /// Memory tier.
    pub kind: vyre_foundation::ir::MemoryKind,
    /// Non-binding optimization hints.
    pub hints: vyre_foundation::ir::MemoryHints,
    /// Element type.
    pub element: vyre_foundation::ir::DataType,
    /// Static element count (`0` means runtime-sized).
    pub count: u32,
    /// Whether this binding is returned to the caller after dispatch.
    pub is_output: bool,
    /// Whether this writable binding must preserve caller-supplied initial bytes.
    pub preserve_input_contents: bool,
    /// Backend-owned trap sidecar; not supplied by callers and not returned as
    /// a public output.
    pub internal_trap: bool,
    /// Whether one caller-provided input slot supplies this binding's contents.
    ///
    /// Recorded from `BufferDecl::consumes_host_input`, the single definition of
    /// the host input ABI, so each binding walk reads one answer rather than
    /// re-deriving the rule from flattened fields that omit `pipeline_live_out`.
    pub consumes_host_input: bool,
}

pub(crate) fn descriptor_buffer_bindings(
    descriptor: &vyre_lower::KernelDescriptor,
    public_output_bindings: &FxHashSet<u32>,
    host_input_bindings: &FxHashSet<u32>,
) -> Result<Vec<BufferBindingInfo>, BackendError> {
    let mut bindings = Vec::new();
    vyre_driver::allocation::try_reserve_vec_to_capacity(
        &mut bindings,
        descriptor.bindings.slots.len(),
    )
    .map_err(|source| {
            BackendError::new(format!(
                "descriptor buffer binding allocation failed for {} slots: {source}. Fix: split the lowered kernel before WGPU pipeline metadata extraction.",
                descriptor.bindings.slots.len()
            ))
        })?;
    for slot in &descriptor.bindings.slots {
        let Some(group) = descriptor_bind_group(slot.memory_class) else {
            continue;
        };
        let access = descriptor_buffer_access(slot.visibility);
        let internal_trap = slot.name == TRAP_SIDECAR_NAME;
        let is_output = public_output_bindings.contains(&slot.slot) && !internal_trap;
        let consumes_host_input = host_input_bindings.contains(&slot.slot) && !internal_trap;
        let preserve_input_contents =
            access == vyre_foundation::ir::BufferAccess::ReadWrite && consumes_host_input;
        bindings.push(BufferBindingInfo {
            group,
            binding: slot.slot,
            name: Arc::from(slot.name.as_str()),
            access,
            kind: descriptor_memory_kind(slot.memory_class),
            hints: vyre_foundation::ir::MemoryHints::default(),
            element: slot.element_type.clone(),
            count: descriptor_element_count(slot.element_count),
            is_output,
            preserve_input_contents,
            internal_trap,
            consumes_host_input,
        });
    }
    Ok(bindings)
}

fn descriptor_element_count(element_count: Option<u32>) -> u32 {
    element_count.unwrap_or_default()
}

pub(crate) fn bind_group_layout_fingerprint(
    bindings: &[BufferBindingInfo],
) -> Result<BackendLayoutFingerprint, BackendError> {
    let mut slots = Vec::new();
    vyre_driver::allocation::try_reserve_vec_to_capacity(&mut slots, bindings.len()).map_err(|source| {
        BackendError::new(format!(
            "bind-group layout fingerprint allocation failed for {} bindings: {source}. Fix: split the lowered kernel before WGPU pipeline metadata extraction.",
            bindings.len()
        ))
    })?;
    for binding in bindings {
        let class = match binding.kind {
            vyre_foundation::ir::MemoryKind::Uniform | vyre_foundation::ir::MemoryKind::Push => {
                BackendLayoutClass::Uniform
            }
            _ => BackendLayoutClass::Storage,
        };
        let read_only = matches!(binding.kind, vyre_foundation::ir::MemoryKind::Readonly)
            || matches!(
                binding.access,
                vyre_foundation::ir::BufferAccess::ReadOnly
                    | vyre_foundation::ir::BufferAccess::Uniform
            );
        slots.push(BackendLayoutSlot {
            group: binding.group,
            binding: binding.binding,
            class,
            read_only,
            element_size: element_size_bytes(&binding.element)?,
        });
    }
    Ok(BackendLayoutFingerprint::new(slots))
}

pub(crate) fn create_bind_group_layouts(
    device: &wgpu::Device,
    buffer_bindings: &[BufferBindingInfo],
    max_group: u32,
) -> Result<Arc<[Arc<wgpu::BindGroupLayout>]>, BackendError> {
    let group_count = max_group.checked_add(1).ok_or_else(|| {
        BackendError::new(
            "bind-group layout count overflowed u32. Fix: lower the maximum bind-group index before WGPU pipeline creation.",
        )
    })?;
    let group_count = usize::try_from(group_count).map_err(|source| {
        BackendError::new(format!(
            "bind-group layout count cannot fit host usize: {source}. Fix: reduce the maximum bind-group index before WGPU pipeline creation."
        ))
    })?;
    let mut layouts: Vec<Arc<wgpu::BindGroupLayout>> = Vec::new();
    vyre_driver::allocation::try_reserve_vec_to_capacity(&mut layouts, group_count).map_err(|source| {
        BackendError::new(format!(
            "bind-group layout vector allocation failed for {group_count} groups: {source}. Fix: split the lowered kernel before WGPU pipeline creation."
        ))
    })?;
    for group_index in 0..=max_group {
        let group_binding_count = buffer_bindings
            .iter()
            .filter(|binding| binding.group == group_index)
            .count();
        let mut entries = Vec::new();
        vyre_driver::allocation::try_reserve_vec_to_capacity(&mut entries, group_binding_count).map_err(|source| {
            BackendError::new(format!(
                "bind-group layout entry allocation failed for group {group_index} with {group_binding_count} bindings: {source}. Fix: split the lowered kernel before WGPU pipeline creation."
            ))
        })?;
        for binding in buffer_bindings
            .iter()
            .filter(|binding| binding.group == group_index)
        {
            let ty = match binding.kind {
                vyre_foundation::ir::MemoryKind::Uniform
                | vyre_foundation::ir::MemoryKind::Push => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                _ => {
                    let read_only =
                        matches!(binding.kind, vyre_foundation::ir::MemoryKind::Readonly)
                            || matches!(
                                binding.access,
                                vyre_foundation::ir::BufferAccess::ReadOnly
                                    | vyre_foundation::ir::BufferAccess::Uniform
                            );
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    }
                }
            };
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: binding.binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty,
                count: None,
            });
        }
        layouts.push(Arc::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("vyre P-6 bind group layout"),
                entries: &entries,
            },
        )));
    }
    Ok(layouts.into())
}

/// The descriptor's trap tag table, as this backend decodes sidecar words with.
///
/// `TrapTag` is an alias of the owner's pair type, so this is the owner's table
/// verbatim: no reprojection, no second allocation, and no way for a code to
/// mean one thing here and another in an emitter.
pub(crate) fn descriptor_trap_tags(
    descriptor: &vyre_lower::KernelDescriptor,
) -> Result<Vec<TrapTag>, BackendError> {
    vyre_lower::descriptor_trap_tags(&descriptor.body).map_err(|source| {
        BackendError::new(format!(
            "descriptor trap tag table unavailable: {source}. Fix: split nested kernel bodies before descriptor metadata extraction."
        ))
    })
}

// Inline: `descriptor_buffer_bindings` is crate-private, and it is the single
// place a canonical host-input answer is recorded onto binding metadata. A
// device is never reached, so this runs on any host.
#[cfg(test)]
mod tests {
    use super::*;
    use vyre_lower::{
        BindingLayout, BindingSlot, BindingVisibility, Dispatch, GridIndexSpace, KernelBody,
        KernelDescriptor, MemoryClass,
    };

    /// Every `MemoryClass`. The match carries no catch-all arm, so a class
    /// added to the lowering descriptor fails to compile here until someone
    /// records whether a binding in it is fed from the host.
    fn all_memory_classes() -> Vec<MemoryClass> {
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
    fn all_visibilities() -> Vec<BindingVisibility> {
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

    fn slot(
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

    fn descriptor_of(slots: Vec<BindingSlot>) -> KernelDescriptor {
        KernelDescriptor {
            id: String::from("host-input-projection"),
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

    /// One slot per `MemoryClass` and `BindingVisibility` pair, so a rule that
    /// holds for the read-only global case and fails for a uniform or a
    /// write-only one cannot pass.
    fn full_grid() -> Vec<BindingSlot> {
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

    #[test]
    fn recorded_host_input_answer_is_the_supplied_set() {
        let slots = full_grid();
        // Alternating membership, so neither an all-true nor an all-false
        // projection can agree with it.
        let host_inputs: FxHashSet<u32> = slots
            .iter()
            .filter(|slot| slot.slot % 2 == 0)
            .map(|slot| slot.slot)
            .collect();
        let bindings =
            descriptor_buffer_bindings(&descriptor_of(slots), &FxHashSet::default(), &host_inputs)
                .expect("binding metadata for a well-formed descriptor");
        assert!(
            !bindings.is_empty(),
            "the grid must reach at least one bind-group-mapped class"
        );
        for binding in &bindings {
            assert_eq!(
                binding.consumes_host_input,
                host_inputs.contains(&binding.binding),
                "binding {} recorded {} for a host-input set that says {}",
                binding.binding,
                binding.consumes_host_input,
                host_inputs.contains(&binding.binding)
            );
        }
    }

    #[test]
    fn trap_sidecar_never_consumes_a_host_input_slot() {
        let slots = vec![slot(
            0,
            TRAP_SIDECAR_NAME,
            MemoryClass::Global,
            BindingVisibility::ReadWrite,
        )];
        let every_slot: FxHashSet<u32> = FxHashSet::from_iter([0]);
        let bindings = descriptor_buffer_bindings(&descriptor_of(slots), &every_slot, &every_slot)
            .expect("binding metadata for a trap sidecar descriptor");
        let sidecar = bindings
            .iter()
            .find(|binding| binding.internal_trap)
            .expect("the trap sidecar slot maps to a bind group");
        assert!(
            !sidecar.consumes_host_input,
            "the trap sidecar is backend-allocated and takes no caller input"
        );
        assert!(
            !sidecar.is_output,
            "the trap sidecar is not a public output"
        );
        assert!(
            !sidecar.preserve_input_contents,
            "a binding with no host input has no contents to preserve"
        );
    }

    #[test]
    fn preserved_contents_require_read_write_and_a_host_input() {
        let slots = full_grid();
        let host_inputs: FxHashSet<u32> = slots.iter().map(|slot| slot.slot).collect();
        let bindings =
            descriptor_buffer_bindings(&descriptor_of(slots), &FxHashSet::default(), &host_inputs)
                .expect("binding metadata for a well-formed descriptor");
        for binding in &bindings {
            let expected = binding.access == vyre_foundation::ir::BufferAccess::ReadWrite
                && binding.consumes_host_input;
            assert_eq!(
                binding.preserve_input_contents,
                expected,
                "binding {} preserves {} under access {:?} and host input {}",
                binding.binding,
                binding.preserve_input_contents,
                binding.access,
                binding.consumes_host_input
            );
        }
    }
}
