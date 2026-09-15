use std::collections::BTreeMap;

use metal::{Buffer, Device};
use vyre_driver::{BackendError, BindingPlan, BindingRole, OutputBindingLayout};

use super::resident::{
    copy_to_shared_buffer, metal_physical_buffer_len, new_host_input_buffer, new_zero_buffer,
    resident_bindings, zero_shared_buffer_range, ResolvedMetalResource,
};
use crate::METAL_BACKEND_ID;

pub(crate) struct PlannedBuffer {
    pub(crate) binding: u32,
    pub(crate) metal_slot: u8,
    pub(crate) buffer: Buffer,
    pub(crate) allocated_bytes: usize,
    pub(crate) host_to_device_bytes: usize,
}

pub(crate) fn output_layout_map(
    output_layouts: Vec<OutputBindingLayout>,
) -> Result<BTreeMap<u32, OutputBindingLayout>, BackendError> {
    let mut by_binding = BTreeMap::new();
    for layout in output_layouts {
        if by_binding.insert(layout.binding, layout).is_some() {
            return Err(BackendError::InvalidProgram {
                fix: "Fix: output layout planning produced duplicate output bindings; rebuild the Program with unique buffer bindings.".to_string(),
            });
        }
    }
    Ok(by_binding)
}

pub(crate) fn metal_slot_map(
    artifact: &vyre_emit_metal::MetalArtifact,
) -> Result<BTreeMap<u32, u8>, BackendError> {
    let mut slots = BTreeMap::new();
    for binding in &artifact.bindings {
        if slots
            .insert(binding.slot, binding.metal_buffer_index)
            .is_some()
        {
            return Err(BackendError::KernelCompileFailed {
                backend: METAL_BACKEND_ID.to_string(),
                compiler_message: format!(
                    "Metal artifact contains duplicate binding metadata for slot {}",
                    binding.slot
                ),
            });
        }
    }
    Ok(slots)
}

pub(crate) fn plan_buffers(
    device: &Device,
    binding_plan: &BindingPlan,
    inputs: &[&[u8]],
    output_by_binding: &BTreeMap<u32, OutputBindingLayout>,
    metal_slots: &BTreeMap<u32, u8>,
    artifact_bindings: &[vyre_emit_metal::MetalBindingMetadata],
) -> Result<Vec<PlannedBuffer>, BackendError> {
    let mut buffers = Vec::new();
    let reserve_len = binding_plan
        .bindings
        .len()
        .checked_add(artifact_bindings.len())
        .ok_or_else(|| BackendError::InvalidProgram {
            fix: "Fix: Metal buffer planning binding count overflowed usize. Split the Program bindings.".to_string(),
        })?;
    buffers.try_reserve(reserve_len).map_err(|error| {
        BackendError::InvalidProgram {
            fix: format!(
                "Fix: Metal buffer planning could not reserve {reserve_len} binding slot(s): {error}. Split the Program bindings.",
            ),
        }
    })?;

    for binding in &binding_plan.bindings {
        if binding.role == BindingRole::Shared {
            continue;
        }
        if binding.role == BindingRole::Persistent {
            return Err(BackendError::UnsupportedFeature {
                name: format!(
                    "Metal persistent buffer binding `{}` in non-resident dispatch",
                    binding.name
                ),
                backend: METAL_BACKEND_ID.to_string(),
            });
        }
        let metal_slot = metal_slots.get(&binding.binding).copied().ok_or_else(|| {
            BackendError::KernelCompileFailed {
                backend: METAL_BACKEND_ID.to_string(),
                compiler_message: format!(
                    "Metal artifact did not include ABI metadata for binding {} (`{}`)",
                    binding.binding, binding.name
                ),
            }
        })?;
        let (buffer, allocated_bytes, host_to_device_bytes) = match binding.role {
            BindingRole::Input | BindingRole::Uniform => {
                let input_index =
                    binding
                        .input_index
                        .ok_or_else(|| BackendError::InvalidProgram {
                            fix: format!(
                        "Fix: Metal binding `{}` is {:?} but has no input index in BindingPlan.",
                        binding.name, binding.role
                    ),
                        })?;
                let bytes = inputs.get(input_index).ok_or_else(|| BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: Metal input index {input_index} for binding `{}` is missing after BindingPlan validation.",
                        binding.name
                    ),
                })?;
                (
                    new_host_input_buffer(device, bytes)?,
                    metal_physical_buffer_len(bytes.len()),
                    bytes.len(),
                )
            }
            BindingRole::Output => {
                let layout = output_by_binding.get(&binding.binding).ok_or_else(|| {
                    BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: Metal output binding {} (`{}`) has no output readback layout.",
                            binding.binding, binding.name
                        ),
                    }
                })?;
                let byte_len = allocation_len_for_output(layout)?;
                (
                    new_zero_buffer(device, byte_len)?,
                    metal_physical_buffer_len(byte_len),
                    0,
                )
            }
            BindingRole::InputOutput => {
                let input_index =
                    binding
                        .input_index
                        .ok_or_else(|| BackendError::InvalidProgram {
                            fix: format!(
                        "Fix: Metal read-write binding `{}` has no input index in BindingPlan.",
                        binding.name
                    ),
                        })?;
                let bytes = inputs.get(input_index).ok_or_else(|| BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: Metal read-write input index {input_index} for binding `{}` is missing after BindingPlan validation.",
                        binding.name
                    ),
                })?;
                let output_len = output_by_binding
                    .get(&binding.binding)
                    .map(allocation_len_for_output)
                    .transpose()?
                    .unwrap_or(4);
                let byte_len = output_len.max(bytes.len()).max(4);
                let buffer = new_zero_buffer(device, byte_len)?;
                copy_to_shared_buffer(&buffer, bytes)?;
                (buffer, metal_physical_buffer_len(byte_len), bytes.len())
            }
            BindingRole::Shared | BindingRole::Persistent => {
                unreachable!(
                    "BindingRole {:?} is filtered above the plan match",
                    binding.role
                )
            }
        };
        buffers.push(PlannedBuffer {
            binding: binding.binding,
            metal_slot,
            buffer,
            allocated_bytes,
            host_to_device_bytes,
        });
    }
    for binding in artifact_bindings {
        if buffers
            .iter()
            .any(|planned| planned.binding == binding.slot)
        {
            continue;
        }
        if binding.name == vyre_lower::TRAP_SIDECAR_NAME {
            buffers.push(PlannedBuffer {
                binding: binding.slot,
                metal_slot: binding.metal_buffer_index,
                buffer: new_zero_buffer(device, trap_sidecar_byte_len()?)?,
                allocated_bytes: metal_physical_buffer_len(trap_sidecar_byte_len()?),
                host_to_device_bytes: 0,
            });
            continue;
        }
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: Metal artifact binding {} (`{}`) was not allocated by the user BindingPlan and is not a recognized backend-owned binding. Keep descriptor artifact bindings synchronized with Program buffers.",
                binding.slot, binding.name
            ),
        });
    }
    Ok(buffers)
}

pub(crate) fn plan_resident_buffers(
    device: &Device,
    binding_plan: &BindingPlan,
    resources: &[ResolvedMetalResource<'_>],
    output_by_binding: &BTreeMap<u32, OutputBindingLayout>,
    metal_slots: &BTreeMap<u32, u8>,
    artifact_bindings: &[vyre_emit_metal::MetalBindingMetadata],
) -> Result<Vec<PlannedBuffer>, BackendError> {
    let mut buffers = Vec::new();
    let reserve_len = binding_plan
        .bindings
        .len()
        .checked_add(artifact_bindings.len())
        .ok_or_else(|| BackendError::InvalidProgram {
            fix: "Fix: Metal resident buffer planning binding count overflowed usize. Split the Program bindings.".to_string(),
        })?;
    buffers
        .try_reserve(reserve_len)
        .map_err(|error| BackendError::InvalidProgram {
            fix: format!(
                "Fix: Metal resident buffer planning could not reserve {reserve_len} binding slot(s): {error}. Split the Program bindings."
            ),
        })?;

    for entry in resident_bindings(binding_plan) {
        let (resource_index, binding) = entry?;
        let metal_slot = metal_slots.get(&binding.binding).copied().ok_or_else(|| {
            BackendError::KernelCompileFailed {
                backend: METAL_BACKEND_ID.to_string(),
                compiler_message: format!(
                    "Metal artifact did not include ABI metadata for binding {} (`{}`)",
                    binding.binding, binding.name
                ),
            }
        })?;
        let resource = resources.get(resource_index).ok_or_else(|| BackendError::InvalidProgram {
            fix: format!(
                "Fix: Metal resident buffer planning missing resource slot {resource_index} for binding {} (`{}`).",
                binding.binding, binding.name
            ),
        })?;
        let (buffer, allocated_bytes, host_to_device_bytes) = match binding.role {
            BindingRole::Input | BindingRole::Uniform => {
                let (allocated_bytes, host_to_device_bytes) =
                    materialized_read_resource_metrics(resource);
                (
                    materialize_read_resource(device, resource)?,
                    allocated_bytes,
                    host_to_device_bytes,
                )
            }
            BindingRole::Output => {
                let layout = output_by_binding.get(&binding.binding).ok_or_else(|| {
                    BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: Metal resident output binding {} (`{}`) has no output readback layout.",
                            binding.binding, binding.name
                        ),
                    }
                })?;
                let required = allocation_len_for_output(layout)?;
                let allocated_bytes = materialized_output_resource_allocation(resource, required);
                (
                    materialize_output_resource(device, resource, layout, binding.binding)?,
                    allocated_bytes,
                    0,
                )
            }
            BindingRole::InputOutput => {
                let output_len = output_by_binding
                    .get(&binding.binding)
                    .map(allocation_len_for_output)
                    .transpose()?
                    .unwrap_or(4);
                let (allocated_bytes, host_to_device_bytes) =
                    materialized_read_write_resource_metrics(resource, output_len);
                (
                    materialize_read_write_resource(device, resource, output_len, binding.binding)?,
                    allocated_bytes,
                    host_to_device_bytes,
                )
            }
            BindingRole::Shared | BindingRole::Persistent => {
                unreachable!(
                    "BindingRole {:?} is filtered above the plan match",
                    binding.role
                )
            }
        };
        buffers.push(PlannedBuffer {
            binding: binding.binding,
            metal_slot,
            buffer,
            allocated_bytes,
            host_to_device_bytes,
        });
    }
    for binding in artifact_bindings {
        if buffers
            .iter()
            .any(|planned| planned.binding == binding.slot)
        {
            continue;
        }
        if binding.name == vyre_lower::TRAP_SIDECAR_NAME {
            let byte_len = trap_sidecar_byte_len()?;
            buffers.push(PlannedBuffer {
                binding: binding.slot,
                metal_slot: binding.metal_buffer_index,
                buffer: new_zero_buffer(device, byte_len)?,
                allocated_bytes: metal_physical_buffer_len(byte_len),
                host_to_device_bytes: 0,
            });
            continue;
        }
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: Metal resident artifact binding {} (`{}`) was not allocated by the user BindingPlan and is not a recognized backend-owned binding. Keep descriptor artifact bindings synchronized with Program buffers.",
                binding.slot, binding.name
            ),
        });
    }
    Ok(buffers)
}

fn materialize_read_resource(
    device: &Device,
    resource: &ResolvedMetalResource<'_>,
) -> Result<Buffer, BackendError> {
    match resource {
        ResolvedMetalResource::Borrowed(bytes) => new_host_input_buffer(device, bytes),
        ResolvedMetalResource::Resident { buffer, .. } => Ok(buffer.clone()),
    }
}

fn materialized_read_resource_metrics(resource: &ResolvedMetalResource<'_>) -> (usize, usize) {
    match resource {
        ResolvedMetalResource::Borrowed(bytes) => {
            (metal_physical_buffer_len(bytes.len()), bytes.len())
        }
        ResolvedMetalResource::Resident { .. } => (0, 0),
    }
}

fn materialize_output_resource(
    device: &Device,
    resource: &ResolvedMetalResource<'_>,
    layout: &OutputBindingLayout,
    binding: u32,
) -> Result<Buffer, BackendError> {
    let required = allocation_len_for_output(layout)?;
    match resource {
        ResolvedMetalResource::Borrowed(_) => new_zero_buffer(device, required),
        ResolvedMetalResource::Resident {
            id,
            buffer,
            byte_len,
        } => {
            if *byte_len < required {
                return Err(BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: Metal resident output binding {binding} (`{}`) requires {required} byte(s), but handle {id} has {byte_len}. Allocate a larger resident output buffer.",
                        layout.name
                    ),
                });
            }
            zero_shared_buffer_range(buffer, 0, required, "resident output clear")?;
            Ok(buffer.clone())
        }
    }
}

fn materialized_output_resource_allocation(
    resource: &ResolvedMetalResource<'_>,
    required: usize,
) -> usize {
    match resource {
        ResolvedMetalResource::Borrowed(_) => metal_physical_buffer_len(required),
        ResolvedMetalResource::Resident { .. } => 0,
    }
}

fn materialize_read_write_resource(
    device: &Device,
    resource: &ResolvedMetalResource<'_>,
    output_len: usize,
    binding: u32,
) -> Result<Buffer, BackendError> {
    match resource {
        ResolvedMetalResource::Borrowed(bytes) => {
            let byte_len = output_len.max(bytes.len()).max(4);
            let buffer = new_zero_buffer(device, byte_len)?;
            copy_to_shared_buffer(&buffer, bytes)?;
            Ok(buffer)
        }
        ResolvedMetalResource::Resident {
            id,
            buffer,
            byte_len,
        } => {
            if *byte_len < output_len {
                return Err(BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: Metal resident read-write binding {binding} requires {output_len} output byte(s), but handle {id} has {byte_len}. Allocate a larger resident buffer."
                    ),
                });
            }
            Ok(buffer.clone())
        }
    }
}

fn materialized_read_write_resource_metrics(
    resource: &ResolvedMetalResource<'_>,
    output_len: usize,
) -> (usize, usize) {
    match resource {
        ResolvedMetalResource::Borrowed(bytes) => (
            metal_physical_buffer_len(output_len.max(bytes.len()).max(4)),
            bytes.len(),
        ),
        ResolvedMetalResource::Resident { .. } => (0, 0),
    }
}

pub(crate) fn resident_input_lengths(
    binding_plan: &BindingPlan,
    resources: &[ResolvedMetalResource<'_>],
) -> Result<Vec<usize>, BackendError> {
    let mut input_lengths = vec![0usize; binding_plan.input_indices.len()];
    let mut resource_index = 0usize;
    for binding in &binding_plan.bindings {
        if binding.role == BindingRole::Shared {
            continue;
        }
        let resource = resources.get(resource_index).ok_or_else(|| BackendError::InvalidProgram {
            fix: format!(
                "Fix: Metal resident input length planning missing resource slot {resource_index} for binding {} (`{}`).",
                binding.binding, binding.name
            ),
        })?;
        if let Some(input_index) = binding.input_index {
            let input_slot_count = input_lengths.len();
            let slot = input_lengths.get_mut(input_index).ok_or_else(|| {
                BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: Metal resident binding `{}` references input index {input_index}, but BindingPlan only has {} input slot(s).",
                        binding.name,
                        input_slot_count
                    ),
                }
            })?;
            *slot = resource.byte_len();
        }
        resource_index += 1;
    }
    Ok(input_lengths)
}

pub(crate) fn allocation_len_for_output(
    layout: &OutputBindingLayout,
) -> Result<usize, BackendError> {
    layout
        .layout
        .copy_offset
        .checked_add(layout.layout.copy_size)
        .map(|required| required.max(layout.layout.full_size).max(4))
        .ok_or_else(|| BackendError::InvalidProgram {
            fix: format!(
                "Fix: output layout for `{}` overflows allocation length. Narrow output_byte_range.",
                layout.name
            ),
        })
}

fn trap_sidecar_byte_len() -> Result<usize, BackendError> {
    usize::try_from(vyre_lower::TRAP_SIDECAR_WORDS)
        .map_err(|error| BackendError::InvalidProgram {
            fix: format!(
                "Fix: trap sidecar word count cannot fit usize: {error}. Keep TRAP_SIDECAR_WORDS within the host index ABI."
            ),
        })?
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| BackendError::InvalidProgram {
            fix: "Fix: trap sidecar byte length overflowed usize. Keep TRAP_SIDECAR_WORDS within the host index ABI.".to_string(),
        })
}
