use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use metal::{Buffer, Device, MTLResourceOptions, NSUInteger};
use vyre_driver::resident_transfer_fusion::{ResidentTransferInterval, ResidentTransferView};
use vyre_driver::{
    BackendError, Binding, BindingPlan, BindingRole, ResidentHandle, ResidentOwner, Resource,
};

use crate::METAL_BACKEND_ID;

#[derive(Clone)]
pub(crate) struct MetalResidentBuffer {
    pub(super) buffer: Buffer,
    pub(crate) byte_len: usize,
}

pub(crate) type MetalResidentBufferTable = Arc<Mutex<HashMap<ResidentHandle, MetalResidentBuffer>>>;

pub(super) enum ResolvedMetalResource<'a> {
    Borrowed(&'a [u8]),
    Resident {
        /// Owner-carrying handle, kept so diagnostics never name a bare id.
        #[allow(dead_code)]
        id: ResidentHandle,
        buffer: Buffer,
        byte_len: usize,
    },
}

impl ResolvedMetalResource<'_> {
    pub(super) fn byte_len(&self) -> usize {
        match self {
            Self::Borrowed(bytes) => bytes.len(),
            Self::Resident { byte_len, .. } => *byte_len,
        }
    }
}

pub(super) fn lock_resident_buffer_table<'a>(
    resident_buffers: &'a MetalResidentBufferTable,
    operation: &'static str,
) -> Result<MutexGuard<'a, HashMap<ResidentHandle, MetalResidentBuffer>>, BackendError> {
    resident_buffers
        .lock()
        .map_err(|error| BackendError::InvalidProgram {
            fix: format!(
                "Fix: Metal resident buffer table was poisoned during {operation}: {error}. Drop and reacquire the Metal backend before reusing resident resources."
            ),
        })
}

pub(super) fn resolve_resident_resources_from_table<'a>(
    resident_owner: ResidentOwner,
    resident_buffers: &MetalResidentBufferTable,
    binding_plan: &BindingPlan,
    resources: &'a [Resource],
) -> Result<Vec<ResolvedMetalResource<'a>>, BackendError> {
    let expected = resident_resource_count(binding_plan)?;
    if resources.len() != expected {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: Metal resident dispatch expected {expected} resource(s) in binding order but received {}.",
                resources.len()
            ),
        });
    }
    let table =
        lock_resident_buffer_table(resident_buffers, "resident dispatch resource resolution")?;
    let mut resolved = Vec::new();
    resolved
        .try_reserve(expected)
        .map_err(|error| BackendError::InvalidProgram {
            fix: format!(
                "Fix: Metal resident dispatch could not reserve {expected} resolved resource descriptor(s): {error}. Split the dispatch bindings."
            ),
        })?;
    for entry in resident_bindings(binding_plan) {
        let (resource_index, binding) = entry?;
        let resource = resources.get(resource_index).ok_or_else(|| BackendError::InvalidProgram {
            fix: format!(
                "Fix: Metal resident dispatch missing resource slot {resource_index} for binding {} (`{}`).",
                binding.binding, binding.name
            ),
        })?;
        match resource {
            Resource::Borrowed(bytes) => resolved.push(ResolvedMetalResource::Borrowed(bytes)),
            Resource::Resident(id) => {
                resident_owner.resolve(*id, "resident dispatch resource resolution")?;
                let resident = table.get(id).cloned().ok_or_else(|| {
                    BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: Metal resident dispatch received stale handle {id} for binding {} (`{}`). Keep resident resources allocated until dispatch completes.",
                            binding.binding, binding.name
                        ),
                    }
                })?;
                resolved.push(ResolvedMetalResource::Resident {
                    id: *id,
                    buffer: resident.buffer,
                    byte_len: resident.byte_len,
                });
            }
        }
    }
    Ok(resolved)
}

/// Bindings a resident dispatch supplies a caller resource for, paired with
/// the resource index each one reads, in binding order.
///
/// A shared binding is workgroup-local and consumes no caller resource, so it
/// takes no index. A persistent binding has no resident dispatch meaning on
/// Metal, so the walk refuses it instead of binding the next resource to the
/// wrong slot.
pub(super) fn resident_bindings(
    binding_plan: &BindingPlan,
) -> impl Iterator<Item = Result<(usize, &Binding), BackendError>> {
    binding_plan
        .bindings
        .iter()
        .filter(|binding| binding.role != BindingRole::Shared)
        .enumerate()
        .map(|(resource_index, binding)| {
            if binding.role == BindingRole::Persistent {
                return Err(BackendError::UnsupportedFeature {
                    name: format!(
                        "Metal persistent buffer binding `{}` in resident dispatch",
                        binding.name
                    ),
                    backend: METAL_BACKEND_ID.to_string(),
                });
            }
            Ok((resource_index, binding))
        })
}

pub(super) fn resident_resource_count(binding_plan: &BindingPlan) -> Result<usize, BackendError> {
    binding_plan
        .bindings
        .iter()
        .try_fold(0usize, |count, binding| {
            if binding.role == BindingRole::Shared {
                return Ok(count);
            }
            count.checked_add(1).ok_or_else(|| BackendError::InvalidProgram {
                fix: "Fix: Metal resident resource count overflowed usize. Split the Program bindings.".to_string(),
            })
        })
}

pub(super) fn next_resident_id(counter: &AtomicU64) -> Result<u64, BackendError> {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1).filter(|next| *next != 0)
        })
        .map_err(|current| BackendError::InvalidProgram {
            fix: format!(
                "Fix: Metal resident handle id counter exhausted at {current}. Drop and reacquire the backend before allocating more resident buffers."
            ),
        })
}

pub(super) fn validate_resident_range(
    handle_id: ResidentHandle,
    allocation_len: usize,
    byte_offset: usize,
    byte_len: usize,
    context: &'static str,
) -> Result<(), BackendError> {
    let end = byte_offset
        .checked_add(byte_len)
        .ok_or_else(|| BackendError::InvalidProgram {
            fix: format!(
                "Fix: Metal {context} overflows usize at offset {byte_offset} len {byte_len} for handle {handle_id}. Split the resident range."
            ),
        })?;
    if end > allocation_len {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: Metal {context} requested byte range [{byte_offset}..{end}) from allocation {allocation_len} on handle {handle_id}. Clamp the range or allocate a larger resident buffer."
            ),
        });
    }
    Ok(())
}

pub(super) fn copy_to_shared_buffer_range(
    buffer: &Buffer,
    dst_offset_bytes: usize,
    bytes: &[u8],
    context: &'static str,
) -> Result<(), BackendError> {
    if bytes.is_empty() {
        return Ok(());
    }
    let capacity = metal_buffer_len(buffer, context)?;
    let end = dst_offset_bytes
        .checked_add(bytes.len())
        .ok_or_else(|| BackendError::InvalidProgram {
            fix: format!(
                "Fix: Metal {context} upload range overflows usize at offset {dst_offset_bytes} len {}.",
                bytes.len()
            ),
        })?;
    if end > capacity {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: Metal {context} upload range [{dst_offset_bytes}..{end}) exceeds physical buffer length {capacity}. Rebuild the resident allocation."
            ),
        });
    }
    // SAFETY: `capacity` was read from Metal, the checked range is in bounds,
    // and the source/destination ranges do not overlap.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            buffer.contents().cast::<u8>().add(dst_offset_bytes),
            bytes.len(),
        );
    }
    buffer.did_modify_range(metal::NSRange::new(
        checked_ns_uint(
            dst_offset_bytes,
            "Metal resident upload modified range offset",
        )?,
        checked_ns_uint(bytes.len(), "Metal resident upload modified range length")?,
    ));
    Ok(())
}

pub(super) fn zero_shared_buffer_range(
    buffer: &Buffer,
    dst_offset_bytes: usize,
    byte_len: usize,
    context: &'static str,
) -> Result<(), BackendError> {
    if byte_len == 0 {
        return Ok(());
    }
    let capacity = metal_buffer_len(buffer, context)?;
    let end = dst_offset_bytes
        .checked_add(byte_len)
        .ok_or_else(|| BackendError::InvalidProgram {
            fix: format!(
                "Fix: Metal {context} zero range overflows usize at offset {dst_offset_bytes} len {byte_len}."
            ),
        })?;
    if end > capacity {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: Metal {context} zero range [{dst_offset_bytes}..{end}) exceeds physical buffer length {capacity}. Rebuild the resident allocation."
            ),
        });
    }
    // SAFETY: the checked range is in bounds for a live StorageModeShared
    // buffer allocated by this backend.
    unsafe {
        std::ptr::write_bytes(
            buffer.contents().cast::<u8>().add(dst_offset_bytes),
            0,
            byte_len,
        );
    }
    buffer.did_modify_range(metal::NSRange::new(
        checked_ns_uint(
            dst_offset_bytes,
            "Metal resident zero modified range offset",
        )?,
        checked_ns_uint(byte_len, "Metal resident zero modified range length")?,
    ));
    Ok(())
}

pub(super) fn copy_shared_buffer_range_into(
    buffer: &Buffer,
    byte_offset: usize,
    byte_len: usize,
    out: &mut Vec<u8>,
    context: &'static str,
) -> Result<(), BackendError> {
    let capacity = metal_buffer_len(buffer, context)?;
    let end = byte_offset
        .checked_add(byte_len)
        .ok_or_else(|| BackendError::InvalidProgram {
            fix: format!(
                "Fix: Metal {context} read range overflows usize at offset {byte_offset} len {byte_len}."
            ),
        })?;
    if end > capacity {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: Metal {context} read range [{byte_offset}..{end}) exceeds physical buffer length {capacity}. Rebuild the resident allocation."
            ),
        });
    }
    let additional = byte_len.saturating_sub(out.capacity());
    if additional != 0 {
        out.try_reserve_exact(additional)
            .map_err(|error| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: Metal {context} could not reserve {byte_len} output byte(s): {error}. Split the resident readback."
                ),
            })?;
    }
    out.clear();
    if byte_len == 0 {
        return Ok(());
    }
    // SAFETY: the checked range is in bounds for a live StorageModeShared
    // buffer allocated by this backend.
    let source = unsafe {
        std::slice::from_raw_parts(buffer.contents().cast::<u8>().add(byte_offset), byte_len)
    };
    out.extend_from_slice(source);
    Ok(())
}

pub(super) fn reserve_fused_resident_view_outputs(
    copies: &[ResidentTransferInterval],
    views: &[ResidentTransferView],
    outputs: &mut [&mut Vec<u8>],
) -> Result<(), BackendError> {
    if views.len() != outputs.len() {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: Metal resident ranged batch download fused {} output view(s) for {} requested output slot(s). Keep resident transfer fusion cardinality-preserving before materializing outputs.",
                views.len(),
                outputs.len()
            ),
        });
    }
    for (view_index, (view, output)) in views.iter().copied().zip(outputs.iter_mut()).enumerate() {
        if view.byte_len != 0 {
            let copy = copies.get(view.copy_slot).ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: Metal resident ranged batch download view {view_index} references missing fused copy slot {}. Rebuild the resident transfer fusion plan before materializing outputs.",
                    view.copy_slot
                ),
            })?;
            let view_end =
                view.byte_offset
                    .checked_add(view.byte_len)
                    .ok_or_else(|| BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: Metal resident ranged batch download view {view_index} overflows usize at offset {} len {}. Rebuild the resident transfer fusion plan before materializing outputs.",
                            view.byte_offset, view.byte_len
                        ),
                    })?;
            if view_end > copy.byte_len {
                return Err(BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: Metal resident ranged batch download view {view_index} requested bytes [{}..{}) from a {} byte fused output. Rebuild the resident transfer fusion plan before materializing outputs.",
                        view.byte_offset,
                        view_end,
                        copy.byte_len
                    ),
                });
            }
        }
        // This pre-pass only grows the destination; `copy_fused_resident_view_into`
        // relies on the retained length for its equal-length fast path, so the
        // contents-preserving owner is required here.
        vyre_foundation::allocation::try_reserve_vec_to_capacity(output, view.byte_len).map_err(
            |error| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: Metal resident ranged batch download could not reserve {} output byte(s) for view {view_index}: {error}. Split the resident readback batch before materializing outputs.",
                    view.byte_len
                ),
            },
        )?;
    }
    Ok(())
}

pub(super) fn copy_fused_resident_view_into(
    fused_outputs: &[Vec<u8>],
    view: ResidentTransferView,
    output: &mut Vec<u8>,
) -> Result<(), BackendError> {
    if view.byte_len == 0 {
        output.clear();
        return Ok(());
    }
    let fused_output = fused_outputs
        .get(view.copy_slot)
        .ok_or_else(|| BackendError::InvalidProgram {
            fix: format!(
                "Fix: Metal resident ranged batch download view references missing fused copy slot {}. Rebuild the resident transfer fusion plan before materializing outputs.",
                view.copy_slot
            ),
        })?;
    let view_end =
        view.byte_offset
            .checked_add(view.byte_len)
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: Metal resident ranged batch download view overflows usize at offset {} len {}. Rebuild the resident transfer fusion plan before materializing outputs.",
                    view.byte_offset, view.byte_len
                ),
            })?;
    let bytes =
        fused_output
            .get(view.byte_offset..view_end)
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: Metal resident ranged batch download view requested bytes [{}..{}) from a {} byte fused output. Rebuild the resident transfer fusion plan before materializing outputs.",
                    view.byte_offset,
                    view_end,
                    fused_output.len()
                ),
            })?;
    if output.len() == bytes.len() {
        output.copy_from_slice(bytes);
        return Ok(());
    }
    vyre_foundation::allocation::reserve_exact_cleared(output, bytes.len()).map_err(|error| {
        BackendError::InvalidProgram {
            fix: format!(
                "Fix: Metal resident ranged batch download could not reserve {} output byte(s): {error}. Split the resident readback batch before materializing outputs.",
                bytes.len()
            ),
        }
    })?;
    output.extend_from_slice(bytes);
    Ok(())
}

pub(super) fn metal_buffer_len(
    buffer: &Buffer,
    context: &'static str,
) -> Result<usize, BackendError> {
    usize::try_from(buffer.length()).map_err(|error| BackendError::InvalidProgram {
        fix: format!(
            "Fix: Metal {context} buffer length cannot fit usize: {error}. Split the resident buffer."
        ),
    })
}

pub(super) fn new_host_input_buffer(device: &Device, bytes: &[u8]) -> Result<Buffer, BackendError> {
    if bytes.is_empty() {
        return new_zero_buffer(device, 4);
    }
    Ok(device.new_buffer_with_data(
        bytes.as_ptr().cast::<c_void>(),
        checked_ns_uint(bytes.len(), "Metal input buffer length")?,
        MTLResourceOptions::StorageModeShared,
    ))
}

pub(super) fn new_zero_buffer(device: &Device, byte_len: usize) -> Result<Buffer, BackendError> {
    Ok(device.new_buffer(
        checked_ns_uint(byte_len.max(4), "Metal zero buffer length")?,
        MTLResourceOptions::StorageModeShared,
    ))
}

pub(super) fn copy_to_shared_buffer(buffer: &Buffer, bytes: &[u8]) -> Result<(), BackendError> {
    if bytes.is_empty() {
        return Ok(());
    }
    let capacity = usize::try_from(buffer.length()).map_err(|error| BackendError::InvalidProgram {
        fix: format!(
            "Fix: Metal buffer length cannot fit usize during upload: {error}. Split the dispatch buffer."
        ),
    })?;
    if bytes.len() > capacity {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: Metal upload length {} exceeds allocated buffer length {capacity}. Rebuild the BindingPlan before dispatch.",
                bytes.len()
            ),
        });
    }
    // SAFETY: the buffer was allocated by this backend with StorageModeShared,
    // `capacity` was read from Metal, and the copy length is checked not to
    // exceed the allocation.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer.contents().cast::<u8>(), bytes.len());
    }
    buffer.did_modify_range(metal::NSRange::new(
        0,
        checked_ns_uint(bytes.len(), "Metal upload modified range")?,
    ));
    Ok(())
}

pub(super) fn checked_ns_uint(
    value: usize,
    field: &'static str,
) -> Result<NSUInteger, BackendError> {
    value.try_into().map_err(|error| BackendError::InvalidProgram {
        fix: format!(
            "Fix: {field} value {value} cannot fit Metal NSUInteger: {error}. Split the dispatch buffer."
        ),
    })
}

pub(super) fn ns_uint_to_u32_saturating(value: NSUInteger) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

pub(super) fn metal_physical_buffer_len(byte_len: usize) -> usize {
    byte_len.max(4)
}
