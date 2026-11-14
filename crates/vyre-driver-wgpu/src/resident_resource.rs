//! Resident resource lifecycle helpers for WGPU backend buffers.

use crate::WgpuBackend;

/// Allocate one backend-resident WGPU buffer.
pub(crate) fn allocate_resident(
    backend: &WgpuBackend,
    byte_len: usize,
) -> Result<vyre_runtime::Resource, vyre_runtime::BackendError> {
    let device_queue = backend.current_device_queue();
    let len = u64::try_from(byte_len).map_err(|source| {
        vyre_runtime::BackendError::new(format!(
            "WGPU resident allocation byte length {byte_len} cannot fit u64: {source}. Fix: shard the resident buffer before allocation."
        ))
    })?;
    let handle = crate::buffer::GpuBufferHandle::alloc(
        &device_queue.0,
        len,
        crate::pipeline::binding::RESIDENT_STORAGE_USAGE,
    )?;
    let id = handle.resident_handle()?;
    backend.resident_handles.insert(id, handle);
    Ok(vyre_runtime::Resource::Resident(id))
}

/// Free one backend-resident WGPU buffer.
pub(crate) fn free_resident(
    backend: &WgpuBackend,
    resource: vyre_runtime::Resource,
) -> Result<(), vyre_runtime::BackendError> {
    let vyre_runtime::Resource::Resident(id) = resource else {
        return Err(vyre_runtime::BackendError::new(
            "WGPU resident free received a borrowed resource. Fix: only free handles returned by allocate_resident.",
        ));
    };
    crate::buffer::check_resident_owner(id, "WGPU resident free")?;
    backend.resident_handles.remove(&id).ok_or_else(|| {
        vyre_runtime::BackendError::new(format!(
            "WGPU resident free received stale handle {id}. Fix: free each resident resource exactly once."
        ))
    })?;
    Ok(())
}
