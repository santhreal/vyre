use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Weak};

use dashmap::DashMap;
use vyre_driver::{BackendError, ResidentHandle, ResidentOwner};

use super::{GpuBufferHandle, GpuBufferInner};

static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(1);
static RESIDENT_BUFFERS: LazyLock<DashMap<u64, Weak<GpuBufferInner>>> = LazyLock::new(DashMap::new);

pub(super) fn resident_buffers() -> &'static DashMap<u64, Weak<GpuBufferInner>> {
    &RESIDENT_BUFFERS
}

/// Identity of the WGPU driver's resident-buffer namespace.
///
/// `NEXT_BUFFER_ID` and `RESIDENT_BUFFERS` are process-wide rather than per
/// backend instance, so every live WGPU resident buffer shares one namespace
/// and therefore one owner. Minting the owner here, next to the registry it
/// describes, keeps that structural: a WGPU resident handle stays valid for as
/// long as its buffer lives, and a handle minted by any other driver is
/// refused instead of being resolved against an unrelated buffer of the same
/// id.
static RESIDENT_OWNER: LazyLock<Result<ResidentOwner, BackendError>> =
    LazyLock::new(ResidentOwner::new);

/// Owner of every WGPU resident handle in this process.
pub(super) fn resident_owner() -> Result<ResidentOwner, BackendError> {
    match &*RESIDENT_OWNER {
        Ok(owner) => Ok(*owner),
        Err(error) => Err(BackendError::new(format!(
            "WGPU resident buffers have no namespace identity: {error} Fix: reduce the number of backend instances created in this process."
        ))),
    }
}

/// Refuse a resident handle minted outside the WGPU resident namespace.
///
/// Resolving one anyway would look up a foreign id in this driver's registry,
/// where the same number names an unrelated live buffer.
pub(crate) fn check_resident_owner(
    handle: ResidentHandle,
    context: &str,
) -> Result<(), BackendError> {
    resident_owner()?.resolve(handle, context)?;
    Ok(())
}

pub(super) fn next_buffer_id() -> u64 {
    NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed)
}

pub(super) fn register_resident_buffer(id: u64, inner: &Arc<GpuBufferInner>) {
    resident_buffers().insert(id, Arc::downgrade(inner));
}

pub(super) fn remove_resident_buffer(id: u64) {
    resident_buffers().remove(&id);
}

pub(super) fn resolve_resident_id(id: u64) -> Option<GpuBufferHandle> {
    let registry = resident_buffers();
    let entry = registry.get(&id)?;
    let upgraded = entry.value().upgrade();
    drop(entry);
    match upgraded {
        Some(inner) => Some(GpuBufferHandle { inner }),
        None => {
            registry.remove(&id);
            None
        }
    }
}
