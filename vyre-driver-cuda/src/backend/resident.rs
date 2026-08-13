//! CUDA-resident buffer table and in-flight handle accounting.

use std::hash::BuildHasherDefault;
use std::sync::{
    atomic::{AtomicU64, AtomicUsize, Ordering},
    Arc,
};

use dashmap::DashMap;
use rustc_hash::{FxHashMap, FxHasher};
use smallvec::SmallVec;
use vyre_driver::accounting::{
    checked_add_u64_lazy, checked_add_usize_lazy, checked_atomic_add_u64_guarded_with_order,
    checked_atomic_add_usize_with_order, checked_atomic_next_u64_with_order,
    checked_atomic_sub_usize_with_order,
};
use vyre_driver::{BackendError, ResidentHandle, ResidentOwner};

use super::accounting::checked_sub_u64;
use super::allocations::{alloc_cuda_ptr, free_cuda_ptr};
use super::staging_reserve::{reserve_hash_map, reserve_smallvec};

#[derive(Debug)]
pub(crate) struct ResidentBuffer {
    pub(crate) ptr: u64,
    pub(crate) byte_len: usize,
}

// SAFETY: FFI to libcuda.so. Pointer args were validated by the matching alloc
// / store API; lifetimes are documented in the surrounding function.
// cuda_check (or matching CUresult guard) propagates non-success codes as
// BackendError.
unsafe impl Send for ResidentBuffer {}
// SAFETY: FFI to libcuda.so. Pointer args were validated by the matching alloc
// / store API; lifetimes are documented in the surrounding function.
// cuda_check (or matching CUresult guard) propagates non-success codes as
// BackendError.
unsafe impl Sync for ResidentBuffer {}

impl Drop for ResidentBuffer {
    fn drop(&mut self) {
        free_cuda_ptr(self.ptr);
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ResidentBufferView {
    pub(crate) ptr: u64,
    pub(crate) byte_len: usize,
}

/// Stable CUDA-resident buffer handle owned by [`crate::backend::CudaBackend`].
///
/// The handle names its owning backend instance, so presenting it to a
/// different instance is refused at the API boundary instead of resolving
/// against that instance's unrelated buffer of the same local id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CudaResidentBuffer {
    /// Owner-qualified handle for this buffer.
    pub handle: ResidentHandle,
    /// Buffer size in bytes.
    pub byte_len: usize,
}

/// One resolved binding for a CUDA resident dispatch.
///
/// Residency is chosen per binding, never per dispatch: a large immutable
/// table stays [`CudaDispatchBinding::Resident`] across many calls while the
/// small per-call buffers beside it arrive as
/// [`CudaDispatchBinding::Borrowed`] and are staged into the transient pool
/// for that one dispatch. Forcing every binding resident just because one of
/// them is would trade a saved upload for allocate/upload/free churn on all
/// the others.
#[derive(Debug, Clone, Copy)]
pub(crate) enum CudaDispatchBinding<'a> {
    /// Backend-resident buffer bound by handle. The caller uploaded it once
    /// and this dispatch neither stages nor frees it.
    Resident(CudaResidentBuffer),
    /// Host bytes staged into a transient device allocation for this dispatch
    /// only, exactly as a fully borrowed dispatch stages its inputs.
    Borrowed(&'a [u8]),
}

impl CudaDispatchBinding<'_> {
    /// Resident handle behind this binding, or `None` when it is staged from
    /// host bytes and therefore has no device identity that outlives the call.
    pub(crate) fn resident(self) -> Option<CudaResidentBuffer> {
        match self {
            Self::Resident(handle) => Some(handle),
            Self::Borrowed(_) => None,
        }
    }
}

pub(crate) type ResidentViewCache = SmallVec<[(CudaResidentBuffer, ResidentBufferView); 8]>;

#[derive(Debug)]
pub(crate) struct CudaResidentStore {
    /// Identity of the backend instance that owns every handle in this store.
    ///
    /// Local ids restart at 1 per instance, so the owner is what makes a
    /// handle meaningful outside the instance that minted it.
    owner: ResidentOwner,
    buffers: DashMap<ResidentHandle, ResidentBuffer, BuildHasherDefault<FxHasher>>,
    inflight: Arc<DashMap<ResidentHandle, AtomicUsize, BuildHasherDefault<FxHasher>>>,
    next_id: AtomicU64,
    resident_bytes: AtomicU64,
}

impl CudaResidentStore {
    pub(crate) fn new() -> Result<Self, BackendError> {
        Ok(Self {
            owner: ResidentOwner::new()?,
            buffers: DashMap::with_hasher(BuildHasherDefault::<FxHasher>::default()),
            inflight: Arc::new(DashMap::with_hasher(
                BuildHasherDefault::<FxHasher>::default(),
            )),
            next_id: AtomicU64::new(1),
            resident_bytes: AtomicU64::new(0),
        })
    }

    pub(crate) fn clear(&self) -> Result<(), BackendError> {
        let inflight = self.inflight_count()?;
        if inflight != 0 {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA cleanup found {inflight} resident buffer handle reference(s) still bound to in-flight dispatches; wait for pending work before shutdown."
                ),
            });
        }
        self.buffers.clear();
        self.inflight.clear();
        self.resident_bytes.store(0, Ordering::Release);
        Ok(())
    }

    pub(crate) fn allocate(
        &self,
        byte_len: usize,
        budget_bytes: u64,
    ) -> Result<CudaResidentBuffer, BackendError> {
        if byte_len == 0 {
            return Err(BackendError::InvalidProgram {
                fix: "Fix: CUDA resident buffers must have a non-zero byte length.".to_string(),
            });
        }
        let requested_bytes = u64::try_from(byte_len).map_err(|_| BackendError::InvalidProgram {
            fix: format!(
                "Fix: CUDA resident allocation request of {byte_len} bytes does not fit u64 accounting; shard the resident buffer."
            ),
        })?;
        reserve_resident_budget(&self.resident_bytes, requested_bytes, budget_bytes)?;
        let ptr = match alloc_cuda_ptr(byte_len, "cuMemAlloc_v2") {
            Ok(ptr) => ptr,
            Err(error) => {
                release_resident_budget_or_repair(
                    &self.resident_bytes,
                    requested_bytes,
                    "CUDA resident budget rollback after allocation failure",
                );
                return Err(error);
            }
        };
        let local_id = match allocate_resident_handle_id(&self.next_id) {
            Ok(id) => id,
            Err(error) => {
                free_cuda_ptr(ptr);
                release_resident_budget_or_repair(
                    &self.resident_bytes,
                    requested_bytes,
                    "CUDA resident budget rollback after handle-id allocation failure",
                );
                return Err(error);
            }
        };
        let handle = self.owner.handle(local_id);
        self.buffers
            .insert(handle, ResidentBuffer { ptr, byte_len });
        Ok(CudaResidentBuffer { handle, byte_len })
    }

    pub(crate) fn free(&self, handle: CudaResidentBuffer) -> Result<(), BackendError> {
        self.check_owner(handle.handle, "resident free")?;
        let in_use = self.inflight_for(handle.handle);
        if in_use != 0 {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA resident buffer handle {} is bound to {in_use} in-flight dispatch(es); wait for the pending dispatch before freeing it.",
                    handle.handle
                ),
            });
        }
        let (_, removed) =
            self.buffers
                .remove(&handle.handle)
                .ok_or_else(|| BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: CUDA resident buffer handle {} is not owned by this backend.",
                        handle.handle
                    ),
                })?;
        let removed_bytes =
            u64::try_from(removed.byte_len).map_err(|_| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA resident buffer handle {} has {} bytes, which does not fit u64 accounting on this target; recreate the backend and shard resident buffers.",
                    handle.handle, removed.byte_len
                ),
            })?;
        if release_resident_budget(&self.resident_bytes, removed_bytes).is_err() {
            self.rebuild_resident_byte_accounting()?;
        }
        self.inflight.remove(&handle.handle);
        Ok(())
    }

    /// Refuse a handle this instance did not mint.
    ///
    /// Keying the buffer table by the owner-qualified handle already makes a
    /// foreign handle miss every lookup, so this exists to name the actual
    /// cause: "unknown or already freed handle" and "handle belonging to a
    /// different backend instance" need different repairs, and silently
    /// reporting the first for the second is how a stale handle survives
    /// review.
    fn check_owner(&self, handle: ResidentHandle, context: &str) -> Result<(), BackendError> {
        self.owner.resolve(handle, context).map(drop)
    }

    pub(crate) fn allocated_bytes(&self) -> u64 {
        self.resident_bytes.load(Ordering::Acquire)
    }

    pub(crate) fn view(
        &self,
        handle: CudaResidentBuffer,
    ) -> Result<ResidentBufferView, BackendError> {
        self.check_owner(handle.handle, "resident buffer view")?;
        let buffer =
            self.buffers
                .get(&handle.handle)
                .ok_or_else(|| BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: CUDA resident buffer handle {} is not owned by this backend.",
                        handle.handle
                    ),
                })?;
        if buffer.byte_len != handle.byte_len {
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA resident buffer handle {} byte length drifted from {} to {}.",
                    handle.handle, handle.byte_len, buffer.byte_len
                ),
            });
        }
        Ok(ResidentBufferView {
            ptr: buffer.ptr,
            byte_len: buffer.byte_len,
        })
    }

    pub(crate) fn view_cached(
        &self,
        handle: CudaResidentBuffer,
        cache: &mut ResidentViewCache,
        context: &'static str,
    ) -> Result<ResidentBufferView, BackendError> {
        for &(cached_handle, cached_view) in cache.iter() {
            if cached_handle.handle != handle.handle {
                continue;
            }
            if cached_handle.byte_len != handle.byte_len {
                return Err(BackendError::InvalidProgram {
                    fix: format!(
                        "Fix: CUDA {context} received resident handle {} with inconsistent byte lengths {} and {}; rebuild the resident handle list from the backend store before dispatch.",
                        handle.handle, cached_handle.byte_len, handle.byte_len
                    ),
                });
            }
            return Ok(cached_view);
        }
        let view = self.view(handle)?;
        reserve_smallvec(cache, 1, context)?;
        cache.push((handle, view));
        Ok(view)
    }

    pub(crate) fn mark_inflight(
        &self,
        handles: &[CudaResidentBuffer],
    ) -> Result<ResidentUseGuard, BackendError> {
        let mut guard = ResidentUseGuard {
            inflight: Arc::clone(&self.inflight),
            ids: SmallVec::new(),
        };
        if handles.is_empty() {
            return Ok(guard);
        }
        reserve_smallvec(
            &mut guard.ids,
            handles.len(),
            "resident in-flight guard ids",
        )?;
        if handles.len() <= 8 {
            let mut seen = SmallVec::<[(ResidentHandle, usize); 8]>::new();
            'mark_small: for handle in handles {
                for (seen_handle, seen_byte_len) in &seen {
                    if *seen_handle == handle.handle {
                        if *seen_byte_len != handle.byte_len {
                            return Err(BackendError::InvalidProgram {
                                fix: format!(
                                    "Fix: CUDA resident buffer handle {} byte length drifted from {} to {} during in-flight marking.",
                                    handle.handle, seen_byte_len, handle.byte_len
                                ),
                            });
                        }
                        continue 'mark_small;
                    }
                }
                seen.push((handle.handle, handle.byte_len));
                self.mark_unique_inflight_handle(*handle, &mut guard)?;
            }
            return Ok(guard);
        }

        let mut seen = FxHashMap::default();
        reserve_hash_map(&mut seen, handles.len(), "resident duplicate check")?;
        for handle in handles {
            if let Some(&seen_byte_len) = seen.get(&handle.handle) {
                if seen_byte_len != handle.byte_len {
                    return Err(BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: CUDA resident buffer handle {} byte length drifted from {} to {} during in-flight marking.",
                            handle.handle, seen_byte_len, handle.byte_len
                        ),
                    });
                }
                continue;
            }
            seen.insert(handle.handle, handle.byte_len);
            self.mark_unique_inflight_handle(*handle, &mut guard)?;
        }
        Ok(guard)
    }

    fn mark_unique_inflight_handle(
        &self,
        handle: CudaResidentBuffer,
        guard: &mut ResidentUseGuard,
    ) -> Result<(), BackendError> {
        self.view(handle)?;
        let counter = self
            .inflight
            .entry(handle.handle)
            .or_insert_with(|| AtomicUsize::new(0));
        checked_atomic_add_usize_with_order(
            &counter,
            1,
            Ordering::Acquire,
            Ordering::AcqRel,
            Ordering::Acquire,
            |value, _| {
                BackendError::InvalidProgram {
            fix: format!(
                "Fix: CUDA resident in-flight reference count overflowed for handle {id} at {value}; wait for pending dispatches before rebinding this resident buffer.",
                id = handle.handle
            ),
            }
            },
        )?;
        guard.ids.push(handle.handle);
        Ok(())
    }

    pub(crate) fn handles_from_resources(
        &self,
        resources: &[vyre_driver::Resource],
    ) -> Result<SmallVec<[CudaResidentBuffer; 8]>, BackendError> {
        let mut handles = SmallVec::new();
        reserve_smallvec(&mut handles, resources.len(), "resident resource handles")?;
        for resource in resources {
            handles.push(self.handle_from_resource(resource)?);
        }
        Ok(handles)
    }

    /// Resolve a dispatch resource list into per-binding sources, keeping
    /// resident and borrowed entries side by side in caller order.
    pub(crate) fn bindings_from_resources<'a>(
        &self,
        resources: &'a [vyre_driver::Resource],
    ) -> Result<SmallVec<[CudaDispatchBinding<'a>; 8]>, BackendError> {
        let mut bindings = SmallVec::new();
        reserve_smallvec(&mut bindings, resources.len(), "resident dispatch bindings")?;
        for resource in resources {
            bindings.push(self.binding_from_resource(resource)?);
        }
        Ok(bindings)
    }

    /// Resolve one dispatch resource into its binding source.
    ///
    /// A [`vyre_driver::Resource::Borrowed`] is not an error here: the resident
    /// dispatch stages it per call. Only lookups that must name device memory
    /// that outlives the call (upload, download, free) go through
    /// [`CudaResidentStore::handle_from_resource`].
    pub(crate) fn binding_from_resource<'a>(
        &self,
        resource: &'a vyre_driver::Resource,
    ) -> Result<CudaDispatchBinding<'a>, BackendError> {
        match resource {
            vyre_driver::Resource::Resident(_) => Ok(CudaDispatchBinding::Resident(
                self.handle_from_resource(resource)?,
            )),
            vyre_driver::Resource::Borrowed(bytes) => {
                Ok(CudaDispatchBinding::Borrowed(bytes.as_slice()))
            }
        }
    }

    pub(crate) fn handle_from_resource(
        &self,
        resource: &vyre_driver::Resource,
    ) -> Result<CudaResidentBuffer, BackendError> {
        match resource {
            vyre_driver::Resource::Resident(handle) => {
                self.check_owner(*handle, "resident handle lookup")?;
                let buffer = self
                    .buffers
                    .get(handle)
                    .ok_or_else(|| BackendError::InvalidProgram {
                        fix: format!(
                            "Fix: CUDA compiled resident dispatch received unknown resident handle {handle}; it was never allocated on this backend or has already been freed."
                        ),
                    })?;
                Ok(CudaResidentBuffer {
                    handle: *handle,
                    byte_len: buffer.byte_len,
                })
            }
            vyre_driver::Resource::Borrowed(_) => Err(BackendError::InvalidProgram {
                fix: "Fix: CUDA resident upload, download, and free name device memory that outlives the call, so they need a Resource::Resident handle; a Resource::Borrowed value has no device identity. Pass a borrowed buffer straight to the dispatch instead, which stages it per call alongside the resident bindings."
                    .to_string(),
            }),
        }
    }

    fn inflight_for(&self, id: ResidentHandle) -> usize {
        match self.inflight.get(&id) {
            Some(count) => count.load(Ordering::Acquire),
            None => 0,
        }
    }

    fn rebuild_resident_byte_accounting(&self) -> Result<(), BackendError> {
        let mut total = 0u64;
        for entry in self.buffers.iter() {
            let bytes = u64::try_from(entry.byte_len).map_err(|_| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA resident buffer handle {} has {} bytes, which does not fit u64 while rebuilding resident byte accounting; recreate the backend and shard resident buffers.",
                    entry.key(),
                    entry.byte_len
                ),
            })?;
            total = checked_add_u64_lazy(total, bytes, || {
                BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA resident byte accounting overflowed while rebuilding from live handle {} with {bytes} bytes; shard the resident set.",
                    entry.key()
                ),
            }
            })?;
        }
        self.resident_bytes.store(total, Ordering::Release);
        Ok(())
    }

    fn inflight_count(&self) -> Result<usize, BackendError> {
        let mut total = 0usize;
        for entry in self.inflight.iter() {
            let count = entry.value().load(Ordering::Acquire);
            total = checked_add_usize_lazy(total, count, || {
                BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA resident in-flight reference count overflowed while summing handle {} with {count} reference(s). Wait for pending work and repair resident dispatch lifetime accounting; never continue with saturated in-flight state.",
                    entry.key()
                ),
            }
            })?;
        }
        Ok(total)
    }
}

/// Lift an all-resident handle list into dispatch bindings.
///
/// Used by call sites that already resolved resident handles for their own
/// bookkeeping and still need to drive the mixed dispatch core.
pub(crate) fn resident_bindings_from_handles(
    handles: &[CudaResidentBuffer],
) -> Result<SmallVec<[CudaDispatchBinding<'static>; 8]>, BackendError> {
    let mut bindings = SmallVec::new();
    reserve_smallvec(&mut bindings, handles.len(), "resident dispatch bindings")?;
    bindings.extend(handles.iter().copied().map(CudaDispatchBinding::Resident));
    Ok(bindings)
}

fn allocate_resident_handle_id(next_id: &AtomicU64) -> Result<u64, BackendError> {
    checked_atomic_next_u64_with_order(
        next_id,
        Ordering::Acquire,
        Ordering::AcqRel,
        Ordering::Acquire,
        |_| {
            BackendError::InvalidProgram {
            fix: "Fix: CUDA resident buffer handle id space is exhausted before allocation; recreate the backend session instead of wrapping handle ids.".to_string(),
        }
        },
    )
}

fn reserve_resident_budget(
    resident_bytes: &AtomicU64,
    requested_bytes: u64,
    budget_bytes: u64,
) -> Result<(), BackendError> {
    checked_atomic_add_u64_guarded_with_order(
        resident_bytes,
        requested_bytes,
        Ordering::Acquire,
        Ordering::AcqRel,
        Ordering::Acquire,
        |observed, requested| {
            BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA resident allocation accounting overflowed while adding {requested} bytes to {observed} resident bytes; shard the resident set."
                ),
            }
        },
        |next| validate_resident_allocation_budget(next, budget_bytes),
    )
}

fn release_resident_budget(
    resident_bytes: &AtomicU64,
    released_bytes: u64,
) -> Result<(), BackendError> {
    checked_sub_u64(resident_bytes, released_bytes, |observed, released| {
        BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA resident allocation accounting underflowed while releasing {released} bytes from {observed} resident bytes; recreate the backend because resident byte accounting is inconsistent."
                ),
            }
    })
}

fn release_resident_budget_or_repair(
    resident_bytes: &AtomicU64,
    released_bytes: u64,
    label: &'static str,
) {
    if let Err(error) = release_resident_budget(resident_bytes, released_bytes) {
        tracing::error!("{label}: {error}. Resident byte accounting was repaired to zero.");
        resident_bytes.store(0, Ordering::Release);
    }
}

pub(crate) fn validate_resident_allocation_budget(
    required_bytes: u64,
    budget_bytes: u64,
) -> Result<(), BackendError> {
    if required_bytes > budget_bytes {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: CUDA resident buffers would require {required_bytes} bytes but the live-device resident budget is {budget_bytes} bytes. Free unused resident handles, shard the resident set, compact outputs, or raise the CUDA resident memory budget deliberately."
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        validate_resident_allocation_budget, CudaResidentBuffer, CudaResidentStore, ResidentBuffer,
        ResidentViewCache,
    };
    use vyre_driver::BackendError;

    #[test]
    fn resident_budget_validation_rejects_cumulative_over_budget_allocation() {
        let error = validate_resident_allocation_budget(1025, 1024)
            .expect_err("resident allocation must fail before CUDA allocation");

        match error {
            BackendError::InvalidProgram { fix } => {
                assert!(fix.contains("CUDA resident buffers would require 1025 bytes"));
                assert!(fix.contains("resident budget is 1024 bytes"));
                assert!(fix.contains("Free unused resident handles"));
            }
            other => panic!("expected InvalidProgram, got {other:?}"),
        }
    }

    #[test]
    fn resident_view_cache_reuses_validated_handle_metadata_and_rejects_drift() {
        let store = CudaResidentStore::new().expect("Fix: owner ids must be available");
        let owned = store.owner.handle(7);
        store.buffers.insert(
            owned,
            ResidentBuffer {
                ptr: 0x1000,
                byte_len: 64,
            },
        );
        let mut cache = ResidentViewCache::new();
        let handle = CudaResidentBuffer {
            handle: owned,
            byte_len: 64,
        };

        let first = store
            .view_cached(handle, &mut cache, "resident view cache test")
            .expect("Fix: replace expect with fallible API or document caller precondition; panic only on programmer error - resident view cache must resolve a live handle");
        assert_eq!(first.ptr, 0x1000);
        assert_eq!(first.byte_len, 64);

        let drifted = store
            .view_cached(
                CudaResidentBuffer {
                    handle: owned,
                    byte_len: 32,
                },
                &mut cache,
                "resident view cache test",
            )
            .expect_err("cached resident handle metadata drift must be rejected");
        match drifted {
            BackendError::InvalidProgram { fix } => {
                assert!(fix.contains("resident handle 7"));
                assert!(fix.contains("inconsistent byte lengths 64 and 32"));
            }
            other => panic!("expected InvalidProgram, got {other:?}"),
        }
    }
}

/// Reference-count guard for resident buffers currently bound to async work.
#[derive(Debug)]
pub(crate) struct ResidentUseGuard {
    inflight: Arc<DashMap<ResidentHandle, AtomicUsize, BuildHasherDefault<FxHasher>>>,
    ids: SmallVec<[ResidentHandle; 8]>,
}

impl Drop for ResidentUseGuard {
    fn drop(&mut self) {
        for id in &self.ids {
            let should_remove = if let Some(count) = self.inflight.get(id) {
                match checked_atomic_sub_usize_with_order(
                    &count,
                    1,
                    Ordering::Acquire,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                    |value, _| value,
                ) {
                    Ok(()) => count.load(Ordering::Acquire) == 0,
                    Err(value) => {
                        tracing::error!(
                            "Fix: CUDA resident in-flight reference count underflowed for handle {id} at {value}; resident dispatch lifetime accounting is corrupt."
                        );
                        false
                    }
                }
            } else {
                false
            };
            if should_remove {
                self.inflight
                    .remove_if(id, |_, count| count.load(Ordering::Acquire) == 0);
            }
        }
    }
}
