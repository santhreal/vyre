use std::hash::BuildHasherDefault;
use std::sync::{
    atomic::{AtomicU64, AtomicUsize, Ordering},
    Arc,
};

use crossbeam_queue::ArrayQueue;
use cudarc::driver::sys::CUresult;
use dashmap::DashMap;
use rustc_hash::FxHasher;
use smallvec::SmallVec;
use vyre_driver::accounting::{
    checked_add_usize_lazy, checked_atomic_add_usize_guarded_with_order,
    checked_atomic_add_usize_with_order, checked_atomic_sub_usize,
    repair_atomic_sub_usize_with_order,
};
use vyre_driver::BackendError;

use super::staging_reserve::reserve_smallvec;

pub(crate) fn cuda_check(result: CUresult, operation: &str) -> Result<(), BackendError> {
    if result == CUresult::CUDA_SUCCESS {
        return Ok(());
    }
    Err(BackendError::DispatchFailed {
        code: Some(cuda_result_code(result)),
        message: format!("{operation} failed with {result:?}"),
    })
}

pub(crate) fn cuda_result_code(result: CUresult) -> i32 {
    result as i32
}

pub(crate) fn alloc_cuda_ptr(byte_len: usize, operation: &str) -> Result<u64, BackendError> {
    if byte_len == 0 {
        return Err(BackendError::InvalidProgram {
            fix: format!(
                "Fix: {operation} cannot allocate zero device bytes through cuMemAlloc_v2. Keep zero-sized CUDA buffers as null sentinels or allocate at least one byte when a captured graph needs a stable address."
            ),
        });
    }
    let mut ptr = 0u64;
    // SAFETY: FFI to libcuda.so cuMemAlloc_v2. &mut ptr is a valid
    // *mut CUdeviceptr output parameter and byte_len is non-zero by the
    // guard above. cuda_check propagates non-success CUresult values.
    unsafe {
        cuda_check(
            cudarc::driver::sys::cuMemAlloc_v2(&mut ptr, byte_len),
            operation,
        )?;
    }
    if ptr == 0 {
        return Err(BackendError::DispatchFailed {
            code: None,
            message: format!(
                "{operation} returned a null device pointer after reporting success for {byte_len} byte(s). Fix: update the CUDA driver or avoid this allocation shape."
            ),
        });
    }
    Ok(ptr)
}

#[derive(Debug)]
pub(crate) struct DispatchAllocations {
    pool: Arc<DeviceAllocationPool>,
    ptrs: SmallVec<[DeviceAllocation; 8]>,
    params: DeviceAllocation,
}

impl DispatchAllocations {
    pub(crate) fn new(
        buffer_count: usize,
        pool: Arc<DeviceAllocationPool>,
    ) -> Result<Self, BackendError> {
        let mut ptrs = SmallVec::new();
        reserve_smallvec(&mut ptrs, buffer_count, "dispatch allocation pointer")?;
        ptrs.extend((0..buffer_count).map(|_| DeviceAllocation::default()));
        Ok(Self {
            pool,
            ptrs,
            params: DeviceAllocation::default(),
        })
    }

    pub(crate) fn set_ptr(
        &mut self,
        index: usize,
        allocation: DeviceAllocation,
        context: &str,
    ) -> Result<(), BackendError> {
        let allocation_count = self.ptrs.len();
        let slot = self
            .ptrs
            .get_mut(index)
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA dispatch allocation table {context} expected buffer index {index} but only {allocation_count} allocation slot(s) exist. Rebuild the binding plan before launch.",
                ),
            })?;
        *slot = allocation;
        Ok(())
    }

    pub(crate) fn ptr(&self, index: usize, context: &str) -> Result<u64, BackendError> {
        let allocation_count = self.ptrs.len();
        self.ptrs
            .get(index)
            .map(|allocation| allocation.ptr)
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA dispatch allocation table {context} expected buffer index {index} but only {allocation_count} allocation slot(s) exist. Rebuild the binding plan before launch.",
                ),
            })
    }

    pub(crate) fn params_ptr(&self) -> u64 {
        self.params.ptr
    }

    pub(crate) fn byte_len(&self, index: usize, context: &str) -> Result<usize, BackendError> {
        let allocation_count = self.ptrs.len();
        self.ptrs
            .get(index)
            .map(|allocation| allocation.byte_len)
            .ok_or_else(|| BackendError::InvalidProgram {
                fix: format!(
                    "Fix: CUDA dispatch allocation table {context} expected buffer index {index} but only {allocation_count} allocation slot(s) exist. Rebuild the binding plan before launch.",
                ),
            })
    }

    pub(crate) fn set_params(&mut self, allocation: DeviceAllocation) {
        self.params = allocation;
    }
}

pub(crate) fn take_cached_allocation<T: Copy>(
    free: &DashMap<usize, ArrayQueue<T>, BuildHasherDefault<FxHasher>>,
    cached_bytes: &AtomicUsize,
    bucket: usize,
    context: &'static str,
) -> Option<T> {
    let queue = free.get(&bucket)?;
    let allocation = queue.pop()?;
    subtract_cached_bytes_or_repair(cached_bytes, bucket, context);
    Some(allocation)
}

impl Drop for DispatchAllocations {
    fn drop(&mut self) {
        for allocation in self.ptrs.drain(..) {
            self.pool.release(allocation);
        }
        self.pool.release(std::mem::take(&mut self.params));
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct DeviceAllocation {
    pub(crate) ptr: u64,
    pub(crate) byte_len: usize,
}

#[derive(Debug)]
pub(crate) struct DeviceAllocationPool {
    free: DashMap<usize, ArrayQueue<u64>, BuildHasherDefault<FxHasher>>,
    cached_bytes: AtomicUsize,
    allocated_bytes: AtomicUsize,
    max_cached_bytes: usize,
    /// Acquisitions served from the free-list (no `cuMemAlloc`): the pool's whole
    /// point. `hits` and `misses` together are the pool-hit-rate evidence W3-4
    /// requires, only the pool can distinguish the two (the caller cannot), so the
    /// counters live here at their source (ONE PLACE) and are read at the telemetry
    /// boundary. Saturating so a runaway loop can never panic the allocator.
    hits: AtomicU64,
    /// Acquisitions that fell through to a real `cuMemAlloc_v2` (an empty bucket).
    misses: AtomicU64,
}

impl DeviceAllocationPool {
    pub(crate) fn new(max_cached_bytes: usize) -> Self {
        Self {
            free: DashMap::with_hasher(BuildHasherDefault::<FxHasher>::default()),
            cached_bytes: AtomicUsize::new(0),
            allocated_bytes: AtomicUsize::new(0),
            max_cached_bytes,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    pub(crate) fn acquire(&self, byte_len: usize) -> Result<DeviceAllocation, BackendError> {
        let bucket = allocation_bucket(byte_len, "CUDA device allocation")?;
        if let Some(ptr) = self.take_cached(bucket)? {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(DeviceAllocation {
                ptr,
                byte_len: bucket,
            });
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        self.free.entry(bucket).or_insert_with(|| {
            ArrayQueue::new(allocation_bucket_cache_slots(bucket, self.max_cached_bytes))
        });
        let ptr = alloc_cuda_ptr(bucket, "cuMemAlloc_v2")?;
        if let Err(error) = add_cached_bytes(
            &self.allocated_bytes,
            bucket,
            "CUDA allocation-pool live device bytes",
        ) {
            free_cuda_ptr(ptr);
            return Err(error);
        }
        Ok(DeviceAllocation {
            ptr,
            byte_len: bucket,
        })
    }

    pub(crate) fn cached_bytes(&self) -> Result<usize, BackendError> {
        Ok(self.cached_bytes.load(Ordering::Acquire))
    }

    pub(crate) fn allocated_bytes(&self) -> Result<usize, BackendError> {
        Ok(self.allocated_bytes.load(Ordering::Acquire))
    }

    /// Acquisitions served from the free-list since the last hit-counter reset.
    pub(crate) fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    /// Acquisitions that fell through to a real `cuMemAlloc_v2` since the last reset.
    pub(crate) fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    /// Reset the hit/miss counters (NOT the cached buffers) so pool-hit-rate is
    /// measured over the same epoch as the rest of the telemetry snapshot. Called
    /// from the backend's `reset_telemetry`, alongside `CudaTelemetry::reset`.
    pub(crate) fn reset_hit_counters(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }

    pub(crate) fn clear(&self) -> Result<(), BackendError> {
        let mut freed_bytes = 0usize;
        for entry in &self.free {
            while let Some(ptr) = entry.value().pop() {
                free_cuda_ptr(ptr);
                freed_bytes = checked_add_usize_lazy(freed_bytes, *entry.key(), || {
                    BackendError::InvalidProgram {
                        fix: "Fix: CUDA allocation-pool clear byte accounting overflowed usize; allocator state is corrupt."
                            .to_string(),
                    }
                })?;
            }
        }
        self.free.clear();
        self.cached_bytes.store(0, Ordering::Release);
        subtract_cached_bytes_or_repair(
            &self.allocated_bytes,
            freed_bytes,
            "CUDA allocation-pool live device bytes",
        );
        Ok(())
    }

    fn take_cached(&self, bucket: usize) -> Result<Option<u64>, BackendError> {
        Ok(take_cached_allocation(
            &self.free,
            &self.cached_bytes,
            bucket,
            "CUDA allocation-pool cached device bytes",
        ))
    }

    pub(crate) fn release(&self, allocation: DeviceAllocation) {
        if allocation.ptr == 0 || allocation.byte_len == 0 {
            return;
        }
        let Some(queue) = self.free.get(&allocation.byte_len) else {
            free_cuda_ptr(allocation.ptr);
            if let Err(error) = subtract_cached_bytes(&self.allocated_bytes, allocation.byte_len) {
                tracing::error!("{error}");
            }
            return;
        };
        if !reserve_cached_bytes(
            &self.cached_bytes,
            self.max_cached_bytes,
            allocation.byte_len,
        ) {
            free_cuda_ptr(allocation.ptr);
            if let Err(error) = subtract_cached_bytes(&self.allocated_bytes, allocation.byte_len) {
                tracing::error!("{error}");
            }
            return;
        }

        if let Err(ptr) = queue.push(allocation.ptr) {
            subtract_cached_bytes_or_repair(
                &self.cached_bytes,
                allocation.byte_len,
                "CUDA allocation-pool cached device bytes",
            );
            free_cuda_ptr(ptr);
            if let Err(error) = subtract_cached_bytes(&self.allocated_bytes, allocation.byte_len) {
                tracing::error!("{error}");
            }
        }
    }
}

pub(crate) fn allocation_bucket(
    byte_len: usize,
    label: &'static str,
) -> Result<usize, BackendError> {
    byte_len
        .max(1)
        .checked_next_power_of_two()
        .ok_or_else(|| BackendError::DispatchFailed {
            code: None,
            message: format!(
                "{label} request of {byte_len} bytes cannot be rounded to a power-of-two bucket. Fix: cap dispatch buffer sizes before allocation."
            ),
        })
}

pub(crate) fn allocation_bucket_cache_slots(bucket: usize, max_cached_bytes: usize) -> usize {
    const ALLOCATION_BUCKET_MAX_SLOTS: usize = 1024;
    let slots_by_budget = max_cached_bytes
        .checked_div(bucket.max(1))
        .unwrap_or(0)
        .max(1);
    slots_by_budget.min(ALLOCATION_BUCKET_MAX_SLOTS)
}

pub(crate) fn reserve_cached_bytes(
    counter: &AtomicUsize,
    max_cached_bytes: usize,
    bytes: usize,
) -> bool {
    checked_atomic_add_usize_guarded_with_order(
        counter,
        bytes,
        Ordering::Acquire,
        Ordering::AcqRel,
        Ordering::Acquire,
        |_, _| (),
        |next| {
            if next > max_cached_bytes {
                Err(())
            } else {
                Ok(())
            }
        },
    )
    .is_ok()
}

pub(crate) fn add_cached_bytes(
    counter: &AtomicUsize,
    bytes: usize,
    label: &'static str,
) -> Result<(), BackendError> {
    checked_atomic_add_usize_with_order(
        counter,
        bytes,
        Ordering::Acquire,
        Ordering::AcqRel,
        Ordering::Acquire,
        |observed, attempted| {
            BackendError::InvalidProgram {
                fix: format!(
                    "Fix: {label} accounting overflowed while adding {attempted} to observed {observed}; shard the allocation workload before enqueueing more CUDA work."
                ),
            }
        },
    )
}

pub(crate) fn subtract_cached_bytes(
    counter: &AtomicUsize,
    bytes: usize,
) -> Result<(), BackendError> {
    checked_atomic_sub_usize(counter, bytes, |observed, attempted| {
        BackendError::InvalidProgram {
            fix: format!(
                "Fix: CUDA allocation-pool byte accounting underflowed while subtracting {attempted} from observed {observed}; allocator state is corrupt."
            ),
        }
    })
}

pub(crate) fn subtract_cached_bytes_or_repair(
    counter: &AtomicUsize,
    bytes: usize,
    label: &'static str,
) {
    repair_atomic_sub_usize_with_order(
        counter,
        bytes,
        Ordering::Acquire,
        Ordering::AcqRel,
        Ordering::Acquire,
        |observed, attempted| {
            tracing::error!(
                "{label} underflowed while subtracting {attempted} from observed {observed}; repaired accounting to zero."
            );
        },
    );
}

pub(crate) fn free_cuda_ptr_with_label(ptr: u64, label: &str) {
    if ptr == 0 {
        return;
    }
    // SAFETY: FFI to libcuda.so cuMemFree_v2. ptr was returned by a
    // matching cuMemAlloc_v2 call (the pool owns the lifetime); the
    // null guard above ensures we never pass 0. CUDA_SUCCESS check records
    // unexpected failures without propagating (free runs on Drop / pool clear
    // paths where ?-propagation is not available).
    unsafe {
        let result = cudarc::driver::sys::cuMemFree_v2(ptr);
        if result != CUresult::CUDA_SUCCESS {
            tracing::error!(
                "Fix: cuMemFree_v2 failed while releasing {label} with {result:?}; ensure all launches using the allocation have completed."
            );
        }
    }
}

pub(crate) fn free_cuda_ptr(ptr: u64) {
    free_cuda_ptr_with_label(ptr, "CUDA allocation");
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;

    use super::subtract_cached_bytes;

    #[test]
    fn subtract_cached_bytes_fails_loudly_on_accounting_underflow() {
        let counter = AtomicUsize::new(4);
        let error = subtract_cached_bytes(&counter, 8)
            .expect_err("Fix: allocation-pool underflow must return a typed error.");
        assert_eq!(
            error.to_string(),
            "Invalid program: Fix: CUDA allocation-pool byte accounting underflowed while subtracting 8 from observed 4; allocator state is corrupt."
        );
    }
}
