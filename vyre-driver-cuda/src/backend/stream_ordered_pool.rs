//! Stream-ordered device allocation via the CUDA driver's own memory pool.
//!
//! The synchronous [`DeviceAllocationPool`](super::allocations::DeviceAllocationPool)
//! bucket-recycles raw `cuMemAlloc_v2` blocks behind a host-side free list: every
//! acquire/release is a *host* operation that must be ordered by hand against the
//! stream that actually consumes the memory. This module binds the device's
//! **default CUDA memory pool** and drives it with the stream-ordered allocator
//! (`cuMemAllocFromPoolAsync` / `cuMemFreeAsync`), so an allocation and its free
//! ride the same stream as the dispatch that uses them, the driver reuses a freed
//! block for the next same-stream allocation without a host round-trip, and keeps
//! freed physical memory *reserved* (bounded by a release threshold) instead of
//! handing it back to the OS on every sync.
//!
//! We bind the **default** pool rather than `cuMemPoolCreate` a private one: the
//! default pool is owned by the context, so there is nothing to destroy here and
//! no `Drop` ordering hazard against the context teardown. The only state we own
//! is the release-threshold policy, which we set on construction.
//!
//! This is a self-contained, hardware-tested allocator surface. It is **not** wired
//! into the current hot dispatch path, and that is a deliberate, measured decision
//! rather than a pending TODO: on the warm steady state the
//! [`DeviceAllocationPool`](super::allocations::DeviceAllocationPool) free list
//! serves an acquire from a lock-free `ArrayQueue::pop` and a release from a
//! `queue.push`: **zero** CUDA driver calls per dispatch, and the release is
//! already correctly stream-ordered because the owning `CudaPendingDispatch` holds
//! the buffers until its completion event fires before dropping them. Routing that
//! same per-dispatch buffer through `cuMemAllocFromPoolAsync` + `cuMemFreeAsync`
//! would add two driver calls where there are currently none, i.e. a hot-path
//! pessimization (Law 7), with no correctness gain. The stream-ordered pool wins
//! only where the free list structurally cannot: concurrent **multi-stream**
//! dispatch (cross-stream block reuse the exact-size buckets cannot share) and
//! auto-trim of reserved memory under pressure. It is kept here, public, proven,
//! and tested, as the ready primitive for that future design, not as an
//! unfinished hot-path integration.

use std::ffi::c_void;

use cudarc::driver::sys::{
    cuDeviceGetDefaultMemPool, cuMemAllocFromPoolAsync, cuMemFreeAsync, cuMemPoolGetAttribute,
    cuMemPoolSetAttribute, cuMemPoolTrimTo, CUdevice, CUdeviceptr, CUmemPool_attribute_enum,
    CUmemoryPool, CUstream,
};
use cudarc::driver::CudaContext;
use vyre_driver::BackendError;

use super::allocations::cuda_check;

/// Retain **all** freed physical memory in the pool for reuse instead of
/// releasing it to the OS on the next stream sync. The default threshold is `0`
/// (release everything on sync), which defeats cross-dispatch reuse, a
/// re-dispatch loop would fault in fresh pages every iteration. `u64::MAX` keeps
/// every freed block reserved, so a subsequent same-size allocation is served
/// from the pool. Operators that need to hand memory back can call
/// [`CudaStreamOrderedPool::trim`].
const RETAIN_ALL_FREED_BYTES: u64 = u64::MAX;

/// A stream-ordered device allocator bound to a device's default CUDA memory
/// pool. Cheap to construct (it only queries the default pool handle and sets a
/// policy attribute); holds no owned device memory of its own.
///
/// Requires the target device's CUDA context to be current on the calling thread
///: construct it only after a [`CudaBackend`](super::dispatch::CudaBackend) for
/// the same ordinal has bound the context.
#[derive(Debug, Clone)]
pub struct CudaStreamOrderedPool {
    pool: CUmemoryPool,
}

// SAFETY: `pool` is a `CUmemoryPool` handle owned by the CUDA context, not by
// this wrapper. The driver's pool operations are internally synchronized, and we
// never free the pool here (it is the context's default pool). Sharing the handle
// across threads is sound as long as the owning context outlives it, which the
// caller guarantees by holding the `CudaBackend`.
unsafe impl Send for CudaStreamOrderedPool {}
unsafe impl Sync for CudaStreamOrderedPool {}

impl CudaStreamOrderedPool {
    /// Bind the default memory pool of the device backing `ctx` and configure it
    /// to retain freed memory for reuse.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] if the default pool cannot be queried or the
    /// release-threshold attribute cannot be set (e.g. the driver predates
    /// stream-ordered memory pools).
    pub fn for_context(ctx: &CudaContext) -> Result<Self, BackendError> {
        let device: CUdevice = ctx.cu_device();
        let mut pool: CUmemoryPool = std::ptr::null_mut();
        // SAFETY: FFI to libcuda.so. `&mut pool` is a valid out-pointer and
        // `device` is a live CUdevice from the bound context. cuda_check
        // converts any non-success CUresult into a BackendError.
        unsafe {
            cuda_check(
                cuDeviceGetDefaultMemPool(&mut pool, device),
                "cuDeviceGetDefaultMemPool",
            )?;
        }
        if pool.is_null() {
            return Err(BackendError::DispatchFailed {
                code: None,
                message: "cuDeviceGetDefaultMemPool reported success but returned a null pool handle. Fix: update the CUDA driver; stream-ordered memory pools require a driver that exposes a per-device default pool.".to_string(),
            });
        }
        let this = Self { pool };
        this.set_release_threshold(RETAIN_ALL_FREED_BYTES)?;
        Ok(this)
    }

    /// Set the pool's release threshold: the driver keeps at least `bytes` of
    /// freed memory reserved across stream syncs before returning any to the OS.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] if the attribute write fails.
    pub fn set_release_threshold(&self, bytes: u64) -> Result<(), BackendError> {
        let value = bytes;
        // SAFETY: FFI. The RELEASE_THRESHOLD attribute takes a `cuuint64_t`
        // value by pointer; `&value` points to a live u64 for the call's
        // duration and the driver copies it. cuda_check propagates failures.
        unsafe {
            cuda_check(
                cuMemPoolSetAttribute(
                    self.pool,
                    CUmemPool_attribute_enum::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    &value as *const u64 as *mut c_void,
                ),
                "cuMemPoolSetAttribute(RELEASE_THRESHOLD)",
            )?;
        }
        Ok(())
    }

    /// Allocate `byte_len` device bytes from the pool, ordered on `stream`. The
    /// returned pointer is valid for work enqueued on `stream` after this call;
    /// the caller must not use it on another stream without an explicit
    /// cross-stream dependency.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] for a zero-byte request (kept as a null sentinel,
    /// matching [`alloc_cuda_ptr`](super::allocations::alloc_cuda_ptr)) or if the
    /// driver allocation fails.
    pub fn alloc_async(&self, byte_len: usize, stream: CUstream) -> Result<u64, BackendError> {
        if byte_len == 0 {
            return Err(BackendError::InvalidProgram {
                fix: "Fix: CudaStreamOrderedPool::alloc_async cannot allocate zero device bytes. Keep zero-sized buffers as null sentinels or request at least one byte.".to_string(),
            });
        }
        let mut ptr: CUdeviceptr = 0;
        // SAFETY: FFI. `&mut ptr` is a valid out-pointer, `byte_len` is non-zero
        // by the guard above, `self.pool` is a live pool handle from the context,
        // and `stream` is a raw CUstream owned by the caller. cuda_check
        // converts failures into BackendError.
        unsafe {
            cuda_check(
                cuMemAllocFromPoolAsync(&mut ptr, byte_len, self.pool, stream),
                "cuMemAllocFromPoolAsync",
            )?;
        }
        if ptr == 0 {
            return Err(BackendError::DispatchFailed {
                code: None,
                message: format!(
                    "cuMemAllocFromPoolAsync returned a null device pointer after reporting success for {byte_len} byte(s). Fix: update the CUDA driver or avoid this allocation shape."
                ),
            });
        }
        Ok(ptr)
    }

    /// Free a pool allocation, ordered on `stream`. The physical block is retained
    /// in the pool (per the release threshold) for reuse by a later same-stream
    /// allocation. A null/zero pointer is a no-op, matching the alloc null
    /// sentinel.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] if the driver free fails.
    pub fn free_async(&self, ptr: u64, stream: CUstream) -> Result<(), BackendError> {
        if ptr == 0 {
            return Ok(());
        }
        // SAFETY: FFI. `ptr` is a live pool allocation from `alloc_async` and
        // `stream` is a raw CUstream owned by the caller. The free is ordered on
        // the stream; cuda_check propagates failures.
        unsafe {
            cuda_check(cuMemFreeAsync(ptr, stream), "cuMemFreeAsync")?;
        }
        Ok(())
    }

    /// Physical memory currently reserved by the pool from the OS (bytes). Grows
    /// as new blocks are allocated and stays put as blocks are freed (up to the
    /// release threshold), so a stable value across a free/realloc cycle is direct
    /// evidence of reuse.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] if the attribute read fails.
    pub fn reserved_bytes(&self) -> Result<u64, BackendError> {
        self.attr_u64(
            CUmemPool_attribute_enum::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
            "CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT",
        )
    }

    /// Device memory currently handed out to live allocations (bytes). Drops back
    /// toward zero as allocations are freed even while `reserved_bytes` stays
    /// high.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] if the attribute read fails.
    pub fn used_bytes(&self) -> Result<u64, BackendError> {
        self.attr_u64(
            CUmemPool_attribute_enum::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
            "CU_MEMPOOL_ATTR_USED_MEM_CURRENT",
        )
    }

    fn attr_u64(
        &self,
        attr: CUmemPool_attribute_enum,
        label: &'static str,
    ) -> Result<u64, BackendError> {
        let mut value: u64 = 0;
        // SAFETY: FFI. These RESERVED/USED attributes yield a `cuuint64_t` into
        // the provided pointer; `&mut value` is a live u64 out-pointer for the
        // call. cuda_check propagates failures.
        unsafe {
            cuda_check(
                cuMemPoolGetAttribute(self.pool, attr, &mut value as *mut u64 as *mut c_void),
                label,
            )?;
        }
        Ok(value)
    }

    /// Release pooled memory back to the OS, keeping at least `min_keep_bytes`
    /// reserved. Use this to bound resident VRAM after a burst of dispatches when
    /// the retained reservation is no longer worth holding.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] if the trim fails.
    pub fn trim(&self, min_keep_bytes: usize) -> Result<(), BackendError> {
        // SAFETY: FFI. `self.pool` is a live pool handle; the driver trims its
        // own reservation. cuda_check propagates failures.
        unsafe {
            cuda_check(
                cuMemPoolTrimTo(self.pool, min_keep_bytes),
                "cuMemPoolTrimTo",
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::dispatch::CudaBackend;
    use crate::stream::CudaStream;
    use cudarc::driver::sys::{cuMemcpyDtoHAsync_v2, cuMemsetD32Async};
    use std::ffi::c_void;

    /// Hardware evidence on the live GPU that the stream-ordered pool hands back
    /// (a) *usable* device memory, proven by a memset-then-readback roundtrip 
    /// and (b) memory it *reuses* across a free/realloc cycle, proven by the
    /// pool's reserved-bytes staying flat while a same-size block is freed and
    /// re-allocated. Both are the raison d'être of the stream-ordered pool over a
    /// plain per-dispatch cuMemAlloc/cuMemFree.
    #[test]
    fn stream_ordered_pool_serves_usable_memory_and_reuses_reserved_blocks_on_gpu() {
        let backend = match CudaBackend::acquire() {
            Ok(backend) => backend,
            Err(error) => panic!(
                "CUDA device required for the stream-ordered pool evidence test but acquire failed: {error}. Fix: run on a host with a visible CUDA GPU (nvidia-smi -L); do not skip GPU tests on a GPU host."
            ),
        };
        let pool = CudaStreamOrderedPool::for_context(&backend.ctx)
            .expect("bind default stream-ordered memory pool");
        let stream = CudaStream::non_blocking().expect("create non-blocking stream");

        const WORDS: usize = 1024;
        const BYTES: usize = WORDS * 4;
        const FILL: u32 = 0xABCD_1234;

        // (a) Usability: allocate, fill on the stream, copy back, verify bytes.
        let ptr = pool
            .alloc_async(BYTES, stream.raw())
            .expect("stream-ordered allocation");
        // SAFETY: FFI. `ptr` is a live BYTES-sized pool block, WORDS u32 words
        // fit exactly, and the memset is ordered on the same stream as the alloc.
        unsafe {
            cuda_check(
                cuMemsetD32Async(ptr, FILL, WORDS, stream.raw()),
                "cuMemsetD32Async",
            )
            .expect("memset stream-ordered block");
        }
        let mut host = vec![0u32; WORDS];
        // SAFETY: FFI. `host` holds WORDS u32 (= BYTES) contiguous bytes, `ptr`
        // is the live device block, and the copy is ordered after the memset on
        // the same stream.
        unsafe {
            cuda_check(
                cuMemcpyDtoHAsync_v2(host.as_mut_ptr() as *mut c_void, ptr, BYTES, stream.raw()),
                "cuMemcpyDtoHAsync_v2",
            )
            .expect("copy stream-ordered block to host");
        }
        stream.synchronize().expect("sync after fill+readback");
        assert!(
            host.iter().all(|&word| word == FILL),
            "stream-ordered pool must return usable device memory: every one of {WORDS} words should read back as {FILL:#010x}, got e.g. {:#010x}",
            host[0]
        );

        let used_live = pool.used_bytes().expect("query used bytes while live");
        assert!(
            used_live >= BYTES as u64,
            "pool used-bytes ({used_live}) must reflect the live {BYTES}-byte allocation"
        );

        // (b) Reuse: free the block (retained per the release threshold), then
        // re-allocate the same size. The pool's reserved reservation must not
        // grow (the freed block is reused rather than faulted in fresh).
        pool.free_async(ptr, stream.raw())
            .expect("stream-ordered free");
        stream.synchronize().expect("sync after free");
        let reserved_after_free = pool
            .reserved_bytes()
            .expect("query reserved bytes after free");
        assert!(
            reserved_after_free >= BYTES as u64,
            "release threshold must keep the freed {BYTES}-byte block reserved for reuse; reserved={reserved_after_free}"
        );

        let ptr2 = pool
            .alloc_async(BYTES, stream.raw())
            .expect("re-allocate after free");
        stream.synchronize().expect("sync after realloc");
        let reserved_after_realloc = pool
            .reserved_bytes()
            .expect("query reserved bytes after realloc");
        assert_eq!(
            reserved_after_realloc, reserved_after_free,
            "re-allocating a just-freed same-size block must reuse the reserved memory, not grow the pool's OS reservation (before={reserved_after_free}, after={reserved_after_realloc})"
        );

        pool.free_async(ptr2, stream.raw())
            .expect("free second allocation");
        stream.synchronize().expect("final sync");

        // Trimming hands the reservation back to the OS: reserved must fall.
        pool.trim(0).expect("trim pool reservation to zero-keep");
        let reserved_after_trim = pool
            .reserved_bytes()
            .expect("query reserved bytes after trim");
        assert!(
            reserved_after_trim < reserved_after_realloc,
            "trim(0) must release retained reservation back to the OS (before={reserved_after_realloc}, after={reserved_after_trim})"
        );
    }

    /// Edge + telemetry coverage the hot-path integration (W3-4) depends on but
    /// the reuse test above does not exercise: (a) a ZERO-byte request errors as a
    /// null sentinel rather than faulting, a real dispatch with `param_bytes == 0`
    /// maps to a null `DeviceAllocation::default()` on the synchronous pool, so the
    /// stream-ordered path must reject 0 the same way instead of driver-faulting;
    /// and (b) `used_bytes` accounts MULTIPLE concurrently-live blocks in aggregate
    /// and drops as one is freed, the live-usage accounting the integration will
    /// surface as its pool telemetry (the equivalent of the sync pool's hit/miss
    /// evidence), so it must move with real allocation state, not stay flat.
    #[test]
    fn stream_ordered_pool_rejects_zero_bytes_and_accounts_multiple_live_blocks_on_gpu() {
        let backend = match CudaBackend::acquire() {
            Ok(backend) => backend,
            Err(error) => panic!(
                "CUDA device required for the stream-ordered pool telemetry test but acquire failed: {error}. Fix: run on a host with a visible CUDA GPU (nvidia-smi -L); do not skip GPU tests on a GPU host."
            ),
        };
        let pool = CudaStreamOrderedPool::for_context(&backend.ctx)
            .expect("bind default stream-ordered memory pool");
        let stream = CudaStream::non_blocking().expect("create non-blocking stream");

        // (a) Zero-byte is a null sentinel (matches the synchronous pool's default
        // allocation), NOT a driver fault, the integration hits this whenever a
        // dispatch has no launch params to upload.
        assert!(
            pool.alloc_async(0, stream.raw()).is_err(),
            "a zero-byte stream-ordered request must error as a null sentinel, not allocate"
        );

        // (b) Three distinct concurrently-live blocks: used-bytes must cover their
        // aggregate, then fall when one is freed (the freed block becomes reserved,
        // not used).
        const A: usize = 4096;
        const B: usize = 8192;
        const C: usize = 16384;
        let pa = pool.alloc_async(A, stream.raw()).expect("alloc A");
        let pb = pool.alloc_async(B, stream.raw()).expect("alloc B");
        let pc = pool.alloc_async(C, stream.raw()).expect("alloc C");
        stream.synchronize().expect("sync after three allocs");
        let used_all = pool.used_bytes().expect("used bytes with three live blocks");
        assert!(
            used_all >= (A + B + C) as u64,
            "used-bytes ({used_all}) must account all three live blocks (>= {} B)",
            A + B + C
        );

        pool.free_async(pb, stream.raw()).expect("free B");
        stream.synchronize().expect("sync after freeing B");
        let used_after = pool.used_bytes().expect("used bytes after freeing B");
        assert!(
            used_after < used_all,
            "freeing a live block must reduce used-bytes, not stay flat (before={used_all}, after={used_after})"
        );
        assert!(
            used_after >= (A + C) as u64,
            "used-bytes ({used_after}) must still account the two blocks left live (>= {} B)",
            A + C
        );

        // Clean up the remaining live blocks on the same stream.
        pool.free_async(pa, stream.raw()).expect("free A");
        pool.free_async(pc, stream.raw()).expect("free C");
        stream.synchronize().expect("final cleanup sync");
    }
}
