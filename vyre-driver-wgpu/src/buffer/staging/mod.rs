//! Reusable staging buffers for readback and upload.

use std::sync::{Arc, Mutex, MutexGuard};

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

/// Upper bound on retained free buffers per (size, usage) class.
const STAGING_BUFFER_POOL_CLASS_CAP: usize = 16;

/// Snapshot of [`StagingBufferPool`] counters.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct StagingBufferPoolStats {
    /// Number of fresh GPU buffer allocations.
    pub allocations: usize,
    /// Number of times a free buffer was reused.
    pub hits: usize,
}

/// Device-local staging buffer pool keyed by `(size, usage)`.
///
/// Hot dispatch paths such as `GpuBufferHandle::readback_until` acquire
/// readback staging buffers from this pool instead of creating a fresh
/// `wgpu::Buffer` on every call. Each `(size, usage)` class is capped at
/// `STAGING_BUFFER_POOL_CLASS_CAP` entries; evictions drop the
/// least-recently-used buffer.
#[derive(Clone, Default)]
pub struct StagingBufferPool {
    inner: Arc<Mutex<StagingBufferPoolInner>>,
}

impl std::fmt::Debug for StagingBufferPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StagingBufferPool").finish_non_exhaustive()
    }
}

#[derive(Default)]
struct StagingBufferPoolInner {
    free: FxHashMap<(u64, u32), SmallVec<[wgpu::Buffer; STAGING_BUFFER_POOL_CLASS_CAP]>>,
    allocations: usize,
    hits: usize,
}

impl StagingBufferPool {
    fn lock_inner(&self) -> MutexGuard<'_, StagingBufferPoolInner> {
        self.inner.lock().unwrap_or_else(|error| {
            tracing::error!(
                "Vyre WGPU staging buffer pool lock was poisoned: {error}. Fix: discard the pool after a panic; continuing with recovered state."
            );
            error.into_inner()
        })
    }

    /// Create an empty staging buffer pool.
    #[must_use]
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Return allocation and hit counters.
    #[must_use]
    pub fn stats(&self) -> StagingBufferPoolStats {
        let inner = self.lock_inner();
        StagingBufferPoolStats {
            allocations: inner.allocations,
            hits: inner.hits,
        }
    }

    /// Acquire a staging buffer with exactly `size` bytes and `usage`.
    ///
    /// Reuses a free buffer when one is available; otherwise creates a fresh
    /// GPU allocation and increments the allocation counter.
    pub fn acquire(
        &self,
        device: &wgpu::Device,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        let key = (size, usage.bits());
        let mut inner = self.lock_inner();
        if let Some(buffers) = inner.free.get_mut(&key) {
            if let Some(buffer) = buffers.pop() {
                inner.hits += 1;
                return buffer;
            }
        }
        inner.allocations += 1;
        drop(inner);
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vyre staging readback"),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Release a staging buffer back to the pool.
    ///
    /// The buffer is pushed to the MRU position of its `(size, usage)` class.
    /// If the class already holds 16 buffers, the LRU entry is dropped.
    pub fn release(&self, buffer: wgpu::Buffer, size: u64, usage: wgpu::BufferUsages) {
        let key = (size, usage.bits());
        let mut inner = self.lock_inner();
        let buffers = inner.free.entry(key).or_default();
        if buffers.len() == STAGING_BUFFER_POOL_CLASS_CAP {
            buffers.remove(0);
        }
        buffers.push(buffer);
    }
}

// Inline: covers the crate-private `StagingBufferPool::lock_inner`, which no
// integration test can reach.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::handle::GpuBufferHandle;

    /// StagingBufferPool must reuse buffers across readback calls so that 100
    /// readbacks of the same size allocate only ~1 buffer.
    #[test]
    fn staging_pool_reuses_buffers_on_hot_readback_loop() {
        let arc = crate::runtime::cached_device()
            .expect("Fix: GPU device is required for staging pool test");
        let (device, queue) = &*arc;

        // Create a small COPY_SRC buffer with known contents.
        let contents: Vec<u8> = vec![0xAB; 64];
        let handle =
            GpuBufferHandle::upload(device, queue, &contents, wgpu::BufferUsages::COPY_SRC)
                .expect("Fix: upload should succeed");

        let pool = StagingBufferPool::new();

        for _ in 0..100 {
            let mut out = Vec::new();
            handle
                .readback_until(device, Some(&pool), queue, &mut out, None)
                .expect("Fix: pooled readback should succeed");
            assert_eq!(out, contents, "readback bytes must match uploaded bytes");
        }

        let stats = pool.stats();
        assert!(
            stats.allocations <= 2,
            "hot loop of 100 identical readbacks should allocate at most 2 staging buffers, got {} allocations and {} hits",
            stats.allocations,
            stats.hits
        );
    }

    /// Without a pool, readback must still work and always create fresh buffers.
    #[test]
    fn readback_without_pool_always_allocates() {
        let arc = crate::runtime::cached_device()
            .expect("Fix: GPU device is required for readback regression test");
        let (device, queue) = &*arc;

        let contents: Vec<u8> = vec![0xCD; 32];
        let handle =
            GpuBufferHandle::upload(device, queue, &contents, wgpu::BufferUsages::COPY_SRC)
                .expect("Fix: upload should succeed");

        for _ in 0..5 {
            let mut out = Vec::new();
            handle
                .readback(device, queue, &mut out)
                .expect("Fix: unpooled readback should succeed");
            assert_eq!(out, contents);
        }
    }

    #[test]
    fn poisoned_staging_pool_lock_recovers_without_aborting_dispatch_path() {
        let pool = StagingBufferPool::new();
        let poisoned = pool.clone();
        let _ = std::thread::spawn(move || {
            let _guard = poisoned.lock_inner();
            panic!("poison staging buffer pool");
        })
        .join();

        std::panic::catch_unwind(|| {
            let _ = pool.stats();
        })
        .expect("Fix: poisoned staging pool must recover so GPU readback pooling does not abort");
    }
}
