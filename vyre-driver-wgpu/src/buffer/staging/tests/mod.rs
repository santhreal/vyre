//! Staging buffer reuse and lock-poisoning recovery.

use super::*;
use crate::buffer::handle::GpuBufferHandle;

/// StagingBufferPool must reuse buffers across readback calls so that 100
/// readbacks of the same size allocate only ~1 buffer.
#[test]
fn staging_pool_reuses_buffers_on_hot_readback_loop() {
    let arc =
        crate::runtime::cached_device().expect("Fix: GPU device is required for staging pool test");
    let (device, queue) = &*arc;

    // Create a small COPY_SRC buffer with known contents.
    let contents: Vec<u8> = vec![0xAB; 64];
    let handle = GpuBufferHandle::upload(device, queue, &contents, wgpu::BufferUsages::COPY_SRC)
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
    let handle = GpuBufferHandle::upload(device, queue, &contents, wgpu::BufferUsages::COPY_SRC)
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
