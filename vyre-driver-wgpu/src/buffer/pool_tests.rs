//! Inline coverage for `class_index`, `free_bucket_capacity`, `release` and
//! `size_class`, which no integration test can name.

#[cfg(feature = "device-tests")]
use super::BufferPool;
use super::{class_index, free_bucket_capacity, size_class};
#[cfg(feature = "device-tests")]
use proptest::prelude::*;

#[test]
fn retained_byte_budget_is_not_used_as_queue_capacity() {
    assert_eq!(
        free_bucket_capacity(1 << 30),
        1024,
        "Fix: a 1 GiB byte budget must not allocate 1 GiB queue slots per bucket"
    );
    assert_eq!(
        free_bucket_capacity(8),
        2,
        "Fix: tiny retained-byte budgets should still translate to bounded entry capacity"
    );
}

#[test]
fn oversized_size_classes_return_errors_instead_of_panicking() {
    let error = size_class((1u64 << 63) + 1)
        .expect_err("oversized buffer length must be rejected before pool indexing");
    assert!(
        error
            .to_string()
            .contains("power-of-two persistent pool size class"),
        "unexpected error: {error}"
    );

    assert_eq!(
        class_index(0).expect("Fix: minimum size class should fit"),
        2
    );
    let error =
        class_index(u64::MAX).expect_err("invalid retained allocation length must be rejected");
    assert!(
        error.to_string().contains("not a power-of-two"),
        "unexpected error: {error}"
    );
}

/// A pool built without tiering has no event queue, so it can never fall
/// behind. Reporting anything but zero there would read as a degradation that
/// did not happen.
#[test]
fn a_pool_without_tiering_reports_no_dropped_events() {
    assert_eq!(
        super::BufferPoolStats::default().dropped_tiering_events,
        0,
        "Fix: the untiered pool's dropped-event count must start and stay at zero"
    );
}

/// The tiering event queue is bounded and its enqueue is `try_send`, so a
/// caller that outruns the metadata worker loses events. That loss is only
/// tolerable because it is counted: an uncounted drop degrades reuse with no
/// counter a caller can read.
///
/// The drain thread is what makes saturation racy, so this drives the counting
/// half through `PoolTiering::from_sender` against a receiver held open and
/// never drained. The queue is therefore full after exactly `capacity` sends,
/// and every send past it is a drop with a known count.
#[test]
fn a_full_event_queue_counts_every_drop_and_never_blocks_the_caller() {
    const CAPACITY: usize = 4;
    const OVERFLOW: usize = 7;

    let (sender, receiver) = crossbeam_channel::bounded(CAPACITY);
    let pending = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let tiering = super::PoolTiering::from_sender(sender, std::sync::Arc::clone(&pending));

    for key in 0..CAPACITY as u64 {
        tiering.record_retained(key, 64);
    }
    assert_eq!(
        tiering.dropped_events(),
        0,
        "Fix: sends that fit the queue must not be counted as drops"
    );
    assert_eq!(
        pending.load(std::sync::atomic::Ordering::Acquire),
        CAPACITY,
        "Fix: an accepted event must stay pending until the worker drains it"
    );

    for key in 0..OVERFLOW as u64 {
        tiering.record_access(key);
    }
    assert_eq!(
        tiering.dropped_events(),
        OVERFLOW,
        "Fix: every send past a full queue must raise the dropped-event count"
    );
    assert_eq!(
        pending.load(std::sync::atomic::Ordering::Acquire),
        CAPACITY,
        "Fix: a dropped event must not be left counted as pending"
    );

    drop(receiver);
    tiering.record_access(u64::MAX);
    assert_eq!(
        tiering.dropped_events(),
        OVERFLOW + 1,
        "Fix: a disconnected worker must be counted as a drop, not silently ignored"
    );
}

#[cfg(feature = "device-tests")]
#[test]
fn acquire_release_reuses_power_of_two_classes() {
    let arc = crate::runtime::cached_device()
        .expect("Fix: GPU device is required for persistent buffer pool test");
    let (device, queue) = &*arc;
    let config = vyre_driver::DispatchConfig::default();
    let pool = BufferPool::new(device.clone(), queue.clone(), &config);
    for len in 1..=1000 {
        let handle = pool
            .acquire(
                len,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            )
            .expect("Fix: pooled allocation should succeed");
        pool.release(handle);
    }
    assert!(
        pool.stats().allocations <= 16,
        "Fix: pool should allocate by power-of-two classes, stats={:?}",
        pool.stats()
    );
}

#[cfg(feature = "device-tests")]
#[test]
fn pooled_reuse_updates_logical_element_count() {
    let arc = crate::runtime::cached_device()
        .expect("Fix: GPU device is required for persistent buffer pool test");
    let (device, queue) = &*arc;
    let config = vyre_driver::DispatchConfig::default();
    let pool = BufferPool::new(device.clone(), queue.clone(), &config);
    let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;

    let large = pool
        .acquire(64, usage)
        .expect("Fix: initial pooled allocation should succeed");
    assert_eq!(large.element_count(), 64);
    pool.release(large);

    let small = pool
        .acquire(7, usage)
        .expect("Fix: pooled reuse should succeed");
    assert_eq!(
        small.element_count(),
        7,
        "Fix: reusing a larger allocation must not leak the previous logical element count"
    );
    assert_eq!(small.byte_len(), 7);
}

#[cfg(feature = "device-tests")]
#[test]
fn tiering_acquire_release_is_nonblocking_under_contention() {
    let arc = crate::runtime::cached_device()
        .expect("Fix: GPU device is required for persistent buffer pool test");
    let (device, queue) = &*arc;
    let config = vyre_driver::DispatchConfig::default();
    let pool = BufferPool::with_tiering(
        device.clone(),
        queue.clone(),
        &config,
        vec![crate::runtime::cache::CacheTier::new("hot", 1 << 20)],
    )
    .expect("Fix: tiered buffer pool construction should succeed");
    let handle = pool
        .acquire(
            64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        )
        .expect("Fix: acquire before poisoning should succeed");
    let tiering = pool
        .inner
        .tiering
        .as_ref()
        .expect("Fix: with_tiering must attach a tiering policy")
        .clone();
    pool.release(handle);
    let mut workers = Vec::new();
    for _ in 0..4 {
        let pool = pool.clone();
        workers.push(std::thread::spawn(move || {
            for _ in 0..32 {
                let handle = pool
                    .acquire(
                        64,
                        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    )
                    .expect("Fix: pooled allocation should not fail under tiering contention");
                pool.release(handle);
            }
        }));
    }
    for worker in workers {
        worker
            .join()
            .expect("Fix: buffer-pool contention worker must not panic");
    }
    tiering.drain_all_for_test();
    assert_eq!(
        pool.stats().dropped_tiering_events,
        0,
        "Fix: normal contention must not drop tiering metadata events"
    );
}

#[cfg(feature = "device-tests")]
proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn alternating_usage_hit_rate(
        sizes in prop::collection::vec(1u64..=65536, 20..=200),
    ) {
        let arc = crate::runtime::cached_device()
            .expect("Fix: GPU device is required for persistent buffer pool test");
        let (device, queue) = &*arc;
        let config = vyre_driver::DispatchConfig::default();
        let pool = BufferPool::new(device.clone(), queue.clone(), &config);

        let usage_a = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let usage_b = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::INDIRECT;

        // Round 1: acquire alternating usages, then release everything.
        let mut handles = Vec::with_capacity(sizes.len());
        for (i, &len) in sizes.iter().enumerate() {
            let usage = if i % 2 == 0 { usage_a } else { usage_b };
            handles.push(pool.acquire(len, usage).unwrap());
        }
        for h in handles {
            pool.release(h);
        }

        let stats_after_first = pool.stats();
        prop_assert_eq!(
            stats_after_first.hits, 0,
            "first round should be 100% fresh allocations"
        );

        // Round 2: identical pattern.
        let mut handles = Vec::with_capacity(sizes.len());
        for (i, &len) in sizes.iter().enumerate() {
            let usage = if i % 2 == 0 { usage_a } else { usage_b };
            handles.push(pool.acquire(len, usage).unwrap());
        }
        for h in handles {
            pool.release(h);
        }

        let stats_after_second = pool.stats();
        let second_round_hits = stats_after_second.hits - stats_after_first.hits;
        let total = sizes.len();
        let hit_rate = second_round_hits as f64 / total as f64;
        prop_assert!(
            hit_rate >= 0.95,
            "second round hit rate should be >= 95%, got {:.2}% ({}/{}), stats={:?}",
            hit_rate * 100.0,
            second_round_hits,
            total,
            stats_after_second
        );
    }
}
