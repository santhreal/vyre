use std::sync::Arc;

use super::*;
use crate::buffer::handle::GpuBufferHandle;

#[test]
fn poisoned_bind_group_cache_lock_recovers_without_aborting_dispatch_path() {
    let cache = BindGroupCache::new();
    let poisoned = cache.clone();
    let _ = std::thread::spawn(move || {
        let _guard = poisoned.lock_cache();
        panic!("poison bind group cache");
    })
    .join();

    std::panic::catch_unwind(|| {
        let _ = cache.stats();
    })
    .expect("Fix: poisoned bind-group cache must recover so GPU dispatch does not abort");
}

#[cfg(feature = "device-tests")]
#[test]
fn bind_group_cache_lru_heap_stays_capacity_scale() {
    let arc = crate::runtime::cached_device()
        .expect("Fix: GPU device is required for bind-group cache test");
    let (device, _) = &*arc;
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("vyre bind-group cache lru test layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(4),
            },
            count: None,
        }],
    });
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vyre bind-group cache lru test buffer"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("vyre bind-group cache lru test bind group"),
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });
    let cache = BindGroupCache::with_cap(4);

    for i in 0..64u64 {
        cache.insert_by_ids(1, &[i, 4], bind_group.clone());
    }

    let inner = cache.lock_cache();
    assert_eq!(inner.entries.len(), 4);
    assert!(
        inner.lru.len() <= inner.entries.len().saturating_mul(4).max(8),
        "Fix: bind-group LRU heap must compact stale entries to cache-capacity scale"
    );
}

/// Pins that bind-group reuse is keyed on the concrete buffers bound, not
/// on the binding layout alone, and counts the creations it saves.
///
/// This exists because a `patterns::bind_group_reuse` module in
/// vyre-emit-naga was removed after an audit found it grouped
/// `KernelDescriptor`s for bind-group sharing by hashing only their
/// binding LAYOUT (slot, dtype, count, memory class, visibility). A
/// `wgpu::BindGroup` binds a layout PLUS concrete resources, so that rule
/// declares two dispatches reading different buffers to be shareable.
/// Acting on it would bind the wrong buffer and silently compute on stale
/// data. This cache is the correct implementation and the only one that
/// ships; the assertion below is the difference between the two rules.
///
/// The counts are the contention-proof evidence that reuse actually fires:
/// six lookups over two distinct buffers must create exactly two bind
/// groups and reuse four. A layout-only key would create ONE and wrongly
/// share it across both buffers, which `misses == 2` rejects. Dropping the
/// buffer identity from the key regresses to exactly that bug.
#[test]
fn bind_group_reuse_keys_on_buffer_identity_not_layout_alone() {
    let arc = crate::runtime::cached_device()
        .expect("Fix: GPU device is required for bind-group identity test");
    let (device, queue) = &*arc;

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("vyre bind-group identity test layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(4),
            },
            count: None,
        }],
    });

    // Two DISTINCT buffers of identical size and usage. Same layout, same
    // byte length: a layout-only reuse rule cannot tell them apart.
    let buffer_a = GpuBufferHandle::upload(device, queue, &[1u8; 4], wgpu::BufferUsages::STORAGE)
        .expect("Fix: upload of buffer a must succeed");
    let buffer_b = GpuBufferHandle::upload(device, queue, &[2u8; 4], wgpu::BufferUsages::STORAGE)
        .expect("Fix: upload of buffer b must succeed");

    let cache = BindGroupCache::new();
    let layout_id = 1usize;
    let mut created = 0usize;

    // A repeated-dispatch sequence: three dispatches against buffer a,
    // then three against buffer b, all through one layout.
    for handle in [
        &buffer_a, &buffer_a, &buffer_a, &buffer_b, &buffer_b, &buffer_b,
    ] {
        let slice = std::slice::from_ref(handle);
        cache.get_or_create(layout_id, slice, || {
            created += 1;
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("vyre bind-group identity test bind group"),
                layout: &layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: handle.buffer().as_entire_binding(),
                }],
            })
        });
    }

    let stats = cache.stats();
    assert_eq!(
        created, 2,
        "six lookups over two distinct buffers must construct exactly two bind groups"
    );
    assert_eq!(
        stats.misses, 2,
        "each distinct buffer must miss exactly once. A layout-only key would \
         report 1 miss and share one bind group across both buffers, binding the \
         wrong resource on every dispatch against the second buffer."
    );
    assert_eq!(
        stats.hits, 4,
        "the two repeats of each buffer must reuse the cached bind group"
    );
    assert_eq!(stats.entries, 2, "one cached entry per distinct buffer");

    // The same buffer through the same layout must resolve to the very same
    // instance, which is what makes the four hits above a real saving.
    let first = cache.get_or_create(layout_id, std::slice::from_ref(&buffer_a), || {
        panic!("Fix: buffer a is already cached and must not be rebuilt")
    });
    let again = cache.get_or_create(layout_id, std::slice::from_ref(&buffer_a), || {
        panic!("Fix: buffer a is already cached and must not be rebuilt")
    });
    assert!(
        Arc::ptr_eq(&first, &again),
        "repeated lookups for one buffer must hand back one bind-group instance"
    );
}
