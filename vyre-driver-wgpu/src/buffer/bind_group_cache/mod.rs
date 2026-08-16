//! Bounded LRU cache for wgpu bind groups.

use std::cmp::{Ordering as CmpOrdering, Reverse};
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use super::handle::{aligned_len_u64, GpuBufferHandle};

/// Default cap for the [`BindGroupCache`] LRU.
const BIND_GROUP_CACHE_CAP: usize = 256;

/// Inline storage for bind-group cache keys: typical shaders use few bindings;
/// `SmallVec` avoids a heap `Vec` on most `get_or_create` calls.
type BindGroupHandleKey = SmallVec<[u64; 16]>;

/// Bounded LRU cache for wgpu bind groups, keyed by layout identity and
/// the ordered set of buffer handles bound to that layout.
///
/// wgpu bind-group creation is non-trivial; this cache eliminates the
/// redundant cost on repeated dispatches that share the same buffer
/// handles.  Capped at 256 entries with LRU eviction to prevent
/// descriptor-heap exhaustion on long-running servers.
#[derive(Clone)]
pub struct BindGroupCache {
    cache: Arc<Mutex<BindGroupCacheInner>>,
    hits: Arc<AtomicUsize>,
    misses: Arc<AtomicUsize>,
    evictions: Arc<AtomicUsize>,
}

struct BindGroupCacheInner {
    entries: FxHashMap<BindGroupCacheKey, BindGroupCacheEntry>,
    lru: BinaryHeap<Reverse<BindGroupLruEntry>>,
    cap: usize,
    next_generation: u64,
}

struct BindGroupCacheEntry {
    bind_group: Arc<wgpu::BindGroup>,
    last_seen: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct BindGroupLruEntry {
    last_seen: u64,
    key: BindGroupCacheKey,
}

impl Ord for BindGroupLruEntry {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        self.last_seen
            .cmp(&other.last_seen)
            .then_with(|| self.key.cmp(&other.key))
    }
}

impl PartialOrd for BindGroupLruEntry {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

fn push_bind_group_handle_key(key: &mut BindGroupHandleKey, handle: &GpuBufferHandle) -> bool {
    key.push(handle.allocation_identity());
    let Ok(aligned_len) = aligned_len_u64(handle.byte_len(), "bind-group handle key byte length")
    else {
        key.pop();
        return false;
    };
    key.push(aligned_len);
    true
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
struct BindGroupCacheKey {
    layout_id: usize,
    handles: BindGroupHandleKey,
}

impl std::fmt::Debug for BindGroupCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BindGroupCache")
            .field("hits", &self.hits.load(Ordering::Relaxed))
            .field("misses", &self.misses.load(Ordering::Relaxed))
            .field("evictions", &self.evictions.load(Ordering::Relaxed))
            .field("entries", &self.lock_cache().entries.len())
            .finish_non_exhaustive()
    }
}

impl Default for BindGroupCache {
    fn default() -> Self {
        Self::new()
    }
}

impl BindGroupCacheInner {
    fn next_lru_generation(&mut self) -> u64 {
        let generation = self.next_generation;
        self.next_generation = self.next_generation.wrapping_add(1);
        generation
    }

    fn touch_existing(&mut self, key: &BindGroupCacheKey) -> Option<Arc<wgpu::BindGroup>> {
        let generation = self.next_lru_generation();
        let bind_group = {
            let entry = self.entries.get_mut(key)?;
            entry.last_seen = generation;
            Arc::clone(&entry.bind_group)
        };
        self.lru.push(Reverse(BindGroupLruEntry {
            last_seen: generation,
            key: key.clone(),
        }));
        self.compact_lru_if_needed();
        Some(bind_group)
    }

    fn insert_entry(&mut self, key: BindGroupCacheKey, bind_group: Arc<wgpu::BindGroup>) {
        let generation = self.next_lru_generation();
        self.entries.insert(
            key.clone(),
            BindGroupCacheEntry {
                bind_group,
                last_seen: generation,
            },
        );
        self.lru.push(Reverse(BindGroupLruEntry {
            last_seen: generation,
            key,
        }));
        self.compact_lru_if_needed();
    }

    fn evict_to_cap(&mut self, mut on_evict: impl FnMut()) {
        while self.entries.len() > self.cap {
            let Some(key) = self.pop_lru_key() else { break };
            if self.entries.remove(&key).is_some() {
                on_evict();
            }
        }
    }

    fn pop_lru_key(&mut self) -> Option<BindGroupCacheKey> {
        while let Some(Reverse(entry)) = self.lru.pop() {
            if self
                .entries
                .get(&entry.key)
                .is_some_and(|current| current.last_seen == entry.last_seen)
            {
                return Some(entry.key);
            }
        }
        None
    }

    fn compact_lru_if_needed(&mut self) {
        let live = self.entries.len();
        if let Some(limit) = stale_lru_limit(live) {
            if self.lru.len() <= limit {
                return;
            }
        }
        self.lru.clear();
        self.lru.extend(self.entries.iter().map(|(key, entry)| {
            Reverse(BindGroupLruEntry {
                last_seen: entry.last_seen,
                key: key.clone(),
            })
        }));
    }
}

fn stale_lru_limit(live: usize) -> Option<usize> {
    live.checked_mul(4).map(|limit| limit.max(8))
}

impl BindGroupCache {
    fn lock_cache(&self) -> MutexGuard<'_, BindGroupCacheInner> {
        self.cache.lock().unwrap_or_else(|error| {
            tracing::error!(
                "Vyre WGPU bind-group cache lock was poisoned: {error}. Fix: discard the cache after a panic; continuing with recovered state."
            );
            error.into_inner()
        })
    }

    /// Create a bind-group cache with the default 256-entry cap.
    #[must_use]
    pub fn new() -> Self {
        Self::with_cap(BIND_GROUP_CACHE_CAP)
    }

    /// Create with an explicit cap (used by tests and consumers that
    /// want to size the LRU against known working-set bounds).
    #[must_use]
    pub fn with_cap(cap: usize) -> Self {
        Self {
            cache: Arc::new(Mutex::new(BindGroupCacheInner {
                entries: FxHashMap::default(),
                lru: BinaryHeap::new(),
                cap: cap.max(1),
                next_generation: 0,
            })),
            hits: Arc::new(AtomicUsize::new(0)),
            misses: Arc::new(AtomicUsize::new(0)),
            evictions: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Return a cached bind group or create one with `factory`.
    ///
    /// `layout_id` must uniquely identify the `wgpu::BindGroupLayout`
    /// (e.g. `Arc::as_ptr(layout).addr()`).
    /// `handles` must be in the same order as the `wgpu::BindGroupEntry`
    /// slice that the caller will pass to `create_bind_group` so that
    /// identical handle sets map to the same cache key.
    pub fn get_or_create(
        &self,
        layout_id: usize,
        handles: &[GpuBufferHandle],
        factory: impl FnOnce() -> wgpu::BindGroup,
    ) -> Arc<wgpu::BindGroup> {
        let Some(key_part_count) = handles.len().checked_mul(2) else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return Arc::new(factory());
        };
        let mut key_parts = SmallVec::with_capacity(key_part_count);
        for handle in handles {
            if !push_bind_group_handle_key(&mut key_parts, handle) {
                self.misses.fetch_add(1, Ordering::Relaxed);
                return Arc::new(factory());
            }
        }
        self.get_or_create_by_ids(layout_id, key_parts, factory)
    }

    pub(crate) fn get_or_create_by_ids(
        &self,
        layout_id: usize,
        handles: SmallVec<[u64; 16]>,
        factory: impl FnOnce() -> wgpu::BindGroup,
    ) -> Arc<wgpu::BindGroup> {
        let key = BindGroupCacheKey { layout_id, handles };
        {
            let mut cache = self.lock_cache();
            if let Some(existing) = cache.touch_existing(&key) {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return existing;
            }
        }
        let bg = Arc::new(factory());
        let mut cache = self.lock_cache();
        cache.insert_entry(key, Arc::clone(&bg));
        cache.evict_to_cap(|| {
            self.evictions.fetch_add(1, Ordering::Relaxed);
        });
        self.misses.fetch_add(1, Ordering::Relaxed);
        bg
    }

    pub(crate) fn get_by_ids(
        &self,
        layout_id: usize,
        handles: &[u64],
    ) -> Option<Arc<wgpu::BindGroup>> {
        let key = BindGroupCacheKey {
            layout_id,
            handles: SmallVec::from_slice(handles),
        };
        let mut cache = self.lock_cache();
        let existing = cache.touch_existing(&key)?;
        self.hits.fetch_add(1, Ordering::Relaxed);
        Some(existing)
    }

    pub(crate) fn insert_by_ids(
        &self,
        layout_id: usize,
        handles: &[u64],
        bind_group: wgpu::BindGroup,
    ) -> Arc<wgpu::BindGroup> {
        let key = BindGroupCacheKey {
            layout_id,
            handles: SmallVec::from_slice(handles),
        };
        let mut cache = self.lock_cache();
        if let Some(existing) = cache.touch_existing(&key) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return existing;
        }
        let bg = Arc::new(bind_group);
        cache.insert_entry(key, Arc::clone(&bg));
        cache.evict_to_cap(|| {
            self.evictions.fetch_add(1, Ordering::Relaxed);
        });
        self.misses.fetch_add(1, Ordering::Relaxed);
        bg
    }

    /// Return cache statistics for diagnostics and tests.
    #[must_use]
    pub fn stats(&self) -> BindGroupCacheStats {
        BindGroupCacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            entries: self.lock_cache().entries.len(),
        }
    }
}

/// Bind-group cache statistics for a compiled wgpu pipeline.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BindGroupCacheStats {
    /// Number of cached bind-group hits.
    pub hits: usize,
    /// Number of bind-group creations caused by cache misses.
    pub misses: usize,
    /// Number of cached bind-group entries evicted to honor the cap.
    pub evictions: usize,
    /// Current number of entries held.
    pub entries: usize,
}

// Inline: covers the crate-private `BindGroupCache::lock_cache` and the inner
// `entries` and `lru` fields, none of which an integration test can reach.
#[cfg(test)]
mod tests {
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
        let buffer_a =
            GpuBufferHandle::upload(device, queue, &[1u8; 4], wgpu::BufferUsages::STORAGE)
                .expect("Fix: upload of buffer a must succeed");
        let buffer_b =
            GpuBufferHandle::upload(device, queue, &[2u8; 4], wgpu::BufferUsages::STORAGE)
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
}
