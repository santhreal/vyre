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

#[cfg(test)]
mod tests;
