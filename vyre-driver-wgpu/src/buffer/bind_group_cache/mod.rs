//! Bounded LRU cache for wgpu bind groups.

mod entry_store;
#[cfg(test)]
mod tests;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use smallvec::SmallVec;

use self::entry_store::{push_bind_group_handle_key, BindGroupCacheInner, BindGroupCacheKey};
use super::handle::GpuBufferHandle;

/// Default cap for the [`BindGroupCache`] LRU.
const BIND_GROUP_CACHE_CAP: usize = 256;

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
            cache: Arc::new(Mutex::new(BindGroupCacheInner::new(cap))),
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
