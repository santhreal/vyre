//! Cached bind groups and the LRU order that evicts them.

use std::cmp::{Ordering as CmpOrdering, Reverse};
use std::collections::BinaryHeap;
use std::sync::Arc;

use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::buffer::handle::{aligned_len_u64, GpuBufferHandle};

/// Inline storage for bind-group cache keys: typical shaders use few bindings;
/// `SmallVec` avoids a heap `Vec` on most `get_or_create` calls.
pub(super) type BindGroupHandleKey = SmallVec<[u64; 16]>;

pub(super) struct BindGroupCacheInner {
    pub(super) entries: FxHashMap<BindGroupCacheKey, BindGroupCacheEntry>,
    pub(super) lru: BinaryHeap<Reverse<BindGroupLruEntry>>,
    pub(super) cap: usize,
    pub(super) next_generation: u64,
}

pub(super) struct BindGroupCacheEntry {
    pub(super) bind_group: Arc<wgpu::BindGroup>,
    pub(super) last_seen: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct BindGroupLruEntry {
    pub(super) last_seen: u64,
    pub(super) key: BindGroupCacheKey,
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

pub(super) fn push_bind_group_handle_key(
    key: &mut BindGroupHandleKey,
    handle: &GpuBufferHandle,
) -> bool {
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
pub(super) struct BindGroupCacheKey {
    pub(super) layout_id: usize,
    pub(super) handles: BindGroupHandleKey,
}

impl BindGroupCacheInner {
    pub(super) fn new(cap: usize) -> Self {
        Self {
            entries: FxHashMap::default(),
            lru: BinaryHeap::new(),
            cap: cap.max(1),
            next_generation: 0,
        }
    }

    fn next_lru_generation(&mut self) -> u64 {
        let generation = self.next_generation;
        self.next_generation = self.next_generation.wrapping_add(1);
        generation
    }

    pub(super) fn touch_existing(
        &mut self,
        key: &BindGroupCacheKey,
    ) -> Option<Arc<wgpu::BindGroup>> {
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

    pub(super) fn insert_entry(
        &mut self,
        key: BindGroupCacheKey,
        bind_group: Arc<wgpu::BindGroup>,
    ) {
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

    pub(super) fn evict_to_cap(&mut self, mut on_evict: impl FnMut()) {
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
