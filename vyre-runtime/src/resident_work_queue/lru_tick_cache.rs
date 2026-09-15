//! Bounded least-recently-used map shared by the megakernel caches.
//!
//! One tick counter orders accesses. Reading a key stamps it with the next
//! tick, so an insert past capacity evicts the entry with the oldest stamp and
//! never the key just inserted. The counter saturates rather than wraps: at
//! `u64::MAX` every stamp resets to zero and ordering restarts from the
//! entries still resident.

use rustc_hash::FxHashMap;
use std::hash::Hash;

struct Entry<V> {
    value: V,
    last_seen: u64,
}

pub(super) struct LruTickCache<K, V> {
    entries: FxHashMap<K, Entry<V>>,
    capacity: usize,
    clock: u64,
}

impl<K: Copy + Eq + Hash, V> LruTickCache<K, V> {
    pub(super) fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
            capacity,
            clock: 0,
        }
    }

    /// Borrow the value for `key` and stamp it as most recently used.
    pub(super) fn get(&mut self, key: &K) -> Option<&V> {
        self.restart_clock_at_saturation();
        let entry = self.entries.get_mut(key)?;
        self.clock += 1;
        entry.last_seen = self.clock;
        Some(&entry.value)
    }

    /// Store `value` under `key`, evicting the least recently used entries
    /// until the map is back within capacity.
    pub(super) fn insert(&mut self, key: K, value: V) {
        let last_seen = self.next_tick();
        self.entries.insert(key, Entry { value, last_seen });
        while self.entries.len() > self.capacity {
            let Some(evicted) = self
                .entries
                .iter()
                .filter(|(candidate, _)| **candidate != key)
                .min_by_key(|(_, entry)| entry.last_seen)
                .map(|(candidate, _)| *candidate)
            else {
                break;
            };
            self.entries.remove(&evicted);
        }
    }

    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(super) fn clear(&mut self) {
        self.entries.clear();
        self.clock = 0;
    }

    fn next_tick(&mut self) -> u64 {
        self.restart_clock_at_saturation();
        self.clock += 1;
        self.clock
    }

    fn restart_clock_at_saturation(&mut self) {
        if self.clock == u64::MAX {
            self.clock = 0;
            for entry in self.entries.values_mut() {
                entry.last_seen = 0;
            }
        }
    }
}
