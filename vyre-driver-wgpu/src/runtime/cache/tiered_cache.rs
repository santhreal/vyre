use crate::runtime::cache::lru::{AccessTracker, IntrusiveLru};
use rustc_hash::FxHashMap;

/// Metadata for a cached entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct CacheEntry {
    /// Unique identifier for the entry.
    pub key: u64,
    /// Size of the entry in bytes.
    pub size: u64,
    /// Index of the tier the entry currently resides in.
    pub tier: usize,
}

/// A single cache tier with a fixed capacity.
///
/// Carries its own recency LRU so eviction picks the coldest entry
/// within the tier in O(1) instead of scanning the global
/// `AccessTracker` looking for a key that happens to live in this
/// tier. Before 0.6 the scan was O(N) in the global tracker size  -
/// catastrophic when the cold key was far from the tier boundary.
#[non_exhaustive]
pub struct CacheTier {
    /// Human-readable name for the tier.
    pub name: String,
    /// Total capacity of the tier in bytes.
    pub capacity: u64,
    /// Currently used bytes in the tier.
    pub used: u64,
    pub(crate) entries: FxHashMap<u64, CacheEntry>,
    pub(crate) lru: IntrusiveLru<u64, ()>,
}

impl CacheTier {
    /// Create a new empty tier.
    #[inline]
    pub fn new(name: impl Into<String>, capacity: u64) -> Self {
        let name = name.into();
        match Self::try_new(name.clone(), capacity) {
            Ok(tier) => tier,
            Err(error) => {
                tracing::error!(
                    tier = %name,
                    capacity,
                    error = %error,
                    "wgpu cache tier LRU reservation failed; continuing with grow-on-use metadata"
                );
                Self {
                    name,
                    capacity,
                    used: 0,
                    entries: FxHashMap::default(),
                    lru: IntrusiveLru::with_reserved_capacity(0),
                }
            }
        }
    }

    /// Fallible version of [`Self::new`].
    ///
    /// # Errors
    ///
    /// Returns [`vyre_driver::BackendError`] if tier LRU metadata cannot be
    /// reserved.
    #[inline]
    pub fn try_new(
        name: impl Into<String>,
        capacity: u64,
    ) -> Result<Self, vyre_driver::BackendError> {
        Ok(Self {
            name: name.into(),
            capacity,
            used: 0,
            entries: FxHashMap::default(),
            lru: IntrusiveLru::try_with_reserved_capacity(1024)?,
        })
    }
}

/// Access statistics used by [`LruPolicy`] promotion decisions.
#[non_exhaustive]
pub struct AccessStats {
    /// Number of recorded accesses.
    pub frequency: u32,
    /// Monotonic tick of the last access. Higher = more recent.
    /// Compare two entries' ticks to determine relative recency in O(1).
    pub last_access: u64,
    /// Size of the entry in bytes.
    pub size: u64,
}

/// LRU eviction policy with frequency-based promotion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct LruPolicy {
    /// Minimum access frequency required for promotion.
    pub promote_threshold: u32,
}

impl LruPolicy {
    /// Default access threshold for promotion.
    pub const DEFAULT_THRESHOLD: u32 = 3;

    /// Create a new policy with the given promotion threshold.
    #[inline]
    pub fn new(promote_threshold: u32) -> Self {
        Self { promote_threshold }
    }
}

impl Default for LruPolicy {
    fn default() -> Self {
        Self::new(Self::DEFAULT_THRESHOLD)
    }
}

impl LruPolicy {
    fn should_promote(&self, _key: u64, stats: &AccessStats) -> bool {
        stats.frequency >= self.promote_threshold
    }

    fn eviction_candidate_per_tier(
        &self,
        _tier: usize,
        entries: &FxHashMap<u64, CacheEntry>,
        _tracker: &AccessTracker,
        tier_lru: &IntrusiveLru<u64, ()>,
    ) -> Option<u64> {
        // O(1) fast path. Walk the tier's own LRU from coldest
        // (tail) until we find a key that still lives in `entries`.
        // Entries and the LRU are mutated in lockstep by
        // TieredCache, so the first iterator step almost always
        // yields the right answer; the loop only runs when a
        // previous eviction race left a stale LRU entry.
        for (key, _) in tier_lru.iter_coldest() {
            if entries.contains_key(key) {
                return Some(*key);
            }
        }
        entries.keys().copied().next()
    }
}

/// Errors that can occur during cache operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CacheError {
    /// The requested key does not exist in the cache.
    KeyNotFound,
    /// The entry is too large to fit in any tier.
    EntryTooLarge,
    /// Tier byte accounting overflowed or underflowed.
    CapacityAccountingOverflow,
}

impl std::fmt::Display for CacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::KeyNotFound => write!(
                f,
                "Key not found in cache. Fix: verify the key was inserted before operating on it."
            ),
            Self::EntryTooLarge => write!(
                f,
                "Entry size exceeds the capacity of the largest tier. Fix: reduce the buffer size or increase the tier capacity."
            ),
            Self::CapacityAccountingOverflow => write!(
                f,
                "Tiered cache byte accounting overflowed. Fix: rebuild the cache or shard entries before continuing."
            ),
        }
    }
}

impl std::error::Error for CacheError {}

/// Generic tiered cache for GPU buffers.
///
/// Tracks hot/cold buffers using the built-in [`LruPolicy`].
/// This is the vyre primitive that helix builds inference intelligence on top of.
#[non_exhaustive]
pub struct TieredCache {
    pub(crate) tiers: Vec<CacheTier>,
    pub(crate) tracker: AccessTracker,
    pub(crate) policy: LruPolicy,
    /// O(1) key → tier index. Eliminates the linear tier scan in `get`.
    index: FxHashMap<u64, usize>,
}

impl TieredCache {
    /// Create a new cache with the given tiers and a default [`LruPolicy`].
    #[inline]
    pub fn new(tiers: Vec<CacheTier>) -> Self {
        match Self::try_new(tiers) {
            Ok(cache) => cache,
            Err(error) => {
                tracing::error!(
                    error = %error,
                    "wgpu tiered cache tracker reservation failed; continuing with grow-on-use metadata"
                );
                Self::with_policy(Vec::new(), LruPolicy::default())
            }
        }
    }

    /// Fallible version of [`Self::new`].
    ///
    /// # Errors
    ///
    /// Returns [`vyre_driver::BackendError`] if cache access metadata cannot be
    /// reserved.
    #[inline]
    pub fn try_new(tiers: Vec<CacheTier>) -> Result<Self, vyre_driver::BackendError> {
        Self::try_with_policy(tiers, LruPolicy::default())
    }
}

impl TieredCache {
    /// Create a new cache with a custom LRU policy.
    #[inline]
    pub fn with_policy(tiers: Vec<CacheTier>, policy: LruPolicy) -> Self {
        match Self::try_with_policy(tiers, policy) {
            Ok(cache) => cache,
            Err(error) => {
                tracing::error!(
                    error = %error,
                    "wgpu tiered cache tracker reservation failed; continuing with grow-on-use metadata"
                );
                Self {
                    tiers: Vec::new(),
                    tracker: AccessTracker::new(),
                    policy,
                    index: FxHashMap::default(),
                }
            }
        }
    }

    /// Fallible version of [`Self::with_policy`].
    ///
    /// # Errors
    ///
    /// Returns [`vyre_driver::BackendError`] if cache access metadata cannot be
    /// reserved.
    #[inline]
    pub fn try_with_policy(
        tiers: Vec<CacheTier>,
        policy: LruPolicy,
    ) -> Result<Self, vyre_driver::BackendError> {
        Ok(Self {
            tiers,
            tracker: AccessTracker::try_new()?,
            policy,
            index: FxHashMap::default(),
        })
    }

    /// Return a reference to the entry with the given key, if it exists.
    #[inline]
    pub fn get(&self, key: u64) -> Option<&CacheEntry> {
        let &tier = self.index.get(&key)?;
        self.tiers[tier].entries.get(&key)
    }

    /// Insert a new entry into the lowest tier that can fit it.
    ///
    /// # Errors
    ///
    /// Returns [`CacheError::EntryTooLarge`] when no tier can hold the entry.
    #[inline]
    pub fn insert(&mut self, key: u64, size: u64) -> Result<(), CacheError> {
        if self.get(key).is_some() {
            self.evict(key);
        }
        self.tracker.set_size(key, size);
        self.insert_into_tier(key, size, 0)
    }

    /// Record an access for the given key.
    #[inline]
    pub fn record_access(&mut self, key: u64) {
        if let Some(&tier_id) = self.index.get(&key) {
            self.tracker.record(key);
            // Touch the per-tier recency LRU so eviction keeps the
            // hot key at the head and the coldest key at the tail.
            self.tiers[tier_id].lru.touch(key);
        }
    }

    /// Promote the entry to the next faster tier if the policy allows it.
    ///
    /// # Errors
    ///
    /// Returns [`CacheError::KeyNotFound`] when the key does not exist.
    #[inline]
    pub fn promote(&mut self, key: u64) -> Result<(), CacheError> {
        let entry = self.get(key).copied().ok_or(CacheError::KeyNotFound)?;
        let stats = self.tracker.stats(key).ok_or(CacheError::KeyNotFound)?;
        if !self.policy.should_promote(key, &stats) {
            return Ok(());
        }
        let target = entry
            .tier
            .checked_add(1)
            .ok_or(CacheError::CapacityAccountingOverflow)?;
        if target >= self.tiers.len() {
            return Ok(());
        }
        let size = entry.size;
        self.remove_entry(key);
        self.move_into_tier(key, size, target, entry.tier)
    }

    /// Demote the entry to the next slower tier.
    ///
    /// # Errors
    ///
    /// Returns [`CacheError::KeyNotFound`] when the key does not exist.
    #[inline]
    pub fn demote(&mut self, key: u64) -> Result<(), CacheError> {
        let entry = self.get(key).copied().ok_or(CacheError::KeyNotFound)?;
        if entry.tier == 0 {
            return Ok(());
        }
        let target = entry.tier - 1;
        let size = entry.size;
        self.remove_entry(key);
        self.move_into_tier(key, size, target, entry.tier)
    }

    fn insert_into_tier(
        &mut self,
        key: u64,
        size: u64,
        mut start: usize,
    ) -> Result<(), CacheError> {
        while start < self.tiers.len() {
            if size > self.tiers[start].capacity {
                start += 1;
                continue;
            }
            if self.make_room(start, size) {
                self.tiers[start].used = checked_tier_used_add(self.tiers[start].used, size)?;
                self.tiers[start].entries.insert(
                    key,
                    CacheEntry {
                        key,
                        size,
                        tier: start,
                    },
                );
                // Register the key in the tier's per-tier LRU so the
                // fast-path eviction can pop its tail in O(1).
                self.tiers[start].lru.ensure(key);
                self.tiers[start].lru.touch(key);
                self.index.insert(key, start);
                return Ok(());
            }
            start += 1;
        }
        Err(CacheError::EntryTooLarge)
    }

    fn move_into_tier(
        &mut self,
        key: u64,
        size: u64,
        target: usize,
        fallback: usize,
    ) -> Result<(), CacheError> {
        if self.make_room(target, size) {
            self.tiers[target].used = checked_tier_used_add(self.tiers[target].used, size)?;
            self.tiers[target].entries.insert(
                key,
                CacheEntry {
                    key,
                    size,
                    tier: target,
                },
            );
            self.tiers[target].lru.ensure(key);
            self.tiers[target].lru.touch(key);
            self.index.insert(key, target);
            Ok(())
        } else {
            self.insert_into_tier(key, size, fallback)
        }
    }

    fn make_room(&mut self, tier: usize, size: u64) -> bool {
        loop {
            let used = self.tiers[tier].used;
            let cap = self.tiers[tier].capacity;
            if used.checked_add(size).is_some_and(|total| total <= cap) {
                return true;
            }
            // O(1) fast-path eviction using the tier's own recency
            // LRU. The default `TierPolicy::eviction_candidate_per_tier`
            // delegates to the slow path so custom policies still work;
            // `LruPolicy` overrides it to pop the tier LRU tail
            // directly.
            let candidate = {
                let tier_ref = &self.tiers[tier];
                self.policy.eviction_candidate_per_tier(
                    tier,
                    &tier_ref.entries,
                    &self.tracker,
                    &tier_ref.lru,
                )
            };
            if let Some(key) = candidate {
                self.evict_from_tier(key, tier);
            } else {
                return false;
            }
        }
    }

    fn remove_entry(&mut self, key: u64) -> Option<CacheEntry> {
        let &tier_id = self.index.get(&key)?;
        let tier = &mut self.tiers[tier_id];
        let entry = tier.entries.remove(&key)?;
        tier.lru.remove(&key);
        debit_tier_used(tier, entry.size);
        self.index.remove(&key);
        Some(entry)
    }

    fn evict(&mut self, key: u64) -> Option<CacheEntry> {
        let &tier_id = self.index.get(&key)?;
        let tier = &mut self.tiers[tier_id];
        let entry = tier.entries.remove(&key)?;
        tier.lru.remove(&key);
        debit_tier_used(tier, entry.size);
        self.index.remove(&key);
        self.tracker.remove(key);
        Some(entry)
    }

    /// Find and remove the coldest entry from the cache.
    ///
    /// This follows the LRU policy across all tiers, starting from the
    /// lowest (coldest) tier. Returns the key of the evicted entry.
    pub fn evict_coldest(&mut self) -> Option<u64> {
        for (tier_idx, tier) in self.tiers.iter().enumerate() {
            if let Some(key) = self.policy.eviction_candidate_per_tier(
                tier_idx,
                &tier.entries,
                &self.tracker,
                &tier.lru,
            ) {
                self.evict_from_tier(key, tier_idx);
                return Some(key);
            }
        }
        None
    }

    fn evict_from_tier(&mut self, key: u64, tier: usize) -> Option<CacheEntry> {
        let tier = &mut self.tiers[tier];
        let entry = tier.entries.remove(&key)?;
        tier.lru.remove(&key);
        debit_tier_used(tier, entry.size);
        self.index.remove(&key);
        self.tracker.remove(key);
        Some(entry)
    }
}

fn checked_tier_used_add(used: u64, size: u64) -> Result<u64, CacheError> {
    used.checked_add(size)
        .ok_or(CacheError::CapacityAccountingOverflow)
}

fn debit_tier_used(tier: &mut CacheTier, size: u64) {
    match tier.used.checked_sub(size) {
        Some(used) => {
            tier.used = used;
        }
        None => {
            tracing::error!(
                tier = %tier.name,
                used = tier.used,
                removed_size = size,
                "tiered cache byte accounting underflowed; repairing from live entries. Fix: investigate mismatched cache tier metadata."
            );
            tier.used = recompute_tier_used(tier);
        }
    }
}

fn recompute_tier_used(tier: &CacheTier) -> u64 {
    let mut total = 0_u64;
    for entry in tier.entries.values() {
        total = match total.checked_add(entry.size) {
            Some(next) => next,
            None => {
                tracing::error!(
                    tier = %tier.name,
                    "tiered cache byte accounting overflowed while repairing from live entries; pinning used bytes to u64::MAX."
                );
                return u64::MAX;
            }
        };
    }
    total
}

// Inline: covers the crate-private `TieredCache::tiers` byte accounting and the
// crate-private `tiered_cache` module itself, neither of which an integration test
// can reach.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiered_cache_repairs_used_bytes_after_underflow_instead_of_panicking() {
        let mut cache = TieredCache::new(vec![CacheTier::new("gpu", 128)]);
        cache.insert(1, 64).expect("Fix: test insert must fit");
        cache.tiers[0].used = 0;

        let removed = cache
            .evict(1)
            .expect("Fix: corrupted entry should still evict");

        assert_eq!(removed.size, 64);
        assert_eq!(cache.tiers[0].used, 0);
        assert!(cache.get(1).is_none());
    }

    #[test]
    fn get_returns_entry_after_insert() {
        let mut cache = TieredCache::new(vec![CacheTier::new("L1", 1024)]);
        cache
            .insert(1, 100)
            .expect("Fix: an entry that fits must insert");
        let entry = cache.get(1).expect("Fix: an inserted key must be gettable");
        assert_eq!(entry.key, 1);
        assert_eq!(entry.size, 100);
        assert_eq!(entry.tier, 0);
    }

    #[test]
    fn get_missing_returns_none() {
        let cache = TieredCache::new(vec![CacheTier::new("L1", 1024)]);
        assert!(cache.get(99).is_none());
    }

    #[test]
    fn insert_replaces_existing_key() {
        let mut cache = TieredCache::new(vec![CacheTier::new("L1", 1024)]);
        cache
            .insert(1, 100)
            .expect("Fix: an entry that fits must insert");
        cache
            .insert(1, 200)
            .expect("Fix: replacing a key must insert");
        let entry = cache.get(1).expect("Fix: a replaced key must be gettable");
        assert_eq!(entry.size, 200);
    }

    #[test]
    fn promote_moves_to_higher_tier() {
        let mut cache =
            TieredCache::new(vec![CacheTier::new("L1", 1024), CacheTier::new("L2", 1024)]);
        cache
            .insert(1, 100)
            .expect("Fix: an entry that fits must insert");
        for _ in 0..LruPolicy::DEFAULT_THRESHOLD {
            cache.record_access(1);
        }
        cache.promote(1).expect("Fix: a hot key must promote");
        let entry = cache.get(1).expect("Fix: a promoted key must be gettable");
        assert_eq!(entry.tier, 1);
    }

    #[test]
    fn demote_moves_to_lower_tier() {
        let mut cache =
            TieredCache::new(vec![CacheTier::new("L1", 1024), CacheTier::new("L2", 1024)]);
        cache
            .insert(1, 100)
            .expect("Fix: an entry that fits must insert");
        for _ in 0..LruPolicy::DEFAULT_THRESHOLD {
            cache.record_access(1);
        }
        cache.promote(1).expect("Fix: a hot key must promote");
        cache.demote(1).expect("Fix: a promoted key must demote");
        let entry = cache.get(1).expect("Fix: a demoted key must be gettable");
        assert_eq!(entry.tier, 0);
    }

    #[test]
    fn make_room_evicts_coldest() {
        let mut cache = TieredCache::new(vec![CacheTier::new("L1", 200)]);
        cache
            .insert(1, 100)
            .expect("Fix: an entry that fits must insert");
        cache
            .insert(2, 100)
            .expect("Fix: an entry that fits must insert");
        // Touch key 1 so key 2 is coldest.
        cache.record_access(1);
        cache
            .insert(3, 100)
            .expect("Fix: an insert that needs room must evict and insert");
        assert!(cache.get(1).is_some());
        assert!(cache.get(2).is_none());
        assert!(cache.get(3).is_some());
    }

    #[test]
    fn stats_returns_last_access_not_rank() {
        let mut tracker = AccessTracker::new();
        tracker.set_size(1, 100);
        tracker.record(1);
        tracker.record(2);
        tracker.record(1);
        let stats1 = tracker
            .stats(1)
            .expect("Fix: a recorded key must have stats");
        let stats2 = tracker
            .stats(2)
            .expect("Fix: a recorded key must have stats");
        // A higher tick is more recent.
        assert!(stats1.last_access > stats2.last_access);
        assert_eq!(stats1.frequency, 2);
        assert_eq!(stats2.frequency, 1);
    }

    #[test]
    fn promote_without_eviction_keeps_both_keys() {
        let mut cache =
            TieredCache::new(vec![CacheTier::new("L1", 200), CacheTier::new("L2", 200)]);
        cache
            .insert(1, 100)
            .expect("Fix: an entry that fits must insert");
        cache
            .insert(2, 100)
            .expect("Fix: an entry that fits must insert");
        for _ in 0..LruPolicy::DEFAULT_THRESHOLD {
            cache.record_access(1);
        }
        cache.promote(1).expect("Fix: a hot key must promote");
        assert!(cache.get(1).is_some());
        assert!(cache.get(2).is_some());
    }

    #[test]
    fn an_insert_past_the_tier_budget_evicts_the_coldest_key() {
        let mut cache = TieredCache::new(vec![CacheTier::new("L1", 250)]);
        cache
            .insert(1, 100)
            .expect("Fix: an entry that fits must insert");
        cache
            .insert(2, 100)
            .expect("Fix: 200 bytes fit in a 250-byte tier");
        // Touch 1 so it is hottest and 2 is the eviction candidate.
        cache.record_access(1);
        cache
            .insert(3, 100)
            .expect("Fix: an insert that needs room must evict and insert");
        assert!(cache.get(1).is_some());
        assert!(cache.get(2).is_none());
        assert!(cache.get(3).is_some());
    }

    /// A replacement is an eviction followed by an insert, so the entry it
    /// replaced has to give its bytes back. Charging both sizes leaves the tier
    /// believing it holds 300 bytes of a 1024-byte budget when it holds 200, and
    /// every later eviction decision is taken against that wrong number.
    #[test]
    fn a_replacement_releases_the_bytes_of_the_entry_it_replaced() {
        let mut cache = TieredCache::new(vec![CacheTier::new("L1", 1024)]);
        cache
            .insert(1, 100)
            .expect("Fix: an entry that fits must insert");
        cache
            .insert(1, 200)
            .expect("Fix: replacing a key must insert");
        let entry = cache.get(1).expect("Fix: a replaced key must be gettable");
        assert_eq!(entry.size, 200);
        assert_eq!(
            cache.tiers[0].entries.len(),
            1,
            "Fix: a replacement must leave one entry under the key, not two."
        );
        assert_eq!(
            cache.tiers[0].used, 200,
            "Fix: a replacement must debit the entry it evicted; 300 means the replaced 100 bytes are still charged to the tier."
        );
    }

    /// The hard promote path: the target tier is full, so promoting evicts that
    /// tier's coldest key. `promote` reaches it through `move_into_tier`, which
    /// asks `make_room` first and falls back to the source tier only when room
    /// cannot be made, so an eviction that did not happen is a promote that
    /// silently stayed put, and an eviction that did not release its bytes is a
    /// tier that can never fit another entry.
    #[test]
    fn a_promote_into_a_full_tier_evicts_that_tiers_coldest_key() {
        let mut cache =
            TieredCache::new(vec![CacheTier::new("L1", 400), CacheTier::new("L2", 100)]);
        cache
            .insert(1, 100)
            .expect("Fix: an entry that fits must insert");
        cache
            .insert(2, 100)
            .expect("Fix: an entry that fits must insert");
        for _ in 0..LruPolicy::DEFAULT_THRESHOLD {
            cache.record_access(2);
        }
        cache.promote(2).expect("Fix: a hot key must promote");
        assert_eq!(
            cache
                .get(2)
                .expect("Fix: a promoted key must be gettable")
                .tier,
            1,
            "Fix: the second tier has to be full before the eviction path is reached."
        );

        for _ in 0..LruPolicy::DEFAULT_THRESHOLD {
            cache.record_access(1);
        }
        cache.promote(1).expect("Fix: a hot key must promote");

        assert_eq!(
            cache
                .get(1)
                .expect("Fix: a promoted key must be gettable")
                .tier,
            1,
            "Fix: a promote into a full tier must evict and move, not leave the key where it was."
        );
        assert!(
            cache.get(2).is_none(),
            "Fix: the key a promote evicted must be unreachable through lookup, not merely uncharged."
        );
        assert_eq!(
            cache.tiers[1].used, 100,
            "Fix: the evicted entry's bytes must be released, or the tier can never fit another promote."
        );
    }
}
