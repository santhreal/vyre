//! [`InMemoryPipelineCache`]  -  sharded zero-persistence cache. The hot
//! path for in-process pipeline reuse.

use std::sync::{Arc, Mutex, MutexGuard};

use rustc_hash::FxHashMap;

use super::fingerprint::PipelineFingerprint;
use super::metrics::{PipelineCacheCounters, PipelineCacheMetrics};
use super::store::PipelineCacheStore;

/// In-memory pipeline cache  -  zero-persistence, zero-network, sharded
/// `FxHashMap`s behind mutexes so concurrent `get`/`put` on different
/// fingerprints rarely contend (VYRE_RUNTIME / PERF hot-cache audit).
#[derive(Debug)]
pub struct InMemoryPipelineCache {
    shards: [Mutex<InMemoryCacheShard>; Self::SHARD_COUNT],
    max_entries_per_shard: usize,
    max_bytes_per_shard: usize,
    metrics: PipelineCacheCounters,
}

impl InMemoryPipelineCache {
    pub(super) const SHARD_COUNT: usize = 256;
    pub(super) const MAX_ENTRIES_PER_SHARD: usize = 256;
    pub(super) const MAX_BYTES_PER_SHARD: usize = 16 * 1024 * 1024;

    #[inline]
    fn shard_index(fp: &PipelineFingerprint) -> usize {
        usize::from(fp.0[0]) % Self::SHARD_COUNT
    }

    /// Lock one cache shard under [`RecoveryClass::RestartableFromCanonicalInput`].
    ///
    /// If a previous thread panicked while holding the shard lock, the shard is
    /// reset to an empty valid state so no half-written entry can be served,
    /// and the caller restarts cleanly from canonical input.
    ///
    /// [`RecoveryClass::RestartableFromCanonicalInput`]: vyre_foundation::RecoveryClass::RestartableFromCanonicalInput
    fn lock_shard(shard: &Mutex<InMemoryCacheShard>) -> MutexGuard<'_, InMemoryCacheShard> {
        vyre_foundation::failure_domain::govern_mutex_restartable(
            shard,
            "runtime in-memory pipeline cache",
            "one cache shard",
            |shard| {
                shard.entries.clear();
                shard.bytes = 0;
            },
        )
    }

    /// Construct an empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct an empty cache with explicit per-shard entry and byte budgets.
    ///
    /// A zero entry budget or zero byte budget creates a disabled cache that
    /// accepts `put` calls but retains no artifacts.
    #[must_use]
    pub fn with_limits(max_entries_per_shard: usize, max_bytes_per_shard: usize) -> Self {
        Self {
            shards: std::array::from_fn(|_| Mutex::new(InMemoryCacheShard::default())),
            max_entries_per_shard,
            max_bytes_per_shard,
            metrics: PipelineCacheCounters::default(),
        }
    }

    /// Current entry count. Thread-safe snapshot.
    pub fn len(&self) -> usize {
        self.shards
            .iter()
            .map(|s| Self::lock_shard(s).entries.len())
            .sum()
    }

    /// Current cached artifact bytes. Thread-safe snapshot.
    pub fn cached_bytes(&self) -> usize {
        self.shards.iter().map(|s| Self::lock_shard(s).bytes).sum()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.shards
            .iter()
            .all(|s| Self::lock_shard(s).entries.is_empty())
    }
}

impl Default for InMemoryPipelineCache {
    fn default() -> Self {
        Self::with_limits(Self::MAX_ENTRIES_PER_SHARD, Self::MAX_BYTES_PER_SHARD)
    }
}

#[derive(Debug, Default)]
struct InMemoryCacheShard {
    entries: FxHashMap<PipelineFingerprint, InMemoryCacheEntry>,
    bytes: usize,
    clock: u64,
}

impl InMemoryCacheShard {
    fn next_tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// Evict least-recently-used entries until both budgets hold. Returns the
    /// evicted entry count and byte total.
    fn evict_to_limits(&mut self, max_entries: usize, max_bytes: usize) -> (u64, u64) {
        let (mut entries, mut bytes) = (0_u64, 0_u64);
        while self.entries.len() > max_entries || self.bytes > max_bytes {
            let Some(victim) = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(fp, _)| *fp)
            else {
                self.bytes = 0;
                break;
            };
            if let Some(removed) = self.entries.remove(&victim) {
                self.bytes = self.bytes.saturating_sub(removed.bytes);
                entries += 1;
                bytes += removed.bytes as u64;
            }
        }
        (entries, bytes)
    }
}

#[derive(Debug)]
struct InMemoryCacheEntry {
    artifact: Arc<Vec<u8>>,
    bytes: usize,
    last_used: u64,
}

impl PipelineCacheStore for InMemoryPipelineCache {
    fn get(&self, fp: &PipelineFingerprint) -> Option<Vec<u8>> {
        self.get_arc(fp).map(|artifact| (*artifact).clone())
    }

    /// V7-PERF-009: zero-clone hot-path lookup. The cache already stores
    /// payloads behind `Arc<Vec<u8>>`, so a hit is one refcount bump.
    fn get_arc(&self, fp: &PipelineFingerprint) -> Option<Arc<Vec<u8>>> {
        PipelineCacheCounters::add(&self.metrics.lookups, 1);
        let i = Self::shard_index(fp);
        let mut shard = Self::lock_shard(&self.shards[i]);
        let tick = shard.next_tick();
        let Some(entry) = shard.entries.get_mut(fp) else {
            PipelineCacheCounters::add(&self.metrics.misses, 1);
            return None;
        };
        entry.last_used = tick;
        PipelineCacheCounters::add(&self.metrics.hits, 1);
        Some(Arc::clone(&entry.artifact))
    }

    fn put(&self, fp: PipelineFingerprint, artifact: Vec<u8>) {
        let i = Self::shard_index(&fp);
        let mut shard = Self::lock_shard(&self.shards[i]);
        let bytes = artifact.len();
        if self.max_entries_per_shard == 0
            || self.max_bytes_per_shard == 0
            || bytes > self.max_bytes_per_shard
        {
            PipelineCacheCounters::add(&self.metrics.rejected_puts, 1);
            if let Some(removed) = shard.entries.remove(&fp) {
                shard.bytes = shard.bytes.saturating_sub(removed.bytes);
                PipelineCacheCounters::add(&self.metrics.evictions, 1);
                PipelineCacheCounters::add(&self.metrics.evicted_bytes, removed.bytes as u64);
            }
            return;
        }

        if let Some(existing) = shard.entries.remove(&fp) {
            shard.bytes = shard.bytes.saturating_sub(existing.bytes);
        }
        let tick = shard.next_tick();
        shard.bytes += bytes;
        shard.entries.insert(
            fp,
            InMemoryCacheEntry {
                artifact: Arc::new(artifact),
                bytes,
                last_used: tick,
            },
        );
        PipelineCacheCounters::add(&self.metrics.puts, 1);
        let (evicted, evicted_bytes) =
            shard.evict_to_limits(self.max_entries_per_shard, self.max_bytes_per_shard);
        if evicted > 0 {
            PipelineCacheCounters::add(&self.metrics.evictions, evicted);
            PipelineCacheCounters::add(&self.metrics.evicted_bytes, evicted_bytes);
        }
    }

    fn metrics(&self) -> PipelineCacheMetrics {
        self.metrics
            .snapshot(self.cached_bytes() as u64, self.len() as u64)
    }
}

// Inline: covers `metrics`, which no integration test can name.
#[cfg(test)]
mod tests {
    use super::*;
    use vyre_test_support::artifact_fixtures::tiny_artifact;

    #[test]
    fn in_memory_cache_roundtrip() {
        let cache = InMemoryPipelineCache::new();
        let fp = PipelineFingerprint::of(&tiny_artifact());
        assert!(cache.get(&fp).is_none());
        cache.put(fp, b"target-bytes".to_vec());
        assert_eq!(cache.get(&fp).unwrap(), b"target-bytes".to_vec());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn in_memory_cache_caps_each_shard() {
        let cache = InMemoryPipelineCache::new();
        for i in 0..(InMemoryPipelineCache::MAX_ENTRIES_PER_SHARD + 17) {
            let mut bytes = [0_u8; 32];
            bytes[1..9].copy_from_slice(&(i as u64).to_le_bytes());
            cache.put(PipelineFingerprint(bytes), vec![i as u8]);
        }
        assert_eq!(cache.len(), InMemoryPipelineCache::MAX_ENTRIES_PER_SHARD);
    }

    #[test]
    fn in_memory_cache_evicts_least_recently_used_entry() {
        let cache = InMemoryPipelineCache::with_limits(2, 1024);
        let a = PipelineFingerprint([0; 32]);
        let mut b_bytes = [0; 32];
        b_bytes[1] = 1;
        let b = PipelineFingerprint(b_bytes);
        let mut c_bytes = [0; 32];
        c_bytes[1] = 2;
        let c = PipelineFingerprint(c_bytes);

        cache.put(a, b"a".to_vec());
        cache.put(b, b"b".to_vec());
        assert_eq!(cache.get(&a).unwrap(), b"a".to_vec());
        cache.put(c, b"c".to_vec());

        assert_eq!(cache.get(&a).unwrap(), b"a".to_vec());
        assert!(cache.get(&b).is_none());
        assert_eq!(cache.get(&c).unwrap(), b"c".to_vec());
    }

    #[test]
    fn in_memory_cache_enforces_byte_budget() {
        let cache = InMemoryPipelineCache::with_limits(8, 10);
        let a = PipelineFingerprint([0; 32]);
        let mut b_bytes = [0; 32];
        b_bytes[1] = 1;
        let b = PipelineFingerprint(b_bytes);
        let mut too_large_bytes = [0; 32];
        too_large_bytes[1] = 2;
        let too_large = PipelineFingerprint(too_large_bytes);

        cache.put(a, vec![1; 6]);
        cache.put(b, vec![2; 6]);
        assert!(cache.get(&a).is_none());
        assert_eq!(cache.get(&b).unwrap(), vec![2; 6]);
        assert_eq!(cache.cached_bytes(), 6);

        cache.put(too_large, vec![3; 11]);
        assert!(cache.get(&too_large).is_none());
        assert_eq!(cache.cached_bytes(), 6);
    }

    #[test]
    fn in_memory_cache_metrics_track_hits_misses_and_evictions() {
        let cache = InMemoryPipelineCache::with_limits(1, 8);
        let a = PipelineFingerprint([0; 32]);
        let mut b_bytes = [0; 32];
        b_bytes[1] = 1;
        let b = PipelineFingerprint(b_bytes);

        assert!(cache.get(&a).is_none());
        cache.put(a, vec![1; 4]);
        assert!(cache.get(&a).is_some());
        cache.put(b, vec![2; 4]);

        let metrics = cache.metrics();
        assert_eq!(metrics.lookups, 2);
        assert_eq!(metrics.hits, 1);
        assert_eq!(metrics.misses, 1);
        assert_eq!(metrics.puts, 2);
        assert_eq!(metrics.evictions, 1);
        assert_eq!(metrics.cached_bytes, 4);
        assert_eq!(metrics.entries, 1);
        assert_eq!(metrics.hit_rate_ppm(), 500_000);
    }

    #[test]
    fn rejected_oversize_put_evicts_the_replaced_entry() {
        let cache = InMemoryPipelineCache::with_limits(8, 8);
        let fp = PipelineFingerprint([0; 32]);

        cache.put(fp, vec![1; 4]);
        cache.put(fp, vec![2; 9]);

        assert!(cache.get(&fp).is_none());
        let metrics = cache.metrics();
        assert_eq!(metrics.rejected_puts, 1);
        assert_eq!(metrics.evictions, 1);
        assert_eq!(metrics.evicted_bytes, 4);
        assert_eq!(metrics.cached_bytes, 0);
    }

    #[test]
    fn poisoned_cache_shard_resets_cleanly_under_restartable_recovery() {
        let cache = Arc::new(InMemoryPipelineCache::new());
        let fp = PipelineFingerprint([0; 32]);
        cache.put(fp, vec![1, 2, 3, 4]);
        assert_eq!(cache.len(), 1);

        let poisoned = Arc::clone(&cache);
        let _ = std::thread::spawn(move || {
            let _guard = InMemoryPipelineCache::lock_shard(&poisoned.shards[0]);
            panic!("poison in-memory pipeline cache shard");
        })
        .join();

        // Under RestartableFromCanonicalInput, accessing the poisoned shard resets it cleanly to empty
        assert_eq!(
            cache.len(),
            0,
            "poisoned shard must reset to empty valid state"
        );
        assert_eq!(cache.get(&fp), None);

        // Subsequent put succeeds normally
        cache.put(fp, vec![5, 6, 7, 8]);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(&fp), Some(vec![5, 6, 7, 8]));
    }
}

impl crate::StateOwnerRecovery for InMemoryPipelineCache {
    fn failure_domain(&self) -> crate::FailureDomain {
        crate::FailureDomain::MemoryState
    }

    fn recovery_class(&self) -> crate::RecoveryClass {
        crate::RecoveryClass::RestartableFromCanonicalInput
    }
}
