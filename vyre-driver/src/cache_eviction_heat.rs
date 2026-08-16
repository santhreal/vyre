//! N5 substrate: spec-cache eviction policy with frequency × recency
//! heat decay.
//!
//! F1/F3 cache compiled pipelines by `SpecCacheKey` but never evict.
//! Long-running daemons that scan many repositories in sequence
//! accumulate dead entries that pin VRAM-resident
//! pipelines. This module owns the *decision*: given a list of
//! cache entry stats and a capacity, return which entries to drop.
//!
//! The score is `hit_count / (1 + age_seconds / DECAY_HALF_LIFE_S)`  -
//! a hot, recent entry stays; a cold, old entry leaves. Pure
//! arithmetic; the actual cache surgery lives in the F1/F3 cache
//! modules and is the consumer's responsibility.

/// Half-life (seconds) for the heat decay term. Entries older than
/// this lose half their hit-count weight; doubled, lose three
/// quarters; etc. Tuned for scan workloads where a "warm" entry is
/// one used in the last few minutes of a long sweep.
pub const DECAY_HALF_LIFE_S: f64 = 300.0;

/// Per-entry stats the eviction policy needs. Caller (the F1/F3
/// cache layer) keeps these alongside each entry and passes a
/// snapshot when capacity pressure triggers.
#[derive(Debug, Clone, Copy)]
pub struct CacheEntryStats {
    /// Stable identifier for the entry (cache slot index, hash,
    /// SpecCacheKey index, etc). Pure pass-through  -  the policy
    /// only uses it to name which entries to evict.
    pub id: u64,
    /// Total hits since the entry was inserted.
    pub hit_count: u32,
    /// Wall-clock time (seconds since epoch or any monotonic clock)
    /// the entry was last hit. Same clock reference as
    /// `current_time_s`.
    pub last_hit_time_s: f64,
}

impl CacheEntryStats {
    /// Heat score: high = keep, low = evict. Combines frequency
    /// (hit_count) with recency via exponential half-life decay.
    #[must_use]
    pub fn heat(&self, current_time_s: f64) -> f64 {
        if !current_time_s.is_finite() || !self.last_hit_time_s.is_finite() {
            return 0.0;
        }
        let age = (current_time_s - self.last_hit_time_s).max(0.0);
        let decay_factor = 0.5_f64.powf(age / DECAY_HALF_LIFE_S);
        let heat = f64::from(self.hit_count) * decay_factor;
        if heat.is_finite() {
            heat
        } else {
            0.0
        }
    }
}

/// Decide which entry IDs to evict given a fixed capacity. Returns
/// the IDs in eviction order (lowest heat first); caller drops
/// until under capacity.
///
/// Entries with identical heat (e.g. two cold entries with the same
/// `hit_count` and `last_hit_time_s`) are evicted in input order
/// for determinism  -  bench reproducibility matters here.
#[must_use]
pub fn entries_to_evict(
    entries: &[CacheEntryStats],
    capacity: usize,
    current_time_s: f64,
) -> Vec<u64> {
    try_entries_to_evict(entries, capacity, current_time_s).unwrap_or_default()
}

/// Fallible variant of [`entries_to_evict`] for daemon/cache paths that must
/// report allocator pressure instead of panicking.
///
/// # Errors
///
/// Returns an actionable error when ranking/result staging cannot reserve.
pub fn try_entries_to_evict(
    entries: &[CacheEntryStats],
    capacity: usize,
    current_time_s: f64,
) -> Result<Vec<u64>, String> {
    if entries.len() <= capacity {
        return Ok(Vec::new());
    }
    let mut ranked: Vec<(usize, &CacheEntryStats, f64)> = Vec::new();
    crate::allocation::try_reserve_vec_to_capacity(&mut ranked, entries.len()).map_err(|error| {
        format!(
            "cache eviction heat ranking could not reserve {} entry slot(s): {error}. Fix: shard the pipeline cache eviction batch.",
            entries.len()
        )
    })?;
    ranked.extend(
        entries
            .iter()
            .enumerate()
            .map(|(idx, e)| (idx, e, e.heat(current_time_s))),
    );
    let compare = |a: &(usize, &CacheEntryStats, f64), b: &(usize, &CacheEntryStats, f64)| {
        a.2.total_cmp(&b.2).then_with(|| a.0.cmp(&b.0))
    };
    let evict_count = entries.len() - capacity;
    if evict_count < ranked.len() {
        ranked.select_nth_unstable_by(evict_count, compare);
    }
    ranked[..evict_count].sort_by(compare);
    let mut evicted = Vec::new();
    crate::allocation::try_reserve_vec_to_capacity(&mut evicted, evict_count).map_err(|error| {
        format!(
            "cache eviction heat result could not reserve {evict_count} entry id slot(s): {error}. Fix: shard the pipeline cache eviction batch."
        )
    })?;
    evicted.extend(ranked.into_iter().take(evict_count).map(|(_, e, _)| e.id));
    Ok(evicted)
}
