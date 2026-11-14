//! Pipeline-cache instrumentation: the public snapshot type and the
//! internal atomic counter struct shared by every concrete backend.

use std::sync::atomic::{AtomicU64, Ordering};

/// Pipeline-cache instrumentation counters.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PipelineCacheMetrics {
    /// Lookup attempts.
    pub lookups: u64,
    /// Successful lookups.
    pub hits: u64,
    /// Failed lookups.
    pub misses: u64,
    /// Accepted put attempts.
    pub puts: u64,
    /// Rejected put attempts, usually because a blob exceeds the byte budget.
    pub rejected_puts: u64,
    /// Entries evicted by capacity or byte-budget pressure.
    pub evictions: u64,
    /// Bytes removed by eviction.
    pub evicted_bytes: u64,
    /// Explicit flush attempts.
    pub flushes: u64,
    /// Explicit flush failures.
    pub flush_errors: u64,
    /// Current retained bytes when the backend can report them cheaply.
    pub cached_bytes: u64,
    /// Current retained entries when the backend can report them cheaply.
    pub entries: u64,
}

impl PipelineCacheMetrics {
    /// Cache-hit rate in parts per million.
    #[must_use]
    pub fn hit_rate_ppm(&self) -> u32 {
        if self.lookups == 0 {
            return 0;
        }
        (u128::from(self.hits) * 1_000_000 / u128::from(self.lookups)).min(u128::from(u32::MAX))
            as u32
    }

    /// Sum two readouts, clamping each counter at `u64::MAX`.
    ///
    /// `PipelineCacheStore::metrics` returns a value, so an aggregate readout
    /// has nowhere to report an overflow to. Clamping is the contract; a
    /// counter that reaches `u64::MAX` is already past every rate this cache
    /// can be asked about.
    pub(super) fn saturating_add(self, rhs: Self) -> Self {
        Self {
            lookups: self.lookups.saturating_add(rhs.lookups),
            hits: self.hits.saturating_add(rhs.hits),
            misses: self.misses.saturating_add(rhs.misses),
            puts: self.puts.saturating_add(rhs.puts),
            rejected_puts: self.rejected_puts.saturating_add(rhs.rejected_puts),
            evictions: self.evictions.saturating_add(rhs.evictions),
            evicted_bytes: self.evicted_bytes.saturating_add(rhs.evicted_bytes),
            flushes: self.flushes.saturating_add(rhs.flushes),
            flush_errors: self.flush_errors.saturating_add(rhs.flush_errors),
            cached_bytes: self.cached_bytes.saturating_add(rhs.cached_bytes),
            entries: self.entries.saturating_add(rhs.entries),
        }
    }
}

#[derive(Debug, Default)]
pub(super) struct PipelineCacheCounters {
    pub(super) lookups: AtomicU64,
    pub(super) hits: AtomicU64,
    pub(super) misses: AtomicU64,
    pub(super) puts: AtomicU64,
    pub(super) rejected_puts: AtomicU64,
    pub(super) evictions: AtomicU64,
    pub(super) evicted_bytes: AtomicU64,
    pub(super) flushes: AtomicU64,
    pub(super) flush_errors: AtomicU64,
}

impl PipelineCacheCounters {
    #[inline]
    pub(super) fn add(counter: &AtomicU64, value: u64) {
        counter.fetch_add(value, Ordering::Relaxed);
    }

    pub(super) fn snapshot(&self, cached_bytes: u64, entries: u64) -> PipelineCacheMetrics {
        PipelineCacheMetrics {
            lookups: self.lookups.load(Ordering::Relaxed),
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            puts: self.puts.load(Ordering::Relaxed),
            rejected_puts: self.rejected_puts.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
            evicted_bytes: self.evicted_bytes.load(Ordering::Relaxed),
            flushes: self.flushes.load(Ordering::Relaxed),
            flush_errors: self.flush_errors.load(Ordering::Relaxed),
            cached_bytes,
            entries,
        }
    }
}

// Inline: covers the crate-private `saturating_add`, which no integration test can name.
#[cfg(test)]
mod tests {
    use super::PipelineCacheMetrics;

    #[test]
    fn pipeline_cache_metrics_generated_hit_rates_are_exact_ppm() {
        for hits in 0..=1024_u64 {
            let metrics = PipelineCacheMetrics {
                lookups: 2048,
                hits,
                ..PipelineCacheMetrics::default()
            };
            assert_eq!(metrics.hit_rate_ppm(), ((hits * 1_000_000) / 2048) as u32);
        }
    }

    #[test]
    fn pipeline_cache_metric_aggregation_saturates_rather_than_wrapping() {
        let lhs = PipelineCacheMetrics {
            cached_bytes: u64::MAX,
            hits: 41,
            ..PipelineCacheMetrics::default()
        };
        let rhs = PipelineCacheMetrics {
            cached_bytes: 1,
            hits: 1,
            ..PipelineCacheMetrics::default()
        };

        let metrics = lhs.saturating_add(rhs);

        assert_eq!(metrics.cached_bytes, u64::MAX);
        assert_eq!(metrics.hits, 42);
    }
}
