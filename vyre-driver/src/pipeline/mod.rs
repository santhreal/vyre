//! Concrete executable cache identity, persistence, and telemetry.

/// Shared on-disk compiled-pipeline cache.
pub(crate) mod cache;
/// Stable cache hashing and device fingerprint helpers.
pub(crate) mod hashing;

pub use cache::{
    DiskPipelineCache, PipelineCacheIdentity, PipelineCacheKey, PipelineCacheMissEvidence,
    PipelineCacheMissReason, PipelineFeatureFlags,
};
pub use hashing::{
    dispatch_policy_cache_digest, dispatch_policy_cache_string, hex_encode, hex_short,
    normalized_program_cache_digest, push_lower_hex, try_normalized_program_cache_digest,
    update_dispatch_policy_cache_hash, PipelineDeviceFingerprint,
};

/// Version mixed into every persistent pipeline cache key.
pub const CURRENT_PIPELINE_CACHE_KEY_VERSION: u32 = 1;
/// Default maximum number of compiled pipeline artifacts retained in memory.
pub const DEFAULT_PIPELINE_CACHE_ENTRIES: usize = 256;
/// Default maximum bytes retained by a backend pipeline cache.
pub const DEFAULT_PIPELINE_CACHE_BYTES: usize = 256 * 1024 * 1024;
/// Baseline one-dimensional workgroup used when a caller supplies no override.
pub const DEFAULT_1D_WORKGROUP_SIZE: [u32; 3] = [64, 1, 1];

/// Backend-reported compiled-pipeline cache counters.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct PipelineCacheSnapshot {
    /// Cache lookups that found an already-compiled artifact.
    pub hits: u64,
    /// Cache lookups that required compile/load work.
    pub misses: u64,
}

/// Pipeline reuse cache hit-rate audit.
///
/// Aggregates backend cache lookup outcomes into a report that records hit,
/// miss, and unknown counts. Unknown outcomes are excluded from the hit-rate
/// denominator because backends without real counters report `None`.
#[derive(Debug, Default, Clone)]
pub struct PipelineCacheAudit {
    hits: u64,
    misses: u64,
    unknowns: u64,
}

/// Snapshot of a [`PipelineCacheAudit`].
#[derive(Debug, Clone, PartialEq)]
pub struct PipelineCacheAuditReport {
    /// Lookups that found an already-compiled artifact.
    pub hits: u64,
    /// Lookups that performed compile/load work.
    pub misses: u64,
    /// Lookups whose backend did not report cache state.
    pub unknowns: u64,
    /// Hit rate in basis points (0..=10_000) over the
    /// `hits + misses` denominator (excluding unknowns). `None` when
    /// `hits + misses == 0` so the caller can distinguish "no data"
    /// from "0% hit rate".
    pub hit_rate_bps: Option<u32>,
    /// Whether the hit rate is below the operator-supplied alarm
    /// threshold. Always `false` when `hit_rate_bps` is `None`.
    pub below_alarm_threshold: bool,
}

impl PipelineCacheAudit {
    /// Empty audit accumulator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Push one outcome from the dispatcher.
    pub fn observe(&mut self, cache_hit: Option<bool>) {
        match cache_hit {
            Some(true) => self.hits = self.hits.saturating_add(1),
            Some(false) => self.misses = self.misses.saturating_add(1),
            None => self.unknowns = self.unknowns.saturating_add(1),
        }
    }

    /// Snapshot the audit, scoring it against `alarm_threshold_bps`.
    /// `alarm_threshold_bps = 8000` flags any audit with under 80% hit
    /// rate; pass `0` to disable the alarm.
    #[must_use]
    pub fn snapshot(&self, alarm_threshold_bps: u32) -> PipelineCacheAuditReport {
        let denominator = self.hits.saturating_add(self.misses);
        let hit_rate_bps = if denominator == 0 {
            None
        } else {
            Some(crate::numeric::ratio_basis_points_u64(
                self.hits,
                denominator,
                0,
                "pipeline cache hit rate",
                "driver",
            ))
        };
        let below_alarm_threshold = match hit_rate_bps {
            Some(rate) if alarm_threshold_bps > 0 => rate < alarm_threshold_bps,
            _ => false,
        };
        PipelineCacheAuditReport {
            hits: self.hits,
            misses: self.misses,
            unknowns: self.unknowns,
            hit_rate_bps,
            below_alarm_threshold,
        }
    }
}

/// Resolve pipeline cache limits from Tier-A operational environment settings.
#[must_use]
pub fn pipeline_cache_limits_from_env() -> (u32, usize) {
    let entries = parse_positive_env(
        "VYRE_PIPELINE_CACHE_ENTRIES",
        DEFAULT_PIPELINE_CACHE_ENTRIES as u32,
    );
    let bytes = parse_positive_env("VYRE_PIPELINE_CACHE_BYTES", DEFAULT_PIPELINE_CACHE_BYTES);
    (entries, bytes)
}

/// Parse a positive Tier-A env integer. Returns `default` when the variable is
/// unset; a present-but-invalid value (unparsable, non-positive, non-unicode)
/// is a misconfiguration surfaced loudly via `tracing::warn!` before falling
/// back so it is never silently discarded.
fn parse_positive_env<T>(name: &str, default: T) -> T
where
    T: std::str::FromStr + PartialOrd + Default + std::fmt::Display + Copy,
{
    let raw = match std::env::var(name) {
        Ok(raw) => raw,
        Err(std::env::VarError::NotPresent) => return default,
        Err(std::env::VarError::NotUnicode(_)) => {
            tracing::warn!(
                "ignoring non-unicode {name}: expected a positive integer; using default {default}"
            );
            return default;
        }
    };
    match raw.parse::<T>() {
        Ok(value) if value > T::default() => value,
        _ => {
            tracing::warn!(
                "ignoring invalid {name}={raw:?}: expected a positive integer; using default {default}"
            );
            default
        }
    }
}

#[cfg(test)]
mod tests;
