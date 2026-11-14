//! Concrete executable cache identity, persistence, and telemetry.

/// Shared on-disk compiled-pipeline cache.
pub(crate) mod cache;
/// Stable cache hashing and device fingerprint helpers.
pub(crate) mod hashing;

pub use cache::{PipelineCacheIdentity, PipelineCacheMissReason, PipelineFeatureFlags};
pub use hashing::{
    dispatch_policy_cache_digest, dispatch_policy_cache_string, hex_encode, hex_short,
    push_lower_hex, try_normalized_program_cache_digest, update_dispatch_policy_cache_hash,
    PipelineDeviceFingerprint,
};

/// Default maximum number of compiled pipeline artifacts retained in memory.
pub(crate) const DEFAULT_PIPELINE_CACHE_ENTRIES: usize = 256;
/// Default maximum bytes retained by a backend pipeline cache.
pub(crate) const DEFAULT_PIPELINE_CACHE_BYTES: usize = 256 * 1024 * 1024;

/// Backend-reported compiled-pipeline cache counters.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct PipelineCacheSnapshot {
    /// Cache lookups that found an already-compiled artifact.
    pub hits: u64,
    /// Cache lookups that required compile/load work.
    pub misses: u64,
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

// Inline: `pipeline` is a `pub(crate)` module and the suite calls the private
// `parse_positive_env`, neither reachable from an integration test.
#[cfg(test)]
mod tests;
