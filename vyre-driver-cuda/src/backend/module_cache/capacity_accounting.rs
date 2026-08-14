//! Byte accounting, saturating telemetry counters, and eviction sizing shared
//! by the PTX source cache and the loaded-module cache.

use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

use smallvec::SmallVec;
use vyre_driver::accounting::{
    checked_atomic_add_usize_with_order, checked_atomic_sub_usize as checked_sub_usize,
    pinning_atomic_increment_u32, pinning_atomic_increment_u64,
};
use vyre_driver::BackendError;

use crate::backend::staging_reserve::reserve_smallvec;

pub(super) fn reserve_cached_source_bytes(
    cached_source_bytes: &AtomicUsize,
    source_bytes: usize,
) -> Result<(), BackendError> {
    checked_atomic_add_usize_with_order(
        cached_source_bytes,
        source_bytes,
        Ordering::Acquire,
        Ordering::AcqRel,
        Ordering::Acquire,
        |observed, attempted| {
            BackendError::new(format!(
                "CUDA PTX source cache byte accounting overflowed while adding {attempted} bytes to {observed}. Fix: shard generated PTX or clear the source cache before inserting another artifact."
            ))
        },
    )
}

pub(super) fn release_cached_source_bytes(
    cached_source_bytes: &AtomicUsize,
    dropped_bytes: usize,
) -> Result<(), BackendError> {
    checked_sub_usize(cached_source_bytes, dropped_bytes, |observed, dropped| {
        BackendError::new(format!(
                "CUDA PTX source-cache byte accounting underflowed while dropping {dropped} bytes from {observed}. Fix: clear the source cache and rebuild PTX cache residency from live entries."
            ))
    })
}

pub(super) fn increment_cache_counter_u64(counter: &AtomicU64, label: &'static str) {
    pinning_atomic_increment_u64(counter, Ordering::Relaxed, Ordering::Relaxed, || {
        tracing::error!(
            "{label} reached u64::MAX and is pinned. Fix: scrape CUDA cache telemetry before u64::MAX or shard the telemetry window."
        );
    });
}

pub(super) fn increment_cache_access_u32(counter: &AtomicU32, label: &'static str) {
    pinning_atomic_increment_u32(counter, Ordering::Relaxed, Ordering::Relaxed, || {
        tracing::error!(
            "{label} reached u32::MAX and is pinned for retention scoring. Fix: clear the CUDA cache or shard retention windows."
        );
    });
}

pub(super) fn retention_problem_size(
    len: usize,
    retain_after_eviction: usize,
    label: &str,
) -> Option<(u32, u32)> {
    let n = match u32::try_from(len) {
        Ok(value) => value,
        Err(source) => {
            tracing::error!("{label} retention candidate count cannot fit u32: {source}. Fix: lower cache soft caps or shard eviction telemetry.");
            return None;
        }
    };
    let k = match u32::try_from(retain_after_eviction) {
        Ok(value) => value,
        Err(source) => {
            tracing::error!("{label} retention target count cannot fit u32: {source}. Fix: lower cache soft caps or shard eviction telemetry.");
            return None;
        }
    };
    if k > n {
        tracing::error!("{label} retention target exceeds candidate count: retain={k}, candidates={n}. Fix: trigger eviction only after the cache reaches its soft cap or correct the retention policy.");
        return None;
    }
    Some((n, k))
}

/// Keys the submodular retention pass drops, or `None` when the retention
/// state could not be built. `None` means the caller clears the whole cache
/// and reports a total eviction; both caches make that same fallback choice.
pub(super) fn select_evicted_keys<K, const CAP: usize>(
    keys: &[K],
    gains: &mut [u32],
    retain_after_eviction: usize,
    cache_label: &str,
    removal_key_label: &'static str,
) -> Option<SmallVec<[K; CAP]>>
where
    [K; CAP]: smallvec::Array<Item = K>,
    K: Copy,
{
    let (n, k) = retention_problem_size(gains.len(), retain_after_eviction, cache_label)?;
    let retention = match vyre_driver::cache_eviction::try_select_retention_set(gains, n, k) {
        Ok(retention) => retention,
        Err(error) => {
            tracing::error!("{cache_label} eviction could not allocate retention state: {error}");
            return None;
        }
    };

    let mut to_remove: SmallVec<[K; CAP]> = SmallVec::new();
    if let Err(error) = reserve_smallvec(&mut to_remove, retention.len(), removal_key_label) {
        tracing::error!(
            "{cache_label} eviction could not reserve {} removal key slot(s): {error}",
            retention.len()
        );
        return None;
    }
    for (i, retain) in retention.iter().enumerate() {
        if *retain == 0 {
            if let Some(key) = keys.get(i) {
                to_remove.push(*key);
            }
        }
    }
    Some(to_remove)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    #[test]
    fn cache_hit_miss_counters_saturate_instead_of_wrapping_to_zero() {
        let counter = std::sync::atomic::AtomicU64::new(u64::MAX - 1);

        super::increment_cache_counter_u64(&counter, "test CUDA cache counter");
        assert_eq!(
            counter.load(Ordering::Acquire),
            u64::MAX,
            "Fix: CUDA cache telemetry must still reach u64::MAX exactly."
        );

        super::increment_cache_counter_u64(&counter, "test CUDA cache counter");
        assert_eq!(
            counter.load(Ordering::Acquire),
            u64::MAX,
            "Fix: CUDA cache telemetry must saturate at u64::MAX instead of wrapping to zero."
        );
    }
}
