//! Contracts for `vyre_driver::validation`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_driver::validation::{
    blocks_per_compute_unit, resident_threads_per_compute_unit, LaunchGeometryLimits,
    ValidationCache,
};

#[test]
fn validation_cache_records_vsa_without_lock_shards() {
    let cache = ValidationCache::new(8, 8, 4);
    let hash = blake3::hash(b"program");
    cache
        .remember_success(hash, &[1, 2, 3, 4])
        .expect("Fix: lock-free VSA cache insertion must not fail");

    assert!(cache.contains_hash(&hash));
    assert_eq!(cache.vsa_hashes.len(), 1);
    assert!(format!("{cache:?}").contains("vsa_hashes"));
}

#[test]
fn validation_cache_bounds_vsa_hashes_by_clear() {
    let cache = ValidationCache::new(8, 2, 4);
    for i in 0..3u32 {
        cache
            .remember_success(blake3::hash(&i.to_le_bytes()), &[i])
            .expect("Fix: VSA cache insertion must stay infallible");
    }
    assert!(
        cache.vsa_hashes.len() <= 2,
        "Fix: bounded VSA cache must not grow past max entries"
    );
}

/// The residency division is integral and both of its edges are pinned,
/// because this arithmetic now has exactly one definition and CUDA's
/// cooperative launch preflight reads it.
///
/// A zero width has no meaningful block count and yields zero rather than
/// dividing. A width wider than the whole per-unit budget hosts no block at
/// all, so it also yields zero: that is a launch the caller must reject,
/// not one silently rounded up to a single block. Both match what
/// `cooperative_thread_residency_block_limit` did before the arithmetic
/// moved here, and a factoring that quietly changed either edge would be
/// worse than the duplicate it replaced.
#[test]
fn residency_division_is_integral_at_both_edges() {
    assert_eq!(blocks_per_compute_unit(1536, 0), 0);
    assert_eq!(resident_threads_per_compute_unit(1536, 0), 0);
    assert_eq!(blocks_per_compute_unit(1536, 2048), 0);
    assert_eq!(resident_threads_per_compute_unit(1536, 2048), 0);
    assert_eq!(blocks_per_compute_unit(0, 256), 0);
    assert_eq!(resident_threads_per_compute_unit(0, 256), 0);

    assert_eq!(blocks_per_compute_unit(1536, 1024), 1);
    assert_eq!(
        resident_threads_per_compute_unit(1536, 1024),
        1024,
        "Fix: 1024 wide against a 1536-thread unit strands 512 slots. The truncation is the whole point of pinning this."
    );
    assert_eq!(blocks_per_compute_unit(1536, 256), 6);
    assert_eq!(resident_threads_per_compute_unit(1536, 256), 1536);
}

/// A backend that reports no per-unit thread budget answers `unknown`, so
/// no residency-aware decision can be derived from a number it never gave.
#[test]
fn unreported_per_unit_budget_answers_unknown_rather_than_zero() {
    let reported = LaunchGeometryLimits {
        backend: "reported",
        max_threads_per_block: 1024,
        max_block_dim: [1024, 1024, 64],
        max_grid_dim: [u32::MAX, u32::MAX, u32::MAX],
        max_threads_per_sm: 1536,
    };
    let unreported = LaunchGeometryLimits {
        max_threads_per_sm: 0,
        ..reported
    };

    assert_eq!(reported.blocks_per_compute_unit(256), Some(6));
    assert_eq!(reported.resident_threads_per_compute_unit(256), Some(1536));
    assert_eq!(reported.blocks_per_compute_unit(0), None);
    assert_eq!(unreported.blocks_per_compute_unit(256), None);
    assert_eq!(unreported.resident_threads_per_compute_unit(256), None);
}
