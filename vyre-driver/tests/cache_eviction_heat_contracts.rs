//! Contracts for `vyre_driver::cache_eviction_heat`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_driver::cache_eviction_heat::{
    entries_to_evict, try_entries_to_evict, CacheEntryStats, DECAY_HALF_LIFE_S,
};

fn entry(id: u64, hits: u32, last_hit: f64) -> CacheEntryStats {
    CacheEntryStats {
        id,
        hit_count: hits,
        last_hit_time_s: last_hit,
    }
}

#[test]
fn under_capacity_evicts_nothing() {
    let entries = vec![entry(1, 10, 100.0), entry(2, 5, 200.0)];
    assert!(entries_to_evict(&entries, 10, 300.0).is_empty());
}

#[test]
fn cold_entry_evicted_before_hot_one() {
    let entries = vec![
        entry(1, 100, 290.0), // very recent, very hot
        entry(2, 1, 0.0),     // ancient, cold
    ];
    let evict = entries_to_evict(&entries, 1, 300.0);
    assert_eq!(evict, vec![2], "ancient cold entry evicted first");
}

#[test]
fn equal_heat_evicts_in_input_order() {
    let entries = vec![
        entry(1, 10, 100.0),
        entry(2, 10, 100.0),
        entry(3, 10, 100.0),
    ];
    let evict = entries_to_evict(&entries, 1, 200.0);
    assert_eq!(evict, vec![1, 2], "tied heat → first two by input order");
}

#[test]
fn frequency_dominates_recency_at_equal_age() {
    let entries = vec![
        entry(1, 1000, 100.0), // ancient but very hit
        entry(2, 1, 100.0),    // ancient and rarely hit
    ];
    let evict = entries_to_evict(&entries, 1, 1000.0);
    assert_eq!(evict, vec![2]);
}

#[test]
fn recency_dominates_frequency_at_equal_hits() {
    // Both have 10 hits; one was 5 minutes ago, one was 1 hour ago.
    let entries = vec![
        entry(1, 10, 0.0),    // 1 hour ago
        entry(2, 10, 3300.0), // 5 minutes ago
    ];
    let evict = entries_to_evict(&entries, 1, 3600.0);
    assert_eq!(evict, vec![1], "older entry of same hit-count evicts first");
}

#[test]
fn heat_decays_with_age() {
    let e = entry(0, 100, 0.0);
    let fresh = e.heat(0.0);
    let half_life = e.heat(DECAY_HALF_LIFE_S);
    let two_half_lives = e.heat(2.0 * DECAY_HALF_LIFE_S);
    assert!((fresh - 100.0).abs() < 1e-9);
    assert!((half_life - 50.0).abs() < 1e-9);
    assert!((two_half_lives - 25.0).abs() < 1e-9);
}

#[test]
fn non_finite_timestamps_never_become_sticky() {
    let entries = vec![
        entry(1, u32::MAX, f64::NAN),
        entry(2, 1, 300.0),
        entry(3, u32::MAX, f64::INFINITY),
    ];
    let evict = entries_to_evict(&entries, 1, 300.0);
    assert_eq!(
        evict,
        vec![1, 3],
        "malformed cache metadata must lose to a finite live entry"
    );
}

#[test]
fn non_finite_current_time_is_total_and_deterministic() {
    let entries = vec![
        entry(1, 10, 100.0),
        entry(2, 10, 100.0),
        entry(3, 10, 100.0),
    ];
    let evict = entries_to_evict(&entries, 1, f64::NAN);
    assert_eq!(
        evict,
        vec![1, 2],
        "invalid clock samples must preserve deterministic eviction order"
    );
}

#[test]
fn try_entries_to_evict_matches_legacy_order() {
    let entries = vec![entry(1, 1, 0.0), entry(2, 10, 10.0), entry(3, 0, 20.0)];

    assert_eq!(
        try_entries_to_evict(&entries, 1, 20.0).unwrap(),
        entries_to_evict(&entries, 1, 20.0)
    );
}
