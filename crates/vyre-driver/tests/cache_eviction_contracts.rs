//! Contracts for `vyre_driver::cache_eviction`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_driver::cache_eviction::{
    eviction_basis_points, record_eviction, record_eviction_counts, select_retention_set,
    select_retention_set_into, try_select_retention_set_into,
};

#[test]
fn retains_top_k_gains() {
    let mut gains = vec![3, 10, 2, 8, 1];
    let picked = select_retention_set(&mut gains, 5, 2);
    assert_eq!(picked, vec![0, 1, 0, 1, 0]);
}

#[test]
fn zero_k_evicts_all() {
    let mut gains = vec![3, 10, 2];
    let picked = select_retention_set(&mut gains, 3, 0);
    assert_eq!(picked, vec![0, 0, 0]);
}

#[test]
fn k_equal_n_keeps_positive_gain_entries() {
    let mut gains = vec![3, 0, 2];
    let picked = select_retention_set(&mut gains, 3, 3);
    assert_eq!(picked, vec![1, 0, 1]);
}

#[test]
fn into_reuses_storage() {
    let mut gains = vec![1, 9, 4];
    let mut picked = Vec::with_capacity(8);
    let ptr = picked.as_ptr();
    select_retention_set_into(&mut gains, 3, 2, &mut picked);
    assert_eq!(picked, vec![0, 1, 1]);
    assert_eq!(picked.as_ptr(), ptr);
}

#[test]
fn try_into_reuses_storage() {
    let mut gains = vec![1, 9, 4];
    let mut picked = Vec::with_capacity(8);
    let ptr = picked.as_ptr();
    try_select_retention_set_into(&mut gains, 3, 2, &mut picked)
        .expect("Fix: retention scratch should be reusable");
    assert_eq!(picked, vec![0, 1, 1]);
    assert_eq!(picked.as_ptr(), ptr);
}

#[test]
fn invalid_sizing_is_clamped_not_panicked() {
    let mut gains = vec![5, 1];
    let picked = select_retention_set(&mut gains, 99, 99);
    assert_eq!(picked, vec![1, 1]);
}

#[test]
fn eviction_basis_points_are_exact_and_bounded() {
    assert_eq!(eviction_basis_points(0, 0), 0);
    assert_eq!(eviction_basis_points(1, 2), 5_000);
    assert_eq!(eviction_basis_points(476, 512), 9_296);
    assert_eq!(eviction_basis_points(9, 3), 10_000);
    assert_eq!(eviction_basis_points(usize::MAX, usize::MAX), 10_000);
}

#[test]
fn eviction_recording_accepts_hostile_ratios() {
    record_eviction(f64::NAN);
    record_eviction(f64::INFINITY);
    record_eviction(f64::NEG_INFINITY);
    record_eviction_counts(usize::MAX, 1);
}
