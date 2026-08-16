//! Cost selection behind the autotune knobs.
//!
//! `best_cost_index` is crate-private and the public knobs refuse an empty
//! candidate set before reaching it, so no integration test can hand it the
//! empty slice its own contract has to answer for. The selection cases that do
//! run through the public knobs are held there, so the index the private
//! function returns is proven against the multiplier the caller ends up with.

use super::*;

/// No measured cost selects nothing.
///
/// The empty case used to be a `debug_assert` in front of `costs[0]`, which is
/// absent from a release build, so the shipped binary indexed an empty slice.
#[test]
fn no_measured_cost_selects_nothing() {
    assert_eq!(best_cost_index(&[]), None);
}

/// The lowest cost wins, and the first of a tie keeps the selection stable.
#[test]
fn the_lowest_cost_wins_and_a_tie_keeps_the_earlier_candidate() {
    assert_eq!(best_cost_index(&[3.0]), Some(0));
    assert_eq!(best_cost_index(&[3.0, 1.0, 2.0]), Some(1));
    assert_eq!(best_cost_index(&[1.0, 5.0, 1.0]), Some(0));
    assert_eq!(best_cost_index(&[5.0, 4.0, 3.0, 2.0]), Some(3));
}

/// Every position is reachable, so the scan reports the index it scanned.
///
/// The scan skips the first cost and counts from the rest, so an index that is
/// off by one selects the neighbour of the cheapest candidate at every position
/// except the first, which is exactly the case a single example misses.
#[test]
fn the_reported_index_is_the_position_of_the_lowest_cost() {
    let width = 6;
    for cheapest in 0..width {
        let costs: Vec<f64> = (0..width)
            .map(|index| if index == cheapest { 1.0 } else { 9.0 })
            .collect();
        assert_eq!(
            best_cost_index(&costs),
            Some(cheapest),
            "Fix: the lowest cost at {cheapest} of {width} selected the wrong candidate"
        );
    }
}

/// A cost that is not a number never beats a measured one.
#[test]
fn an_unmeasurable_cost_never_wins() {
    assert_eq!(best_cost_index(&[f64::NAN, 2.0]), Some(1));
    assert_eq!(best_cost_index(&[2.0, f64::NAN]), Some(0));
    assert_eq!(best_cost_index(&[f64::NAN, f64::NAN]), Some(0));
}

/// The public knobs return the candidate that the cheapest cost sits against.
#[test]
fn the_autotune_knobs_return_the_candidate_paired_with_the_lowest_cost() {
    let policy = ResidentLaunchPolicy::standard();
    assert_eq!(
        policy.autotune_workgroup_size(&[64, 128, 256], &[3.0, 1.0, 2.0], 32),
        128
    );
    assert_eq!(
        policy.autotune_hit_capacity_multiplier(&[2, 4, 8], &[5.0, 4.0, 1.0]),
        8
    );
}

/// A knob with nothing measured keeps the value it was given.
#[test]
fn the_autotune_knobs_keep_the_current_value_when_nothing_was_measured() {
    let policy = ResidentLaunchPolicy::standard();
    assert_eq!(policy.autotune_workgroup_size(&[64, 128], &[], 32), 32);
    assert_eq!(policy.autotune_workgroup_size(&[], &[1.0], 32), 32);
    assert_eq!(
        policy.autotune_hit_capacity_multiplier(&[2, 4], &[]),
        policy.hit_capacity_multiplier
    );
    assert_eq!(
        policy.autotune_hit_capacity_multiplier(&[], &[1.0]),
        policy.hit_capacity_multiplier
    );
}

/// More candidates than costs selects only among the costs that exist.
#[test]
fn a_candidate_without_a_cost_is_not_selected() {
    let policy = ResidentLaunchPolicy::standard();
    assert_eq!(
        policy.autotune_workgroup_size(&[64, 128, 256], &[2.0, 1.0], 32),
        128
    );
}
