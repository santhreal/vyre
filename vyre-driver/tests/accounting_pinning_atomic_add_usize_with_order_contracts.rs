//! Contracts for `vyre_driver::accounting`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use vyre_driver::accounting::{
    pinning_atomic_add_usize_with_order, repair_atomic_sub_usize_fetch_with_order,
};

#[test]
fn pinning_atomic_add_usize_pins_without_wrapping_and_returns_previous() {
    let counter = AtomicUsize::new(usize::MAX - 1);
    let mut pinned = None;

    let previous = pinning_atomic_add_usize_with_order(
        &counter,
        2,
        Ordering::AcqRel,
        Ordering::Acquire,
        |observed, value| pinned = Some((observed, value)),
    );

    assert_eq!(previous, usize::MAX - 1);
    assert_eq!(counter.load(Ordering::Acquire), usize::MAX);
    assert_eq!(pinned, Some((usize::MAX - 1, 2)));

    let mut called_again = false;
    let previous = pinning_atomic_add_usize_with_order(
        &counter,
        1,
        Ordering::AcqRel,
        Ordering::Acquire,
        |_, _| called_again = true,
    );

    assert_eq!(previous, usize::MAX);
    assert_eq!(counter.load(Ordering::Acquire), usize::MAX);
    assert!(!called_again);
}

#[test]
fn repair_atomic_sub_usize_fetch_repairs_and_returns_observed() {
    let counter = AtomicUsize::new(3);
    let mut repair = None;

    let previous = repair_atomic_sub_usize_fetch_with_order(
        &counter,
        5,
        Ordering::Acquire,
        Ordering::AcqRel,
        Ordering::Acquire,
        |observed, value| repair = Some((observed, value)),
    );

    assert_eq!(previous, 3);
    assert_eq!(counter.load(Ordering::Acquire), 0);
    assert_eq!(repair, Some((3, 5)));
}
