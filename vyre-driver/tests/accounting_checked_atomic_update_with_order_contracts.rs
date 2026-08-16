//! Contracts for `vyre_driver::accounting`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

#[test]
fn checked_atomic_update_u64_publishes_checked_next_and_returns_observed() {
    let counter = AtomicU64::new(41);

    let previous = checked_atomic_update_u64_with_order(
        &counter,
        Ordering::Acquire,
        Ordering::AcqRel,
        Ordering::Acquire,
        |observed| observed.checked_add(1).ok_or("overflow"),
        |_, _| Ok(()),
    )
    .expect("Fix: reject accounting updates that overflow the tracked counter range - update should fit");

    assert_eq!(previous, 41);
    assert_eq!(counter.load(Ordering::Acquire), 42);
}

#[test]
fn checked_atomic_update_u32_rejects_without_publishing() {
    let counter = AtomicU32::new(u32::MAX);

    let error = checked_atomic_update_u32_with_order(
        &counter,
        Ordering::Acquire,
        Ordering::AcqRel,
        Ordering::Acquire,
        |observed| observed.checked_add(1).ok_or("overflow"),
        |_, _| Ok(()),
    )
    .expect_err("overflow should be surfaced");

    assert_eq!(error, "overflow");
    assert_eq!(counter.load(Ordering::Acquire), u32::MAX);
}
