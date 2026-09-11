//! Substring search composes at every boundary lane count.
//!
//! WHY: the lane count reaches the composition as a plain `u32` and drives the
//! tile arithmetic. Zero, one, the largest count an adapter grants, and one past
//! it are the sizes where that arithmetic divides by zero or wraps. A panic
//! there reaches the caller as an aborted compile instead of an error it can act
//! on, so the contract is that a boundary count returns, whatever it returns.

#![cfg(feature = "pattern-substring")]

use std::panic::{catch_unwind, AssertUnwindSafe};

use vyre_foundation::optimizer::AdapterCaps;

/// The largest workgroup lane count any modelled adapter grants.
const MAX_WORKGROUP_LANES: u32 = AdapterCaps::high_end().max_invocations_per_workgroup;

#[test]
fn substring_search_boundaries_do_not_panic() {
    for &lanes in &[0, 1, MAX_WORKGROUP_LANES, MAX_WORKGROUP_LANES + 1] {
        catch_unwind(AssertUnwindSafe(|| {
            let _ = vyre_libs_pattern::pattern::substring_search(
                "haystack", "needle", "matches", lanes, 1,
            );
        }))
        .unwrap_or_else(|_| {
            panic!("Fix: substring_search must not panic at {lanes} workgroup lanes")
        });
    }
}
