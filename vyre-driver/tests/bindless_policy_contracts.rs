//! Contracts for `vyre_driver::bindless_policy`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_driver::bindless_policy::{
    decide_bindless, BindlessDecision, BindlessInputs, BindlessSupport,
    BINDLESS_RESOURCE_COUNT_THRESHOLD,
};

fn inp(count: u32, support: BindlessSupport, dynamic: bool) -> BindlessInputs {
    BindlessInputs {
        resource_count: count,
        support,
        dynamic_indexing: dynamic,
    }
}

#[test]
fn unsupported_always_returns_traditional() {
    for count in [0, 8, 24, 100, u32::MAX] {
        for dynamic in [false, true] {
            assert_eq!(
                decide_bindless(inp(count, BindlessSupport::Unsupported, dynamic)),
                BindlessDecision::TraditionalBindings
            );
        }
    }
}

#[test]
fn below_threshold_returns_traditional_on_full_support() {
    // 23 < threshold(24).
    assert_eq!(
        decide_bindless(inp(23, BindlessSupport::Full, true)),
        BindlessDecision::TraditionalBindings
    );
}

#[test]
fn at_threshold_returns_bindless_on_full_support() {
    assert_eq!(
        decide_bindless(inp(24, BindlessSupport::Full, true)),
        BindlessDecision::Bindless
    );
}

#[test]
fn above_threshold_returns_bindless_on_full_support() {
    assert_eq!(
        decide_bindless(inp(100, BindlessSupport::Full, false)),
        BindlessDecision::Bindless
    );
}

#[test]
fn static_support_with_dynamic_access_returns_traditional() {
    // Static can't satisfy dynamic indexing of unbound slots  -
    // dynamic access on Static-only support falls back.
    assert_eq!(
        decide_bindless(inp(100, BindlessSupport::Static, true)),
        BindlessDecision::TraditionalBindings
    );
}

#[test]
fn static_support_with_static_access_returns_bindless() {
    // Static support with non-dynamic access is the sweet spot
    // for fixed descriptor arrays.
    assert_eq!(
        decide_bindless(inp(100, BindlessSupport::Static, false)),
        BindlessDecision::Bindless
    );
}

#[test]
fn static_support_below_threshold_returns_traditional() {
    // Even with non-dynamic access, low count → traditional.
    assert_eq!(
        decide_bindless(inp(10, BindlessSupport::Static, false)),
        BindlessDecision::TraditionalBindings
    );
}

#[test]
fn zero_resources_always_traditional() {
    for support in [
        BindlessSupport::Full,
        BindlessSupport::Static,
        BindlessSupport::Unsupported,
    ] {
        assert_eq!(
            decide_bindless(inp(0, support, false)),
            BindlessDecision::TraditionalBindings
        );
    }
}

#[test]
fn threshold_constant_matches_documentation() {
    // Pin the calibrated threshold so casual edits don't move it
    // without a corresponding benchmark update.
    assert_eq!(BINDLESS_RESOURCE_COUNT_THRESHOLD, 24);
}
