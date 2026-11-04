//! Registry/coverage closure for every `vyre-debug` program builder.
//!
//! WHY: this crate's builders are the fixtures the diagnostics surfaces and the
//! documented examples run on, so a builder here that is neither submitted
//! through `inventory` nor pinned by a test is a fixture whose shape nothing
//! checks, and a diagnostic rendered from a drifted fixture proves nothing. The
//! contract, the enumerator, and what this gate does not catch are stated once
//! in `vyre_test_support`.
#![forbid(unsafe_code)]

vyre_test_support::registry_closure_gate! {
    // This crate publishes one fixture builder. Any floor below the measured
    // count would be a floor of zero here, and zero builders are what a broken
    // scan reports, so the floor is the measured count. Adding a fixture never
    // fails this; removing the last one is a decision that has to be recorded on
    // this line.
    floor: 1,
    // Empty, and it must stay that way by fixing builders rather than listing
    // them: the enumerator's stale and now-covered guards make this list
    // only-shrinkable, so anything added here is a debt with no scheduled payer.
    waiver: [],
}
