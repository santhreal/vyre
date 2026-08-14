//! Registry/coverage closure for every `vyre-foundation` program builder.
//!
//! WHY: `vyre-foundation` owns the IR and the registry every other crate
//! submits into, so a builder here that is neither submitted through
//! `inventory` nor pinned by a test is IR construction nothing executes. The
//! contract, the enumerator, and what this gate does not catch are stated once
//! in `vyre_test_support`.
#![forbid(unsafe_code)]

vyre_test_support::registry_closure_gate! {
    // Under the current population by enough room that consolidating one step
    // builder does not need this line edited, and high enough that a parser
    // regression which drops most of the tree fails instead of reporting a
    // clean sweep of a nearly empty set.
    floor: 4,
    // Empty, and it must stay that way by fixing builders rather than listing
    // them: the enumerator's stale and now-covered guards make this list
    // only-shrinkable, so anything added here is a debt with no scheduled payer.
    waiver: [],
}
