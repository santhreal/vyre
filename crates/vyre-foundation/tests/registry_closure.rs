//! Registry/coverage closure for every `vyre-foundation` program builder.
//!
//! WHY: `vyre-foundation` owns the IR and the registry every other crate
//! submits into, so a builder here that is neither submitted through
//! `inventory` nor pinned by a test is IR construction nothing executes. The
//! contract, the enumerator, and what this gate does not catch are stated once
//! in `vyre_test_support`.
#![forbid(unsafe_code)]

vyre_test_support::registry_closure_gate! {
    // Measured floor: 2 tracked builders on 2026-08-15, down from 4. The two
    // that left were in `src/transform/compiler/`, deleted in dddd1eec08 as
    // compiler primitive specs nothing lowered; their IR construction is gone
    // from the tree, so there is nothing left for a test to pin. Lowering this
    // line is legitimate only for that reason: it is high enough that a parser
    // regression which drops most of the tree fails instead of reporting a
    // clean sweep of a nearly empty set.
    floor: 2,
    // Empty, and it must stay that way by fixing builders rather than listing
    // them: the enumerator's stale and now-covered guards make this list
    // only-shrinkable, so anything added here is a debt with no scheduled payer.
    waiver: [],
}
