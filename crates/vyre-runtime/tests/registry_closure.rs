//! Registry/coverage closure for every `vyre-runtime` program builder.
//!
//! WHY: the runtime's builders are the shapes a resident dispatch actually
//! executes (sharded, priority, JIT, workspace-adapter variants), so a builder
//! here that is neither submitted through `inventory` nor pinned by a test is a
//! dispatch shape nobody runs. The contract, the enumerator, and what this gate
//! does not catch are stated once in `vyre_test_support`.
#![forbid(unsafe_code)]

vyre_test_support::registry_closure_gate! {
    // Under the current population of resident dispatch variants by enough room
    // that folding two of them together does not need this line edited, and
    // high enough that a parser regression which drops most of the tree fails
    // instead of reporting a clean sweep of a nearly empty set.
    floor: 10,
    // Empty, and it must stay that way by fixing builders rather than listing
    // them: the enumerator's stale and now-covered guards make this list
    // only-shrinkable, so anything added here is a debt with no scheduled payer.
    waiver: [],
}
