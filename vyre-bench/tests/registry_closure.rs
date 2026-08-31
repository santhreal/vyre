//! Registry/coverage closure for every `vyre-bench` program builder.
//!
//! WHY: a benchmark builder that is neither submitted through `inventory` nor
//! pinned by a test is a measured shape nothing else executes, so it can drift
//! from the program the release evidence claims to time. The contract, the
//! enumerator, and what this gate does not catch are stated once in
//! `vyre_test_support`.
#![forbid(unsafe_code)]

vyre_test_support::registry_closure_gate! {
    // This crate's public builder population is small: most of its programs are
    // crate-private case constructors, which the enumeration excludes. Any floor
    // below the measured count would be a floor of zero here, and zero builders
    // are what a broken scan reports, so the floor is the measured count. Adding
    // a builder never fails this; removing the last one is a decision that has
    // to be recorded on this line.
    floor: 1,
    // Empty, and it must stay that way by fixing builders rather than listing
    // them: the enumerator's stale and now-covered guards make this list
    // only-shrinkable, so anything added here is a debt with no scheduled payer.
    waiver: [],
}
