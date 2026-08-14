//! Registry/coverage closure for every `vyre-driver` program builder.
//!
//! WHY: `vyre-driver` owns the backend-neutral machinery, so a builder here that
//! is neither submitted through `inventory` nor pinned by a test is a program
//! every concrete backend inherits and nobody checks. The contract, the
//! enumerator, and what this gate does not catch are stated once in
//! `vyre_test_support`.
#![forbid(unsafe_code)]

vyre_test_support::registry_closure_gate! {
    // This crate's builders are the elementwise parity program plus the three
    // hostile-input probes, a population small enough that any lower floor would
    // be a floor of zero, and zero builders are what a broken scan reports. So
    // the floor is the measured count: adding a builder never fails this, and
    // dropping one is a decision that has to be recorded on this line.
    floor: 4,
    // Empty. The hostile-input probe builders used to be the whole population of
    // this list, and they are covered by `hostile_input_probe_shapes.rs`
    // instead: the enumerator's stale and now-covered guards make this list
    // only-shrinkable, so anything added here is a debt with no scheduled payer.
    waiver: [],
}
