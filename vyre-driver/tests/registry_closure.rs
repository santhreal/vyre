//! Registry/coverage closure for every `vyre-driver` program builder.
//!
//! WHY: `vyre-driver` owns the backend-neutral machinery, so a builder here that
//! is neither submitted through `inventory` nor pinned by a test is a program
//! every concrete backend inherits and nobody checks. The contract, the
//! enumerator, and what this gate does not catch are stated once in
//! `vyre_test_support`.
#![forbid(unsafe_code)]

vyre_test_support::registry_closure_gate! {
    // Zero. The four builders this crate has - the elementwise parity program
    // and the three hostile-input probes - live in `#[cfg(test)] mod` fixtures
    // (`parity_harness`, `hostile_input_closure`), so no build outside this
    // crate's own unit tests can reach them and they are not a published
    // surface. A broken scan is caught here by the production-file guard rather
    // than by this number, so the first production builder added to this crate
    // needs a test rather than an edit to this line.
    floor: 0,
    // Empty. The hostile-input probe builders used to be the whole population of
    // this list, and they are covered by `hostile_input_probe_shapes.rs`
    // instead: the enumerator's stale and now-covered guards make this list
    // only-shrinkable, so anything added here is a debt with no scheduled payer.
    waiver: [],
}
