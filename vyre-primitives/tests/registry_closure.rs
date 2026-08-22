//! Registry/coverage closure for every `vyre-primitives` program builder.
//!
//! WHY: `vyre-primitives` owns the hardware intrinsics, and its `inventory`
//! registry is what the conformance matrix and the op-matrix documents walk for
//! cross-backend parity. A builder here that is neither submitted through that
//! registry nor pinned by a test is an intrinsic no backend is ever compared on.
//! The contract, the enumerator, and what this gate does not catch are stated
//! once in `vyre_test_support`.
#![forbid(unsafe_code)]

vyre_test_support::registry_closure_gate! {
    // The population is the `hardware/` tree: the named builders plus the ones
    // the capability macros generate. The floor sits under that count and above
    // zero, so a parser regression that drops the tree fails here instead of
    // reporting a clean sweep of an empty set. It was 200 while the composition
    // domains were parked in this crate; they now live in `vyre-libs`, whose own
    // closure gate carries the population that used to justify that number.
    floor: 5,
    // Empty. The five builders that used to sit here are the alternate call
    // conventions over builders the registry already sweeps: two take an
    // explicit op id, three take a params struct, and every call to them was in
    // production code. `delegating_builder_equivalence.rs` compares each against
    // the arm it delegates to instead, which is the assertion a waiver entry
    // would have been standing in for.
    waiver: [],
}
