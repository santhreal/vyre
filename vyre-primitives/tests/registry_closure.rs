//! Registry/coverage closure for every `vyre-primitives` program builder.
//!
//! WHY: `vyre-primitives` is the Tier 2.5 substrate, and its `inventory`
//! registry is what the conformance matrix and the op-matrix documents walk for
//! cross-backend parity. A builder here that is neither submitted through that
//! registry nor pinned by a test is a primitive no backend is ever compared on.
//! The contract, the enumerator, and what this gate does not catch are stated
//! once in `vyre_test_support`.
#![forbid(unsafe_code)]

vyre_test_support::registry_closure_gate! {
    // Well under the current population, with room for the duplicate-builder
    // consolidation this crate is undergoing, and high enough that a parser
    // regression which drops most of the tree fails instead of reporting a
    // clean sweep of a nearly empty set.
    floor: 200,
    // Empty. The five builders that used to sit here are the alternate call
    // conventions over builders the registry already sweeps: two take an
    // explicit op id, three take a params struct, and every call to them was in
    // production code. `delegating_builder_equivalence.rs` compares each against
    // the arm it delegates to instead, which is the assertion a waiver entry
    // would have been standing in for.
    waiver: [],
}
