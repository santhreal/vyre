//! Registry/coverage closure for every `vyre-primitives` program builder.
//!
//! WHY: `vyre-primitives` is the Tier 2.5 substrate, and its `inventory`
//! registry is what the conformance matrix and the op-matrix documents walk for
//! cross-backend parity. A builder here that is neither submitted through that
//! registry nor pinned by a test is a primitive no backend is ever compared on.
//! It still compiles, still appears in the catalog documents that are generated
//! from source, and still diverges from its reference arm with nothing red. The
//! enumerator that closes this gap lives in `vyre-test-support`; this file is
//! the caller for this crate.
//!
//! The candidate set is derived from the tree at run time rather than listed
//! here: every `pub fn ... -> Program` under `src/` is enumerated on each run,
//! so a builder added tomorrow is judged tomorrow. The `floor` argument is
//! what stops a broken derivation from passing vacuously, which is the failure
//! mode a source enumerator has: a regex that stops matching finds zero
//! builders, and zero builders are trivially all covered.
//!
//! What this does not catch: a builder that is covered only by a test which
//! names it and asserts nothing. Coverage here means reachable from the
//! registry or from test source, not that the assertion is worth anything.

#![forbid(unsafe_code)]

/// Uncovered builders with a recorded reason.
///
/// Empty. The five builders that used to sit here are the alternate call
/// conventions over builders the registry already sweeps: two take an explicit
/// op id, three take a params struct, and every call to them was in production
/// code. `delegating_builder_equivalence.rs` compares each against the arm it
/// delegates to instead, which is the assertion a waiver entry would have been
/// standing in for. The enumerator's stale and now-covered guards make this
/// list only-shrinkable, so anything added here is a debt with no scheduled
/// payer.
const COVERAGE_WAIVER: &[&str] = &[];

/// Minimum builder count the source enumeration must find.
///
/// Well under the current population, with room for the duplicate-builder
/// consolidation this crate is undergoing, and high enough that a parser
/// regression which drops most of the tree fails instead of reporting a clean
/// sweep of a nearly empty set.
const BUILDER_FLOOR: usize = 200;

#[test]
fn every_program_builder_is_tested_registered_or_explicitly_waived() {
    vyre_test_support::assert_registry_closure(
        env!("CARGO_MANIFEST_DIR"),
        COVERAGE_WAIVER,
        BUILDER_FLOOR,
    );
}
