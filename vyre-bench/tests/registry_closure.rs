//! Registry/coverage closure for every `vyre-bench` program builder.
//!
//! WHY: a benchmark builder that is neither submitted through `inventory` nor
//! pinned by a test is a measured shape nothing else executes, so it can drift
//! from the program the release evidence claims to time. It still compiles,
//! still appears in the catalog documents that are generated from source, and
//! still diverges from its reference arm with nothing red. The enumerator that
//! closes this gap lives in `vyre-test-support`; this file is the caller for
//! this crate.
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
/// Empty, and it must stay that way by fixing builders rather than listing
/// them: the enumerator's stale and now-covered guards make this list
/// only-shrinkable, so anything added here is a debt with no scheduled payer.
const COVERAGE_WAIVER: &[&str] = &[];

/// Minimum builder count the source enumeration must find.
///
/// This crate's public builder population is small: most of its programs are
/// crate-private case constructors, which the enumeration excludes. Any floor
/// below the measured count would be a floor of zero here, and zero builders
/// are what a broken scan reports, so the floor is the measured count. Adding a
/// builder never fails this; removing the last one is a decision that has to be
/// recorded on this line.
const BUILDER_FLOOR: usize = 1;

#[test]
fn every_program_builder_is_tested_registered_or_explicitly_waived() {
    vyre_test_support::assert_registry_closure(
        env!("CARGO_MANIFEST_DIR"),
        COVERAGE_WAIVER,
        BUILDER_FLOOR,
    );
}
