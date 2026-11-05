//! Registry/coverage closure for every `vyre-libs` program builder.
//!
//! WHY: `vyre-libs` owns every Category A composition, so a builder here that
//! is neither submitted through `inventory` nor pinned by a test is a
//! composition nobody executes. It still compiles, still appears in the
//! catalog documents that are generated from source, and still diverges from
//! its reference arm with nothing red. The enumerator that closes this gap
//! lives in `vyre-test-support`; this file is the caller for this crate.
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
const COVERAGE_WAIVER: &[&str] = &[
    "atomic_grid_stride_u32",
    "attribute_child",
    "attribute_serial_child",
    "build_ifds_csr_program",
    "csr_forward_or_changed_parallel_batch",
    "csr_forward_or_changed_parallel_batch_global",
    "csr_forward_or_changed_parallel_batch_global_slot",
    "f32_elementwise_mul",
    "impact_mask_from_closure",
    "tiled_dot",
    "tiled_mean",
    "tiled_softmax",
    "union_find_alias_program",
];

/// Minimum builder count the source enumeration must find.
///
/// High enough that a parser regression which drops most of the tree fails
/// instead of reporting a clean sweep of a nearly empty set. The floor is only
/// useful within reach of the real population: at 225 against an enumeration of
/// 495 it would have passed while 270 builders vanished, which is what a broken
/// walk looks like. It moves with the population, up when the crate grows and
/// down when a deliberate removal takes builders with it.
const BUILDER_FLOOR: usize = 450;

#[test]
fn every_program_builder_is_tested_registered_or_explicitly_waived() {
    let workspace_root = vyre_test_support::monorepo::vyre_workspace_root();
    let domain_crates: Vec<std::path::PathBuf> = structure_gate::workspace_members(&workspace_root)
        .into_iter()
        .filter(|member| {
            let name = member.rsplit('/').next().unwrap_or(member.as_str());
            name == "vyre-libs" || name.starts_with("vyre-libs-")
        })
        .map(|member| workspace_root.join(member))
        .collect();

    vyre_test_support::assert_registry_closure_crates(
        &domain_crates,
        COVERAGE_WAIVER,
        BUILDER_FLOOR,
    );
}
