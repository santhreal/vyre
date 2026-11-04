//! SPIR-V target-capability admission and exact-artifact contracts.
//!
//! Descriptors and target profiles come from `vyre_lower::descriptor_builder`,
//! the one owner of that scaffolding. What stays here is the part only this
//! crate can assert: that admission does not perturb the emitted words, and
//! which admission errors it produces, spelled exactly.

use vyre_lower::descriptor_builder::{
    all_subgroup_capabilities, descriptor, emission_target, op, permissive_workgroup_limits,
    target_without_subgroups, workgroup_limits,
};
use vyre_lower::{KernelOpKind, SubgroupCapabilities};

const FIXTURE_ID: &str = "spirv-capability-fixture";

fn baseline_target() -> vyre_lower::EmissionTargetCapabilities {
    emission_target(permissive_workgroup_limits(), all_subgroup_capabilities())
}

#[test]
fn positive_capability_path_preserves_exact_spirv_artifact() {
    let desc = descriptor(FIXTURE_ID).dispatch(64, 1, 1).build();
    let pinned = vyre_emit_spirv::emit(&desc).expect("pinned fixture must emit");
    let admitted = vyre_emit_spirv::emit_with_capabilities(&desc, &baseline_target())
        .expect("supported target must emit");
    assert_eq!(admitted, pinned);
    assert_eq!(admitted[0], vyre_emit_spirv::SPIRV_MAGIC);
}

#[test]
fn negative_unsupported_subgroup_capability_has_stable_error() {
    let desc = descriptor(FIXTURE_ID)
        .dispatch(64, 1, 1)
        .ops([op(KernelOpKind::SubgroupLocalId, [], 0)])
        .build();
    let err = vyre_emit_spirv::emit_with_capabilities(&desc, &target_without_subgroups())
        .expect_err("missing subgroup support must reject emission");
    assert_eq!(
        err.to_string(),
        "naga emission failed: unsupported emission capability `subgroup.basic`"
    );
}

#[test]
fn boundary_workgroup_equal_to_target_limit_preserves_exact_words() {
    let desc = descriptor(FIXTURE_ID).dispatch(1024, 1, 1).build();
    let pinned = vyre_emit_spirv::emit(&desc).expect("boundary fixture must emit");
    let admitted = vyre_emit_spirv::emit_with_capabilities(
        &desc,
        &emission_target(
            workgroup_limits([1024, 1, 1], 1024),
            SubgroupCapabilities::default(),
        ),
    )
    .expect("exact target boundary must emit");
    assert_eq!(admitted, pinned);
}

#[test]
fn adversarial_oversized_workgroup_fails_closed_with_stable_error() {
    let desc = descriptor(FIXTURE_ID)
        .dispatch(u32::MAX, u32::MAX, u32::MAX)
        .build();
    let err = vyre_emit_spirv::emit_with_capabilities(
        &desc,
        &emission_target(
            workgroup_limits([u32::MAX; 3], 1024),
            SubgroupCapabilities::default(),
        ),
    )
    .expect_err("overflowing invocation product must reject emission");
    assert_eq!(
        err.to_string(),
        "naga emission failed: unsupported emission capability `workgroup`: workgroup requests 4294967295 invocations, target limit is 1024"
    );
}
