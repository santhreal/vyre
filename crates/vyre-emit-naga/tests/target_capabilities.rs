//! Naga emission target-capability admission contracts.
//!
//! Descriptors and target profiles come from `vyre_lower::descriptor_builder`,
//! the one owner of that scaffolding. What stays here is the part only this
//! crate can assert: which admission errors it produces, spelled exactly.

use vyre_lower::descriptor_builder::{
    all_subgroup_capabilities, descriptor, emission_target, op, permissive_workgroup_limits,
    target_without_subgroups, workgroup_limits,
};
use vyre_lower::{KernelOpKind, SubgroupCapabilities};

const FIXTURE_ID: &str = "capability-fixture";

#[test]
fn positive_supported_target_emits_the_requested_module() {
    let desc = descriptor(FIXTURE_ID).dispatch(64, 1, 1).build();
    let module = vyre_emit_naga::emit_with_capabilities(&desc, &target_without_subgroups())
        .expect("supported descriptor must emit");
    assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);
}

#[test]
fn negative_unsupported_subgroup_capability_has_stable_error() {
    let desc = descriptor(FIXTURE_ID)
        .dispatch(64, 1, 1)
        .ops([op(KernelOpKind::SubgroupLocalId, [], 0)])
        .build();
    let err = vyre_emit_naga::emit_with_capabilities(&desc, &target_without_subgroups())
        .expect_err("missing subgroup support must reject emission");
    assert_eq!(
        err.to_string(),
        "unsupported emission capability `subgroup.basic`"
    );
}

/// The same descriptor a target without subgroup support rejects must be
/// admitted once the target declares that support, or the rejection above is
/// measuring something other than the capability gate.
#[test]
fn positive_declared_subgroup_capability_admits_the_same_descriptor() {
    let desc = descriptor(FIXTURE_ID)
        .dispatch(64, 1, 1)
        .ops([op(KernelOpKind::SubgroupLocalId, [], 0)])
        .build();
    let module = vyre_emit_naga::emit_with_capabilities(
        &desc,
        &emission_target(permissive_workgroup_limits(), all_subgroup_capabilities()),
    )
    .expect("declared subgroup support must admit the descriptor");
    assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);
}

#[test]
fn boundary_workgroup_equal_to_target_limit_is_supported() {
    let desc = descriptor(FIXTURE_ID).dispatch(1024, 1, 1).build();
    let module = vyre_emit_naga::emit_with_capabilities(
        &desc,
        &emission_target(
            workgroup_limits([1024, 1, 1], 1024),
            SubgroupCapabilities::default(),
        ),
    )
    .expect("exact workgroup boundary must emit");
    assert_eq!(module.entry_points[0].workgroup_size, [1024, 1, 1]);
}

#[test]
fn adversarial_oversized_workgroup_returns_stable_error_without_emitting() {
    let desc = descriptor(FIXTURE_ID)
        .dispatch(u32::MAX, u32::MAX, u32::MAX)
        .build();
    let err = vyre_emit_naga::emit_with_capabilities(
        &desc,
        &emission_target(
            workgroup_limits([u32::MAX; 3], 1024),
            SubgroupCapabilities::default(),
        ),
    )
    .expect_err("invocation product overflow must fail closed");
    assert_eq!(
        err.to_string(),
        "unsupported emission capability `workgroup`: workgroup requests 4294967295 invocations, target limit is 1024"
    );
}
