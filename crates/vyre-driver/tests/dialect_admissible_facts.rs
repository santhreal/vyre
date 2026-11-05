//! A backend reports what it can run, not what its adapter advertises.
//!
//! The megakernel envelope validates every target payload against the profile
//! the backend's own target dialect registers. When the adapter allows more
//! than the dialect admits, a composition that reads the raw adapter limit
//! builds geometry the envelope rejects at payload construction, and the
//! failure surfaces as `MKC017_MALFORMED_TARGET_PAYLOAD` far from the fact that
//! caused it. These tests pin the intersection every driver crate uses to
//! report an admissible fact.

use vyre_driver::target_dialect::{EmittedDialectModule, TargetDialect};
use vyre_megakernel::{SelectedLowering, TargetCompileError, TargetProfile};

fn unused_emit(
    _selected: &SelectedLowering,
    _profile: &TargetProfile,
) -> Result<EmittedDialectModule, TargetCompileError> {
    unreachable!("these tests never compile a lowering")
}

fn dialect(max_workgroup_size: [u32; 3], max_invocations_per_workgroup: u32) -> TargetDialect {
    TargetDialect {
        backend_id: "test-backend",
        dialect: "TEST",
        format: "test",
        format_version: 1,
        generation: 1,
        max_workgroup_size,
        max_invocations_per_workgroup,
        max_dynamic_shared_bytes: 16_384,
        subgroup_size: 0,
        emit: unused_emit,
    }
}

#[test]
fn a_dialect_narrower_than_the_adapter_caps_every_axis() {
    let spec_baseline = dialect([256, 256, 64], 256);

    assert_eq!(
        spec_baseline.admissible_workgroup_size([1024, 1024, 64]),
        [256, 256, 64],
        "Fix: a payload sized for the adapter is refused by the dialect that compiles it"
    );
    assert_eq!(
        spec_baseline.admissible_invocations_per_workgroup(1024),
        256,
        "Fix: the envelope multiplies the extents out and checks this total too"
    );
}

#[test]
fn a_dialect_wider_than_the_adapter_reports_the_adapter() {
    let permissive = dialect([1024, 1024, 64], 1024);

    assert_eq!(
        permissive.admissible_workgroup_size([256, 64, 1]),
        [256, 64, 1],
        "Fix: the dialect ceiling is not a floor; the device still cannot run more than it has"
    );
    assert_eq!(permissive.admissible_invocations_per_workgroup(256), 256);
}

#[test]
fn each_axis_is_intersected_on_its_own() {
    let mixed = dialect([256, 1024, 16], 256);

    assert_eq!(
        mixed.admissible_workgroup_size([1024, 64, 64]),
        [256, 64, 16],
        "Fix: intersecting one axis and copying the rest reports two facts the target refuses"
    );
}

#[test]
fn an_equal_limit_is_reported_unchanged() {
    let matched = dialect([1024, 1024, 64], 1024);

    assert_eq!(
        matched.admissible_workgroup_size([1024, 1024, 64]),
        [1024, 1024, 64]
    );
    assert_eq!(matched.admissible_invocations_per_workgroup(1024), 1024);
}

#[test]
fn an_admissible_fact_is_never_larger_than_the_registered_profile() {
    let spec_baseline = dialect([256, 256, 64], 256);
    let profile = spec_baseline
        .profile()
        .expect("the test dialect registers a valid profile");

    for adapter in [[1, 1, 1], [64, 8, 4], [1024, 1024, 64], [u32::MAX; 3]] {
        let admitted = spec_baseline.admissible_workgroup_size(adapter);
        for (axis, (extent, limit)) in admitted
            .iter()
            .zip(profile.max_workgroup_size())
            .enumerate()
        {
            assert!(
                *extent <= limit,
                "Fix: axis {axis} reports {extent}, which the registered profile limit {limit} rejects"
            );
        }
        let invocations = admitted.iter().copied().try_fold(1u32, u32::checked_mul);
        if let Some(invocations) = invocations {
            assert!(
                spec_baseline.admissible_invocations_per_workgroup(invocations)
                    <= profile.max_invocations_per_workgroup(),
                "Fix: the reported total exceeds what the registered profile admits"
            );
        }
    }
}
