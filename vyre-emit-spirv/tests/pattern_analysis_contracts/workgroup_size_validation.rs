//! `workgroup_size_validation` pattern analysis contracts.

use vyre_lower::WorkgroupLimitViolation;
use vyre_lower::WorkgroupLimits;
use vyre_emit_spirv::patterns::workgroup_size_validation::*;
use vyre_lower::descriptor_builder::body;
use vyre_lower::{BindingLayout, Dispatch, KernelDescriptor};

fn empty_with_dispatch(d: Dispatch) -> KernelDescriptor {
    KernelDescriptor {
        id: "k".into(),
        bindings: BindingLayout { slots: vec![] },
        dispatch: d,
        body: body().build(),
    }
}

#[test]
fn small_workgroup_is_valid() {
    let report = analyze(&empty_with_dispatch(Dispatch::new(64, 1, 1)));
    assert!(report.ok());
    assert_eq!(report.invocations(), 64);
}

#[test]
fn standard_1d_1024_workgroup_is_valid_at_baseline() {
    let report = analyze(&empty_with_dispatch(Dispatch::new(1024, 1, 1)));
    assert!(report.ok());
}

#[test]
fn dim_x_over_1024_violates_dim_limit() {
    let report = analyze(&empty_with_dispatch(Dispatch::new(2048, 1, 1)));
    assert!(!report.ok());
    let has_dim_violation = report.violations.iter().any(|v| {
        matches!(
            v,
            WorkgroupLimitViolation::DimensionExceeded { axis: 0, .. }
        )
    });
    assert!(has_dim_violation);
}

#[test]
fn dim_z_over_64_violates_baseline() {
    let report = analyze(&empty_with_dispatch(Dispatch::new(1, 1, 128)));
    assert!(!report.ok());
    let has = report.violations.iter().any(|v| {
        matches!(
            v,
            WorkgroupLimitViolation::DimensionExceeded {
                axis: 2,
                actual: 128,
                limit: 64
            }
        )
    });
    assert!(has);
}

#[test]
fn product_over_1024_violates_invocations() {
    // 32x32x2 = 2048  -  within per-dim, over invocations.
    let report = analyze(&empty_with_dispatch(Dispatch::new(32, 32, 2)));
    assert!(!report.ok());
    let has = report.violations.iter().any(|v| {
        matches!(
            v,
            WorkgroupLimitViolation::InvocationsExceeded { actual: 2048, .. }
        )
    });
    assert!(has);
}

#[test]
fn zero_dim_y_flagged() {
    let report = analyze(&empty_with_dispatch(Dispatch::new(64, 0, 1)));
    let has = report
        .violations
        .iter()
        .any(|v| matches!(v, WorkgroupLimitViolation::ZeroDimension { axis: 1 }));
    assert!(has);
}

#[test]
fn high_end_device_profile_allows_more() {
    // Custom profile: NVIDIA modern desktop allows 1024x1024x1024
    // (well above baseline z=64) and a higher product limit.
    let limits = WorkgroupLimits {
        max_size: [1024, 1024, 1024],
        max_invocations: 1024,
    };
    // 1x1x128 should fail baseline (z>64) but pass this profile (z<1024).
    let report = analyze_against(&empty_with_dispatch(Dispatch::new(1, 1, 128)), limits);
    assert!(report.ok());
}

#[test]
fn invocations_helper_computes_product() {
    let report = analyze(&empty_with_dispatch(Dispatch::new(8, 8, 4)));
    assert_eq!(report.invocations(), 256);
}

#[test]
fn carries_kernel_id() {
    let mut desc = empty_with_dispatch(Dispatch::new(1, 1, 1));
    desc.id = "named".into();
    let report = analyze(&desc);
    assert_eq!(report.kernel_id, "named");
}
