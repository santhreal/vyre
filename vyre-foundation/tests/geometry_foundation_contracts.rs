//! Contract tests for geometry requirements and launch geometry models in vyre-foundation.
//!
//! Asserts:
//! 1. Neutrality: geometry types represent pure execution invariants without device-specific limits.
//! 2. Correctness: builder methods, defaults, validation, and program metadata reflection.

use vyre_foundation::geometry::{
    CooperativeWidth, ElementPolicy, GeometryLoweringError, GeometryRequirements, LaunchGeometry,
    Uniformity,
};
use vyre_foundation::ir::Program;

#[test]
fn geometry_requirements_defaults_are_agnostic() {
    let req = GeometryRequirements::default();
    assert_eq!(req.cooperative_width, CooperativeWidth::Agnostic);
    assert_eq!(req.min_shared_bytes, 0);
    assert_eq!(req.per_invocation_elements, ElementPolicy::Any);
    assert_eq!(req.subgroup_uniformity, Uniformity::None);
}

#[test]
fn geometry_requirements_builder_methods_compose() {
    let req = GeometryRequirements::cooperative(CooperativeWidth::AtLeast(64))
        .with_min_shared_bytes(4096)
        .with_element_policy(ElementPolicy::Multiple(4))
        .with_subgroup_uniformity(Uniformity::SubgroupUniform);

    assert_eq!(req.cooperative_width, CooperativeWidth::AtLeast(64));
    assert_eq!(req.min_shared_bytes, 4096);
    assert_eq!(req.per_invocation_elements, ElementPolicy::Multiple(4));
    assert_eq!(req.subgroup_uniformity, Uniformity::SubgroupUniform);
}

#[test]
fn launch_geometry_computes_total_invocations_and_validates() {
    let valid_geo = LaunchGeometry {
        workgroup: [256, 1, 1],
        grid: [4, 1, 1],
        elements_per_invocation: 4,
        pipeline_stages: 2,
        shared_bytes: 1024,
    };

    assert!(valid_geo.is_valid());
    assert_eq!(valid_geo.workgroup_invocations(), 256);
    assert_eq!(valid_geo.grid_total(), 4);
    assert_eq!(valid_geo.total_invocations(), 1024);

    let invalid_wg = LaunchGeometry {
        workgroup: [0, 1, 1],
        grid: [1, 1, 1],
        elements_per_invocation: 1,
        pipeline_stages: 1,
        shared_bytes: 0,
    };
    assert!(!invalid_wg.is_valid());

    let invalid_epi = LaunchGeometry {
        workgroup: [256, 1, 1],
        grid: [1, 1, 1],
        elements_per_invocation: 0,
        pipeline_stages: 1,
        shared_bytes: 0,
    };
    assert!(!invalid_epi.is_valid());
}

#[test]
fn program_applies_launch_geometry() {
    let mut program = Program::empty();
    assert_eq!(program.workgroup_size(), [1, 1, 1]);

    let geo = LaunchGeometry {
        workgroup: [512, 1, 1],
        grid: [8, 1, 1],
        elements_per_invocation: 2,
        pipeline_stages: 1,
        shared_bytes: 2048,
    };

    let with_geo = program.with_launch_geometry(&geo);
    assert_eq!(with_geo.workgroup_size(), [512, 1, 1]);

    program.set_launch_geometry(&geo);
    assert_eq!(program.workgroup_size(), [512, 1, 1]);
}

#[test]
fn geometry_lowering_errors_format_descriptively() {
    let err1 = GeometryLoweringError::UnsatisfiableRequirements("test conflict".into());
    assert!(err1.to_string().contains("test conflict"));

    let err2 = GeometryLoweringError::ExceedsWorkgroupLimits {
        requested: 1024,
        max: 256,
    };
    assert!(err2.to_string().contains("1024"));
    assert!(err2.to_string().contains("256"));

    let err3 = GeometryLoweringError::ExceedsSharedMemoryLimits {
        requested: 65536,
        max: 32768,
    };
    assert!(err3.to_string().contains("65536"));
    assert!(err3.to_string().contains("32768"));

    let err4 = GeometryLoweringError::UnsupportedCooperativeWidth {
        requested: 512,
        admitted: 256,
    };
    assert!(err4.to_string().contains("512"));
    assert!(err4.to_string().contains("256"));
}
