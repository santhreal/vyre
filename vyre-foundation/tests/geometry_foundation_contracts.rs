//! Contract tests for geometry requirements and launch geometry models in vyre-foundation.
//!
//! Asserts:
//! 1. Neutrality: geometry types represent pure execution invariants without device-specific limits.
//! 2. Correctness: builder methods, defaults, validation, and program metadata reflection.

use std::sync::Arc;

use vyre_foundation::ir::{BufferDecl, DataType, Expr, MemoryOrdering, Node, Program};
use vyre_foundation::{
    CooperativeWidth, ElementPolicy, GeometryConstraintConflict, GeometryRequirements,
    LaunchGeometry, Uniformity,
};
#[test]
fn geometry_requirements_defaults_are_agnostic() {
    let req = GeometryRequirements::default();
    assert_eq!(req.cooperative_width, CooperativeWidth::Agnostic);
    assert_eq!(req.subgroup_width, CooperativeWidth::Agnostic);
    assert_eq!(req.min_shared_bytes, 0);
    assert_eq!(req.per_invocation_elements, ElementPolicy::Any);
    assert_eq!(req.subgroup_uniformity, Uniformity::None);
    assert!(!req.requires_cooperative_launch);
    assert_eq!(req.memory_ordering, None);
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

/// WHY: `Any` is the identity requirement and must not erase the distinction
/// between exactly one element and an arbitrary multiple of one.
#[test]
fn agnostic_composition_preserves_scalar_element_policy() {
    let scalar = GeometryRequirements::agnostic().with_element_policy(ElementPolicy::Scalar);
    let any = GeometryRequirements::agnostic();

    assert_eq!(
        scalar
            .compose(any)
            .expect("scalar composed with agnostic geometry"),
        scalar
    );
    assert_eq!(
        any.compose(scalar)
            .expect("agnostic geometry composed with scalar"),
        scalar
    );
}

#[test]
fn schedule_constraints_compose_every_neutral_dimension() {
    let left = GeometryRequirements::cooperative(CooperativeWidth::AtLeast(64))
        .with_subgroup_width(CooperativeWidth::AtLeast(16))
        .with_min_shared_bytes(1024)
        .with_element_policy(ElementPolicy::Multiple(4))
        .with_subgroup_uniformity(Uniformity::SubgroupUniform)
        .with_memory_ordering(MemoryOrdering::Acquire);
    let right = GeometryRequirements::cooperative(CooperativeWidth::Exactly(128))
        .with_subgroup_width(CooperativeWidth::Exactly(32))
        .with_min_shared_bytes(2048)
        .with_element_policy(ElementPolicy::Multiple(6))
        .with_subgroup_uniformity(Uniformity::WorkgroupUniform)
        .with_cooperative_launch()
        .with_memory_ordering(MemoryOrdering::Release);

    let composed = left.compose(right).expect("compatible constraints compose");
    assert_eq!(composed.cooperative_width, CooperativeWidth::Exactly(128));
    assert_eq!(composed.subgroup_width, CooperativeWidth::Exactly(32));
    assert_eq!(composed.min_shared_bytes, 2048);
    assert_eq!(
        composed.per_invocation_elements,
        ElementPolicy::Multiple(12)
    );
    assert_eq!(composed.subgroup_uniformity, Uniformity::WorkgroupUniform);
    assert!(composed.requires_cooperative_launch);
    assert_eq!(composed.memory_ordering, Some(MemoryOrdering::AcqRel));
}

#[test]
fn schedule_constraints_reject_each_width_conflict_with_stable_scope() {
    for (scope, left, right) in [
        (
            "workgroup",
            GeometryRequirements::cooperative(CooperativeWidth::Exactly(64)),
            GeometryRequirements::cooperative(CooperativeWidth::Exactly(32)),
        ),
        (
            "subgroup",
            GeometryRequirements::agnostic().with_subgroup_width(CooperativeWidth::Exactly(64)),
            GeometryRequirements::agnostic().with_subgroup_width(CooperativeWidth::Exactly(32)),
        ),
    ] {
        assert_eq!(
            left.compose(right),
            Err(GeometryConstraintConflict::ExactWidth {
                scope,
                left: 64,
                right: 32,
            })
        );
    }
}

#[test]
fn schedule_constraints_reject_every_zero_dimension() {
    assert_eq!(
        GeometryRequirements::cooperative(CooperativeWidth::Exactly(0))
            .compose(GeometryRequirements::agnostic()),
        Err(GeometryConstraintConflict::ZeroWidth { scope: "workgroup" })
    );
    assert_eq!(
        GeometryRequirements::agnostic()
            .with_subgroup_width(CooperativeWidth::AtLeast(0))
            .compose(GeometryRequirements::agnostic()),
        Err(GeometryConstraintConflict::ZeroWidth { scope: "subgroup" })
    );
    assert_eq!(
        GeometryRequirements::agnostic()
            .with_element_policy(ElementPolicy::Multiple(0))
            .compose(GeometryRequirements::agnostic()),
        Err(GeometryConstraintConflict::ZeroElementMultiple)
    );
}

#[test]
fn program_semantics_derive_width_scratch_uniformity_and_cooperative_launch() {
    let program = Program::wrapped(
        vec![
            BufferDecl::output("out", 0, DataType::U32).with_count(1),
            BufferDecl::workgroup("scratch", 64, DataType::U32),
        ],
        [64, 1, 1],
        vec![
            Node::store("out", Expr::u32(0), Expr::LocalId { axis: 0 }),
            Node::barrier_with_ordering(MemoryOrdering::GridSync),
        ],
    );

    let constraints =
        GeometryRequirements::from_program(&program).expect("semantic facts are bounded");
    assert_eq!(constraints.cooperative_width, CooperativeWidth::Exactly(64));
    assert_eq!(constraints.min_shared_bytes, 256);
    assert_eq!(
        constraints.subgroup_uniformity,
        Uniformity::WorkgroupUniform
    );
    assert!(constraints.requires_cooperative_launch);
    assert_eq!(constraints.memory_ordering, Some(MemoryOrdering::GridSync));
}

/// WHY: schedule-free barriers must retain every ordering and uniformity fact
/// that target admission reads after logical scheduling.
#[test]
fn logical_barriers_preserve_every_declared_ordering_requirement() {
    let orderings: Vec<_> = (0..=u8::MAX)
        .filter_map(|tag| MemoryOrdering::from_wire_tag(tag).ok())
        .collect();
    assert!(!orderings.is_empty());

    for ordering in orderings {
        let program =
            Program::wrapped(Vec::new(), [1, 1, 1], vec![Node::logical_barrier(ordering)]);
        let constraints =
            GeometryRequirements::from_program(&program).expect("logical ordering is bounded");
        assert_eq!(constraints.memory_ordering, Some(ordering));
        assert_eq!(
            constraints.subgroup_uniformity,
            Uniformity::WorkgroupUniform
        );
        assert_eq!(
            constraints.requires_cooperative_launch,
            ordering == MemoryOrdering::GridSync
        );
    }
}

#[test]
fn program_semantics_preserve_the_declared_atomic_ordering() {
    let program = Program::wrapped(
        vec![
            BufferDecl::read_write("state", 0, DataType::U32).with_count(1),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::atomic_add_ordered("state", Expr::u32(0), Expr::u32(1), MemoryOrdering::Relaxed),
        )],
    );

    let constraints =
        GeometryRequirements::from_program(&program).expect("semantic facts are bounded");
    assert_eq!(constraints.memory_ordering, Some(MemoryOrdering::Relaxed));
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
    let mut program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::Return],
    )
    .with_entry_op_id("vyre-libs::geometry::test");
    program.non_composable_with_self = true;
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
    assert!(
        Arc::ptr_eq(program.entry_arc(), with_geo.entry_arc()),
        "with_launch_geometry must preserve the entry Arc without deep cloning"
    );
    assert_eq!(with_geo.buffers().len(), 1);
    assert_eq!(with_geo.entry_op_id, program.entry_op_id);
    assert!(with_geo.non_composable_with_self);

    let with_rewritten = program.with_rewritten_launch_geometry(&geo);
    assert_eq!(with_rewritten.workgroup_size(), [512, 1, 1]);
    assert!(
        Arc::ptr_eq(program.entry_arc(), with_rewritten.entry_arc()),
        "with_rewritten_launch_geometry must preserve the entry Arc without deep cloning"
    );

    program.set_launch_geometry(&geo);
    assert_eq!(program.workgroup_size(), [512, 1, 1]);
    assert_eq!(program.buffers().len(), 1);
}
