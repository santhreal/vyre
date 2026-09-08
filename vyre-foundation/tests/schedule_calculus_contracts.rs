//! Contract tests for Row 107: Compositional schedule calculus.

use vyre_foundation::schedule::{
    DependencyPreservationCertificate, MappingLevel, ScheduleCostModel,
    ScheduleDiff, ScheduleDiffItem, ScheduleOp, SchedulePlan,
    ScheduleResourceBounds, ScheduleTree, SynchronizationScope,
};

#[test]
fn schedule_calculus_composition_and_validation() {
    // 1. Build a multi-level tiled, vectorised, and hierarchy-mapped schedule tree
    let inner_leaf = ScheduleTree::leaf(ScheduleOp::Vectorize {
        axis: 1,
        vector_width: 4,
    });

    let cert = DependencyPreservationCertificate::new(
        "polyhedral_tiling_legality_v1",
        vec![0, 1],
        true,
    );

    let tiled_node = ScheduleTree::node(
        ScheduleOp::Tile {
            axis: 0,
            tile_size: 16,
            inner_axis: 1,
        },
        inner_leaf,
        Some(cert),
    );

    let map_node = ScheduleTree::node(
        ScheduleOp::MapToHierarchy {
            axis: 0,
            level: MappingLevel::Workgroup,
            dimension: 0,
        },
        tiled_node,
        None,
    );

    let bounds = ScheduleResourceBounds {
        logical_points: 1024,
        shared_bytes: 4096,
        private_bytes: 64,
        registers_per_invocation: 32,
        pipeline_slots: 2,
        queue_capacity: 0,
    };

    let plan = SchedulePlan::new(map_node, bounds);
    assert_eq!(plan.version, 2);
    plan.validate().expect("well-formed schedule plan validates cleanly");
    assert_eq!(plan.root.node_count(), 3);
}

#[test]
fn schedule_calculus_rejects_invalid_zero_tile_and_unbounded_pipeline() {
    // Zero tile size must be rejected
    let invalid_tile = ScheduleTree::leaf(ScheduleOp::Tile {
        axis: 0,
        tile_size: 0,
        inner_axis: 1,
    });
    let plan1 = SchedulePlan::new(invalid_tile, ScheduleResourceBounds::default());
    assert!(plan1.validate().is_err());

    // Unbounded pipeline queue must be rejected
    let invalid_pipe = ScheduleTree::leaf(ScheduleOp::AsyncPipeline {
        stages: 0,
        ring_size: 0,
        role_groups: vec![],
    });
    let plan2 = SchedulePlan::new(invalid_pipe, ScheduleResourceBounds::default());
    assert!(plan2.validate().is_err());
}

#[test]
fn schedule_calculus_diffing_and_cost_modeling() {
    let tree_a = ScheduleTree::sequence(vec![
        ScheduleTree::leaf(ScheduleOp::Tile {
            axis: 0,
            tile_size: 16,
            inner_axis: 1,
        }),
        ScheduleTree::leaf(ScheduleOp::Synchronize {
            scope: SynchronizationScope::Workgroup,
        }),
    ]);

    let tree_b = ScheduleTree::sequence(vec![
        ScheduleTree::leaf(ScheduleOp::Tile {
            axis: 0,
            tile_size: 32,
            inner_axis: 1,
        }),
        ScheduleTree::leaf(ScheduleOp::Synchronize {
            scope: SynchronizationScope::Workgroup,
        }),
    ]);

    let plan_a = SchedulePlan::new(
        tree_a,
        ScheduleResourceBounds {
            logical_points: 1024,
            shared_bytes: 2048,
            private_bytes: 32,
            registers_per_invocation: 24,
            pipeline_slots: 1,
            queue_capacity: 0,
        },
    );

    let plan_b = SchedulePlan::new(
        tree_b,
        ScheduleResourceBounds {
            logical_points: 1024,
            shared_bytes: 4096,
            private_bytes: 32,
            registers_per_invocation: 32,
            pipeline_slots: 1,
            queue_capacity: 0,
        },
    );

    let diff = ScheduleDiff::diff_plans(&plan_a, &plan_b);
    assert!(!diff.is_empty(), "diff detects parameter and resource changes");
    assert!(diff.items.iter().any(|item| matches!(
        item,
        ScheduleDiffItem::ResourceBoundChanged { name, before: 2048, after: 4096 } if name == "shared_bytes"
    )));

    let cost_model = ScheduleCostModel::new(32, 65536, 64);
    let cost_a = cost_model.evaluate(&plan_a);
    let cost_b = cost_model.evaluate(&plan_b);

    assert!(cost_a.estimated_cycles > 0);
    assert!(cost_b.estimated_cycles > 0);
    // Larger tile size reduces global memory traffic
    assert!(cost_b.memory_traffic_bytes <= cost_a.memory_traffic_bytes);
}
