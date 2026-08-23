//! Backend-neutral selected-schedule closure contracts.
//!
//! WHY: candidate search previously represented only fusion groups, one width,
//! and one topology. These tests close the transform variant space at the
//! foundation boundary and require typed preconditions, inverse/source
//! provenance, deterministic identity, resource bounds, and fail-closed
//! mutation behavior. They do not prove target-specific instruction selection.

use vyre_foundation::schedule::{
    MappingLevel, MemoryPlacement, PipelineRole, PipelineRoleGroup, ScheduleAxis,
    ScheduleLegalityError, SchedulePhase, SchedulePhaseId, ScheduleResourceBounds,
    ScheduleTransform, SelectedSchedule, SynchronizationScope, SCHEDULE_IR_VERSION,
};

fn axis(region: u32, axis: u32) -> ScheduleAxis {
    ScheduleAxis {
        region,
        axis,
        extent: 64,
    }
}

fn phase(id: u32, regions: Vec<u32>, axes: Vec<ScheduleAxis>) -> SchedulePhase {
    SchedulePhase {
        id: SchedulePhaseId(id),
        source_regions: regions,
        axes,
        grid: [64, 1, 1],
        workgroup: [32, 1, 1],
        vector_width: 1,
        mappings: Vec::new(),
        predecessors: Vec::new(),
        resources: ScheduleResourceBounds {
            logical_points: 64,
            ..ScheduleResourceBounds::default()
        },
    }
}

fn schedule() -> SelectedSchedule {
    let phases = vec![
        phase(0, vec![0, 4], vec![axis(0, 0), axis(4, 0)]),
        phase(1, vec![1], vec![axis(1, 0)]),
        phase(2, vec![2], vec![axis(2, 0)]),
        phase(3, vec![3], vec![axis(3, 0)]),
    ];
    let resources = ScheduleResourceBounds {
        logical_points: 256,
        ..ScheduleResourceBounds::default()
    };
    SelectedSchedule {
        version: SCHEDULE_IR_VERSION,
        logical_identity: [7; 32],
        source_phases: phases.clone(),
        source_resources: resources,
        phases,
        transforms: Vec::new(),
        resources,
    }
}

fn every_transform() -> Vec<ScheduleTransform> {
    vec![
        ScheduleTransform::PhaseFission {
            phase: SchedulePhaseId(0),
            split_after_region: 0,
        },
        ScheduleTransform::Fuse {
            phases: vec![SchedulePhaseId(1), SchedulePhaseId(2)],
        },
        ScheduleTransform::Tile {
            phase: SchedulePhaseId(1),
            tiles: vec![(axis(1, 0), 8)],
        },
        ScheduleTransform::Split {
            phase: SchedulePhaseId(1),
            axis: axis(1, 0),
            factor: 8,
        },
        ScheduleTransform::Reorder {
            phase: SchedulePhaseId(0),
            axes: vec![axis(4, 0), axis(0, 0)],
        },
        ScheduleTransform::Vectorize {
            phase: SchedulePhaseId(1),
            axis: axis(1, 0),
            width: 4,
        },
        ScheduleTransform::Map {
            phase: SchedulePhaseId(1),
            axis: axis(1, 0),
            level: MappingLevel::Subgroup,
        },
        ScheduleTransform::SetWorkgroup {
            phase: SchedulePhaseId(1),
            shape: [64, 1, 1],
        },
        ScheduleTransform::PlaceMemory {
            phase: SchedulePhaseId(1),
            value: 9,
            placement: MemoryPlacement::Workgroup,
            bytes: 1024,
        },
        ScheduleTransform::Prefetch {
            phase: SchedulePhaseId(1),
            value: 9,
            distance: 2,
            bytes: 256,
        },
        ScheduleTransform::Pipeline {
            producer: SchedulePhaseId(0),
            consumer: SchedulePhaseId(1),
            ring_slots: 3,
            roles: vec![
                PipelineRoleGroup {
                    role: PipelineRole::Producer,
                    workers: 8,
                },
                PipelineRoleGroup {
                    role: PipelineRole::Consumer,
                    workers: 24,
                },
            ],
        },
        ScheduleTransform::Recompute {
            phase: SchedulePhaseId(1),
            values: vec![9, 10],
        },
        ScheduleTransform::PersistentQueue {
            phase: SchedulePhaseId(1),
            capacity: 128,
        },
        ScheduleTransform::SpatialPartition {
            phase: SchedulePhaseId(1),
            partitions: 2,
            level: MappingLevel::ComputeUnitPartition,
        },
        ScheduleTransform::DispatchCut {
            before: SchedulePhaseId(0),
            after: SchedulePhaseId(1),
        },
        ScheduleTransform::Synchronize {
            phases: vec![SchedulePhaseId(0), SchedulePhaseId(1)],
            scope: SynchronizationScope::Device,
        },
        ScheduleTransform::AsymmetricJoin {
            producers: vec![SchedulePhaseId(0), SchedulePhaseId(1)],
            consumer: SchedulePhaseId(2),
        },
    ]
}

#[test]
fn every_neutral_transform_records_typed_provenance_and_changes_identity() {
    for transform in every_transform() {
        let mut selected = schedule();
        let before = selected.identity().unwrap();
        selected
            .apply(transform.clone())
            .unwrap_or_else(|error| panic!("{transform:?} must be legal: {error}"));
        selected.validate().unwrap();
        let record = selected.transforms.last().unwrap();
        assert_eq!(record.transform, transform);
        assert!(!record.preconditions.is_empty());
        assert!(!record.provenance.source_phases.is_empty());
        assert!(!record.provenance.source_regions.is_empty());
        assert_eq!(record.provenance.inverse.previous_identity, before);
        assert_ne!(selected.identity().unwrap(), before);
        let mut malformed = selected.clone();
        malformed.transforms.last_mut().unwrap().transform = ScheduleTransform::SetWorkgroup {
            phase: SchedulePhaseId(1),
            shape: [16, 1, 1],
        };
        assert!(
            malformed.validate().is_err(),
            "replay must reject a mutated {transform:?} record"
        );
    }
}

#[test]
fn every_transform_boundary_fails_closed_without_mutating_the_schedule() {
    let invalid = [
        ScheduleTransform::Split {
            phase: SchedulePhaseId(1),
            axis: axis(1, 0),
            factor: 0,
        },
        ScheduleTransform::Vectorize {
            phase: SchedulePhaseId(1),
            axis: axis(1, 0),
            width: 3,
        },
        ScheduleTransform::SetWorkgroup {
            phase: SchedulePhaseId(1),
            shape: [0, 1, 1],
        },
        ScheduleTransform::Pipeline {
            producer: SchedulePhaseId(0),
            consumer: SchedulePhaseId(1),
            ring_slots: 1,
            roles: vec![PipelineRoleGroup {
                role: PipelineRole::Producer,
                workers: 1,
            }],
        },
        ScheduleTransform::PersistentQueue {
            phase: SchedulePhaseId(1),
            capacity: 0,
        },
        ScheduleTransform::SpatialPartition {
            phase: SchedulePhaseId(1),
            partitions: 1,
            level: MappingLevel::Lane,
        },
        ScheduleTransform::DispatchCut {
            before: SchedulePhaseId(2),
            after: SchedulePhaseId(1),
        },
    ];
    for transform in invalid {
        let mut selected = schedule();
        let before = selected.clone();
        assert!(selected.apply(transform).is_err());
        assert_eq!(
            selected, before,
            "a rejected transform must be transactional"
        );
    }
}

#[test]
fn selected_phase_geometry_resources_and_order_are_identity_inputs() {
    let baseline = schedule();
    for mutate in [
        |selected: &mut SelectedSchedule| selected.phases[0].grid[0] = 32,
        |selected: &mut SelectedSchedule| selected.phases[0].workgroup[0] = 64,
        |selected: &mut SelectedSchedule| selected.phases[0].resources.shared_bytes = 4,
        |selected: &mut SelectedSchedule| selected.phases.swap(0, 1),
    ] {
        let mut changed = baseline.clone();
        mutate(&mut changed);
        assert_ne!(baseline.identity().unwrap(), changed.identity().unwrap());
    }
}

#[test]
fn stale_versions_duplicate_regions_cycles_and_overflow_are_rejected() {
    let mut stale = schedule();
    stale.version += 1;
    assert!(matches!(
        stale.validate(),
        Err(ScheduleLegalityError::UnsupportedVersion { .. })
    ));

    let mut duplicate = schedule();
    duplicate.phases[1].source_regions.push(0);
    assert_eq!(
        duplicate.validate(),
        Err(ScheduleLegalityError::DuplicateRegion(0))
    );

    let mut cycle = schedule();
    cycle.phases[0].predecessors.push(SchedulePhaseId(1));
    cycle.phases[1].predecessors.push(SchedulePhaseId(0));
    assert!(matches!(
        cycle.validate(),
        Err(ScheduleLegalityError::DependencyCycle { .. })
    ));

    let mut overflow = schedule();
    overflow.resources.shared_bytes = u64::MAX;
    let before = overflow.clone();
    assert_eq!(
        overflow.apply(ScheduleTransform::PlaceMemory {
            phase: SchedulePhaseId(1),
            value: 9,
            placement: MemoryPlacement::Workgroup,
            bytes: 1,
        }),
        Err(ScheduleLegalityError::ResourceOverflow("shared_bytes"))
    );
    assert_eq!(overflow, before);
}

#[test]
fn persisted_transform_preconditions_provenance_and_final_state_replay_fail_closed() {
    let transform = ScheduleTransform::SetWorkgroup {
        phase: SchedulePhaseId(1),
        shape: [64, 1, 1],
    };
    let mut selected = schedule();
    selected.apply(transform).unwrap();

    for mutate in [
        |schedule: &mut SelectedSchedule| schedule.transforms[0].preconditions.clear(),
        |schedule: &mut SelectedSchedule| {
            schedule.transforms[0].provenance.inverse.previous_identity[0] ^= 1;
        },
        |schedule: &mut SelectedSchedule| schedule.phases[1].workgroup[0] = 32,
    ] {
        let mut malformed = selected.clone();
        mutate(&mut malformed);
        assert!(malformed.validate().is_err());
    }
}
