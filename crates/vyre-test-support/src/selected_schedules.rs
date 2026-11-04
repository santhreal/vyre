//! Selected-schedule fixtures shared by the lowering contracts.
//!
//! Two suites needed the same two-phase schedule: the asynchronous transaction
//! contracts, which stage transfers under a bounded pipeline, and the physical
//! handoff contracts, which read every fact the projection carries. Each
//! carried its own copy of the transform sequence, and a copy that gains a
//! transform the other does not have is two different schedules under one name.
//!
//! The builders state transforms, never fields: a schedule reached by
//! `apply` is a schedule the legality rules admitted, so a fixture cannot
//! present a state the search could not select.

use vyre_foundation::schedule::{
    MappingLevel, PipelineRole, PipelineRoleGroup, ScheduleAxis, SchedulePhase, SchedulePhaseId,
    ScheduleResourceBounds, ScheduleTransform, SelectedSchedule, SynchronizationScope,
    SCHEDULE_IR_VERSION,
};

/// The baseline schedule selection derives for a validated logical graph.
///
/// A suite that lowers a real program needs the phases, extents and order that
/// program states, which only the compiler's own baseline carries. Delegating
/// keeps one definition of that shape instead of a harness copy that drifts.
#[cfg(feature = "semantic-requests")]
#[must_use]
pub fn baseline(logical: &vyre_foundation::logical::LogicalProgramGraph<'_>) -> SelectedSchedule {
    vyre_megakernel::baseline_schedule(logical)
}

/// A phase per region, one thread wide, with no axes and no dependencies.
///
/// A planner fixture needs a schedule of a stated size and nothing else: the
/// regions it phases have no extents to map and no order to keep. Selection
/// derives its own baseline from a validated logical graph, so this shape is a
/// harness input and never reaches a compile.
#[must_use]
pub fn synthetic(region_count: usize) -> SelectedSchedule {
    let phases = (0..region_count)
        .map(|index| {
            let region = u32::try_from(index).unwrap_or(u32::MAX);
            SchedulePhase {
                id: SchedulePhaseId(region),
                source_regions: vec![region],
                axes: Vec::new(),
                grid: [1, 1, 1],
                workgroup: [1, 1, 1],
                vector_width: 1,
                mappings: Vec::new(),
                predecessors: Vec::new(),
                resources: ScheduleResourceBounds {
                    logical_points: 1,
                    ..ScheduleResourceBounds::default()
                },
            }
        })
        .collect::<Vec<_>>();
    let mut source = Vec::with_capacity(8);
    source.extend_from_slice(&(region_count as u64).to_le_bytes());
    let points = region_count as u64;
    SelectedSchedule {
        version: SCHEDULE_IR_VERSION,
        logical_identity: *blake3::hash(&source).as_bytes(),
        source_phases: phases.clone(),
        source_resources: ScheduleResourceBounds {
            logical_points: points,
            ..ScheduleResourceBounds::default()
        },
        phases,
        transforms: Vec::new(),
        resources: ScheduleResourceBounds {
            logical_points: points,
            ..ScheduleResourceBounds::default()
        },
    }
}

/// A two-phase schedule whose first phase runs as a three-slot pipeline.
///
/// The phase carries an exact workgroup shape, one axis mapped to the subgroup
/// level, a producer/consumer role split, and a workgroup synchronization
/// boundary across both phases.
///
/// # Panics
///
/// Panics when a transform the legality rules used to admit is refused, which
/// is a changed contract rather than a broken fixture.
#[must_use]
pub fn mapped_pipelined_two_phase() -> SelectedSchedule {
    let mut schedule = synthetic(2);
    // The synthetic baseline carries no axes, and only an axis the phase
    // declares can be mapped.
    let axis = ScheduleAxis {
        region: 0,
        axis: 0,
        extent: 64,
    };
    schedule.source_phases[0].axes.push(axis);
    schedule.phases[0].axes.push(axis);
    schedule
        .apply(ScheduleTransform::SetWorkgroup {
            phase: SchedulePhaseId(0),
            shape: [32, 2, 1],
        })
        .expect("Fix: restate the workgroup shape so it divides the phase extents");
    schedule
        .apply(ScheduleTransform::Map {
            phase: SchedulePhaseId(0),
            axis,
            level: MappingLevel::Subgroup,
        })
        .expect("Fix: map an axis this phase declares, at a level the schedule admits");
    schedule
        .apply(ScheduleTransform::Pipeline {
            producer: SchedulePhaseId(0),
            consumer: SchedulePhaseId(1),
            ring_slots: 3,
            roles: vec![
                PipelineRoleGroup {
                    role: PipelineRole::Producer,
                    workers: 1,
                },
                PipelineRoleGroup {
                    role: PipelineRole::Consumer,
                    workers: 2,
                },
            ],
        })
        .expect("Fix: state ring slots and role workers a two-phase pipeline admits");
    schedule
        .apply(ScheduleTransform::Synchronize {
            phases: vec![SchedulePhaseId(0), SchedulePhaseId(1)],
            scope: SynchronizationScope::Workgroup,
        })
        .expect("Fix: synchronize phases that share the workgroup scope");
    schedule
}

/// [`mapped_pipelined_two_phase`] plus a device boundary and a bounded queue.
///
/// Every term a physical projection can carry is selected here, so a projection
/// that drops one is visible.
///
/// # Panics
///
/// Panics when a transform the legality rules used to admit is refused.
#[must_use]
pub fn richly_transformed_two_phase() -> SelectedSchedule {
    let mut schedule = mapped_pipelined_two_phase();
    schedule
        .apply(ScheduleTransform::Synchronize {
            phases: vec![SchedulePhaseId(0)],
            scope: SynchronizationScope::Device,
        })
        .expect("Fix: synchronize a phase the device scope reaches");
    schedule
        .apply(ScheduleTransform::PersistentQueue {
            phase: SchedulePhaseId(0),
            capacity: 128,
        })
        .expect("Fix: state a persistent queue capacity the phase admits");
    schedule
}
