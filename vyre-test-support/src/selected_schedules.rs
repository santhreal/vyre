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
    MappingLevel, PipelineRole, PipelineRoleGroup, ScheduleAxis, SchedulePhaseId,
    ScheduleTransform, SelectedSchedule, SynchronizationScope,
};

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
    let mut schedule = SelectedSchedule::synthetic(2);
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
        .expect("an exact workgroup shape is legal");
    schedule
        .apply(ScheduleTransform::Map {
            phase: SchedulePhaseId(0),
            axis,
            level: MappingLevel::Subgroup,
        })
        .expect("mapping an axis of the phase is legal");
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
        .expect("a bounded pipeline between two phases is legal");
    schedule
        .apply(ScheduleTransform::Synchronize {
            phases: vec![SchedulePhaseId(0), SchedulePhaseId(1)],
            scope: SynchronizationScope::Workgroup,
        })
        .expect("a workgroup synchronization boundary is legal");
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
        .expect("a device synchronization boundary is legal");
    schedule
        .apply(ScheduleTransform::PersistentQueue {
            phase: SchedulePhaseId(0),
            capacity: 128,
        })
        .expect("a bounded persistent queue is legal");
    schedule
}
