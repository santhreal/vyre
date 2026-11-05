//! Projection of one selected-schedule phase into the facts a target receives.
//!
//! Neutral lowering used to hand an emitter a workgroup shape and nothing else.
//! Every other selected fact, logical coverage, vector width, the hierarchy
//! level each axis was mapped to, pipeline role groups, ring depth,
//! synchronization boundaries and the checked resource ceiling, stopped at the
//! lowering boundary, so a backend that needed one rediscovered it from the op
//! stream or chose its own. This projection is the whole handoff: it is built
//! once from the validated schedule, it is checked before any target sees it,
//! and a backend selects instructions, transfer mechanisms and launch tactics
//! under it without changing it.

use vyre_foundation::schedule::{
    PipelineRole, SchedulePhaseId, ScheduleTransform, SelectedSchedule,
};

use super::{BarrierPhase, PhysicalSchedule};
use crate::verified_lowering::PhysicalLoweringError;

/// Current physical-schedule projection schema version.
///
/// This advances when the projection gains or changes a field, which changes
/// what a target is promised. It is independent of the schedule schema version,
/// which the projection also records.
pub const PHYSICAL_SCHEDULE_VERSION: u16 = 1;

impl BarrierPhase {
    /// Whether this boundary is the odd one of an alternating pair.
    ///
    /// A target that double-buffers barrier state alternates on this rather
    /// than counting boundaries itself.
    #[must_use]
    pub const fn parity(&self) -> bool {
        self.index % 2 == 1
    }
}

impl PhysicalSchedule {
    /// Project the facts one phase of `schedule` froze.
    ///
    /// # Errors
    ///
    /// Returns an error when the phase is absent from the schedule or the
    /// projected facts do not check.
    pub fn project(
        schedule: &SelectedSchedule,
        phase: SchedulePhaseId,
    ) -> Result<Self, PhysicalLoweringError> {
        let selected = schedule
            .phases
            .iter()
            .find(|candidate| candidate.id == phase)
            .ok_or_else(|| {
                PhysicalLoweringError::new(format!(
                    "selected schedule has no phase {}. Fix: project the phase the artifact entry point names.",
                    phase.0
                ))
            })?;

        let mut roles = Vec::new();
        let mut ring_slots = 0;
        let mut barriers = Vec::new();
        let mut queue_capacity = 0;
        for record in &schedule.transforms {
            match &record.transform {
                ScheduleTransform::Pipeline {
                    producer,
                    consumer,
                    ring_slots: slots,
                    roles: groups,
                } if *producer == phase || *consumer == phase => {
                    ring_slots = ring_slots.max(*slots);
                    for group in groups {
                        if !roles.contains(group) {
                            roles.push(group.clone());
                        }
                    }
                }
                ScheduleTransform::Synchronize { phases, scope } if phases.contains(&phase) => {
                    barriers.push(BarrierPhase {
                        index: u32::try_from(barriers.len()).unwrap_or(u32::MAX),
                        scope: *scope,
                    });
                }
                ScheduleTransform::PersistentQueue {
                    phase: persistent,
                    capacity,
                } if *persistent == phase => {
                    queue_capacity = queue_capacity.max(*capacity);
                }
                _ => {}
            }
        }

        let projected = Self {
            version: PHYSICAL_SCHEDULE_VERSION,
            schedule_version: schedule.version,
            logical_identity: schedule.logical_identity,
            phase: phase.0,
            logical_coverage: selected.grid,
            workgroup: selected.workgroup,
            vector_width: selected.vector_width,
            mappings: selected.mappings.clone(),
            roles,
            ring_slots,
            barriers,
            queue_capacity,
            resources: selected.resources,
        };
        projected.validate()?;
        Ok(projected)
    }

    /// Check every fact a target is allowed to rely on.
    ///
    /// # Errors
    ///
    /// Returns an error when the projection version is not this library's, a
    /// selected extent is absent, or a pipeline is stated without both of its
    /// role groups.
    pub fn validate(&self) -> Result<(), PhysicalLoweringError> {
        if self.version != PHYSICAL_SCHEDULE_VERSION {
            return Err(PhysicalLoweringError::new(format!(
                "physical schedule projection version {} is not {PHYSICAL_SCHEDULE_VERSION}. Fix: re-lower the selected schedule with this library.",
                self.version
            )));
        }
        if self.workgroup.iter().any(|extent| *extent == 0) {
            return Err(PhysicalLoweringError::new(format!(
                "projected workgroup {:?} has a zero extent. Fix: select an exact workgroup shape before lowering.",
                self.workgroup
            )));
        }
        if self.logical_coverage.iter().any(|extent| *extent == 0) {
            return Err(PhysicalLoweringError::new(format!(
                "projected logical coverage {:?} has a zero extent. Fix: select the exact coverage of the phase before lowering.",
                self.logical_coverage
            )));
        }
        if self.vector_width == 0 {
            return Err(PhysicalLoweringError::new(
                "projected vector width is zero. Fix: select a vector width of at least one before lowering.",
            ));
        }
        if self.is_pipelined() == self.roles.is_empty() {
            return Err(PhysicalLoweringError::new(format!(
                "projected pipeline states {} ring slots with {} role groups. Fix: project ring depth and role assignment from the same pipeline transform.",
                self.ring_slots,
                self.roles.len()
            )));
        }
        if self.is_pipelined()
            && (self.role_workers(PipelineRole::Producer) == 0
                || self.role_workers(PipelineRole::Consumer) == 0)
        {
            return Err(PhysicalLoweringError::new(
                "projected pipeline has no producer or no consumer workers. Fix: assign both roles in the pipeline transform.",
            ));
        }
        Ok(())
    }

    /// Invocations one workgroup of this phase launches.
    #[must_use]
    pub const fn invocations_per_workgroup(&self) -> u64 {
        self.workgroup[0] as u64 * self.workgroup[1] as u64 * self.workgroup[2] as u64
    }

    /// Workers assigned to `role`, zero when the phase is not pipelined.
    #[must_use]
    pub fn role_workers(&self, role: PipelineRole) -> u32 {
        self.roles
            .iter()
            .filter(|group| group.role == role)
            .map(|group| group.workers)
            .sum()
    }

    /// Whether the phase runs as a bounded asynchronous pipeline.
    #[must_use]
    pub const fn is_pipelined(&self) -> bool {
        self.ring_slots > 0
    }

    /// Whether the phase runs through a bounded persistent queue.
    #[must_use]
    pub const fn is_persistent(&self) -> bool {
        self.queue_capacity > 0
    }
}
