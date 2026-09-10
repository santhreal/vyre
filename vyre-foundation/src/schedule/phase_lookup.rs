//! Phase and axis lookup behind every schedule legality question.
//!
//! A transform names phases and axes by identity. Resolving those names, and
//! refusing the ones the schedule does not carry, is answered here so the
//! checked rewrite in [`legality`](super::legality) reads as the rewrite alone.

use std::collections::BTreeSet;

use super::{
    ScheduleAxis, ScheduleLegalityError, SchedulePhase, SchedulePhaseId, SelectedSchedule,
};

impl SelectedSchedule {
    pub(super) fn phase(&self, id: SchedulePhaseId) -> Option<&SchedulePhase> {
        self.phases.iter().find(|phase| phase.id == id)
    }

    pub(super) fn phase_mut(&mut self, id: SchedulePhaseId) -> Option<&mut SchedulePhase> {
        self.phases.iter_mut().find(|phase| phase.id == id)
    }

    pub(super) fn require_phase(
        &self,
        id: SchedulePhaseId,
    ) -> Result<&SchedulePhase, ScheduleLegalityError> {
        self.phase(id)
            .ok_or(ScheduleLegalityError::MissingPhase(id))
    }

    pub(super) fn require_axis(
        &self,
        phase: SchedulePhaseId,
        axis: ScheduleAxis,
    ) -> Result<(), ScheduleLegalityError> {
        if self.require_phase(phase)?.axes.contains(&axis) {
            Ok(())
        } else {
            Err(ScheduleLegalityError::MissingAxis { phase, axis })
        }
    }

    pub(super) fn distinct_phases(
        &self,
        phases: &[SchedulePhaseId],
        minimum: usize,
    ) -> Result<Vec<SchedulePhaseId>, ScheduleLegalityError> {
        if phases.len() < minimum {
            return Err(ScheduleLegalityError::Empty("transform phases"));
        }
        let unique = phases.iter().copied().collect::<BTreeSet<_>>();
        if unique.len() != phases.len() {
            return Err(ScheduleLegalityError::DuplicateTransformPhase);
        }
        for phase in &unique {
            self.require_phase(*phase)?;
        }
        Ok(unique.into_iter().collect())
    }
    pub(super) fn require_factor(factor: u32, extent: u64) -> Result<(), ScheduleLegalityError> {
        if factor == 0 {
            return Err(ScheduleLegalityError::Zero("transform factor"));
        }
        if extent % u64::from(factor) != 0 {
            return Err(ScheduleLegalityError::NonDivisible { extent, factor });
        }
        Ok(())
    }

    pub(super) fn require_forward_edge(
        &self,
        from: SchedulePhaseId,
        to: SchedulePhaseId,
    ) -> Result<(), ScheduleLegalityError> {
        self.require_phase(from)?;
        self.require_phase(to)?;
        if from >= to {
            return Err(ScheduleLegalityError::DependencyCycle { from, to });
        }
        Ok(())
    }
}
