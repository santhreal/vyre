//! Transform application and the normalization it ends in.
//!
//! `apply` records the proof `legality` derived, joins the resource bounds it
//! reported, and leaves the schedule in canonical form: phases ordered by id,
//! region and predecessor lists sorted and deduplicated, dependency edges
//! acyclic.

use std::collections::{BTreeMap, BTreeSet};

use super::{
    ScheduleInverse, ScheduleLegalityError, SchedulePhase, SchedulePhaseId, ScheduleTransform,
    ScheduleTransformProvenance, ScheduleTransformRecord, SelectedSchedule,
};

impl SelectedSchedule {
    /// Apply one transform after proving its typed preconditions.
    pub fn apply(&mut self, transform: ScheduleTransform) -> Result<(), ScheduleLegalityError> {
        let previous_identity = self.identity()?;
        let (preconditions, source_phases, resource_bounds) = self.check_transform(&transform)?;
        let source_regions = source_phases
            .iter()
            .filter_map(|phase| self.phase(*phase))
            .flat_map(|phase| phase.source_regions.iter().copied())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let mut next = self.clone();
        next.apply_checked(&transform, resource_bounds)?;
        next.resources = next.resources.checked_join(resource_bounds)?;
        next.transforms.push(ScheduleTransformRecord {
            transform,
            preconditions,
            provenance: ScheduleTransformProvenance {
                source_regions,
                source_phases,
                inverse: ScheduleInverse { previous_identity },
            },
            resource_bounds,
        });
        next.canonicalize();
        *self = next;
        Ok(())
    }

    pub(super) fn validate_acyclic(&self) -> Result<(), ScheduleLegalityError> {
        let by_id = self
            .phases
            .iter()
            .map(|phase| (phase.id, phase))
            .collect::<BTreeMap<_, _>>();
        fn visit(
            id: SchedulePhaseId,
            by_id: &BTreeMap<SchedulePhaseId, &SchedulePhase>,
            visiting: &mut BTreeSet<SchedulePhaseId>,
            done: &mut BTreeSet<SchedulePhaseId>,
        ) -> Result<(), ScheduleLegalityError> {
            if done.contains(&id) {
                return Ok(());
            }
            if !visiting.insert(id) {
                return Err(ScheduleLegalityError::DependencyCycle { from: id, to: id });
            }
            let phase = by_id
                .get(&id)
                .ok_or(ScheduleLegalityError::MissingPhase(id))?;
            for predecessor in &phase.predecessors {
                visit(*predecessor, by_id, visiting, done)?;
            }
            visiting.remove(&id);
            done.insert(id);
            Ok(())
        }
        let mut visiting = BTreeSet::new();
        let mut done = BTreeSet::new();
        for id in by_id.keys().copied() {
            visit(id, &by_id, &mut visiting, &mut done)?;
        }
        Ok(())
    }

    pub(super) fn canonicalize(&mut self) {
        self.phases.sort_by_key(|phase| phase.id);
        for phase in &mut self.phases {
            phase.source_regions.sort_unstable();
            phase.source_regions.dedup();
            phase
                .mappings
                .sort_by_key(|mapping| (mapping.axis, mapping.level));
            phase.predecessors.sort_unstable();
            phase.predecessors.dedup();
        }
    }
}
