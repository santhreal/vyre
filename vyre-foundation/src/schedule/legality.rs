//! Typed legality proofs behind schedule transform application.
//!
//! `SelectedSchedule::apply` and `SelectedSchedule::validate` state what a
//! transform must prove. The proof machinery is here: phase and axis lookup,
//! precondition derivation, resource joins, and the checked rewrite.

use std::collections::BTreeSet;

use super::{
    AxisMapping, MappingLevel, MemoryPlacement, PipelineRole, ScheduleAxis, ScheduleBoundKind,
    ScheduleLegalityError, SchedulePhase, SchedulePhaseId, SchedulePrecondition,
    ScheduleResourceBounds, ScheduleTransform, SelectedSchedule,
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

    pub(super) fn check_transform(
        &self,
        transform: &ScheduleTransform,
    ) -> Result<
        (
            Vec<SchedulePrecondition>,
            Vec<SchedulePhaseId>,
            ScheduleResourceBounds,
        ),
        ScheduleLegalityError,
    > {
        use ScheduleTransform as T;
        let mut bounds = ScheduleResourceBounds::default();
        let result = match transform {
            T::PhaseFission {
                phase,
                split_after_region,
            } => {
                let selected = self.require_phase(*phase)?;
                let position = selected
                    .source_regions
                    .iter()
                    .position(|region| region == split_after_region)
                    .ok_or(ScheduleLegalityError::MissingRegion(*split_after_region))?;
                if position + 1 >= selected.source_regions.len() {
                    return Err(ScheduleLegalityError::InvalidFission(*phase));
                }
                (
                    vec![SchedulePrecondition::PhaseExists(*phase)],
                    vec![*phase],
                )
            }
            T::Fuse { phases } => {
                let phases = self.distinct_phases(phases, 2)?;
                (
                    vec![SchedulePrecondition::DistinctPhases(phases.clone())],
                    phases,
                )
            }
            T::Tile { phase, tiles } => {
                self.require_phase(*phase)?;
                if tiles.is_empty() {
                    return Err(ScheduleLegalityError::Empty("tile axes"));
                }
                let mut conditions = vec![SchedulePrecondition::PhaseExists(*phase)];
                for (axis, factor) in tiles {
                    self.require_axis(*phase, *axis)?;
                    Self::require_factor(*factor, axis.extent)?;
                    conditions.push(SchedulePrecondition::AxisExists(*axis));
                    conditions.push(SchedulePrecondition::Divisible {
                        extent: axis.extent,
                        factor: *factor,
                    });
                }
                (conditions, vec![*phase])
            }
            T::Split {
                phase,
                axis,
                factor,
            }
            | T::Vectorize {
                phase,
                axis,
                width: factor,
            } => {
                self.require_axis(*phase, *axis)?;
                Self::require_factor(*factor, axis.extent)?;
                (
                    vec![
                        SchedulePrecondition::PhaseExists(*phase),
                        SchedulePrecondition::AxisExists(*axis),
                        SchedulePrecondition::NonZero(ScheduleBoundKind::Factor),
                        SchedulePrecondition::Divisible {
                            extent: axis.extent,
                            factor: *factor,
                        },
                    ],
                    vec![*phase],
                )
            }
            T::Reorder { phase, axes } => {
                let selected = self.require_phase(*phase)?;
                if selected.axes.iter().copied().collect::<BTreeSet<_>>()
                    != axes.iter().copied().collect::<BTreeSet<_>>()
                    || selected.axes.len() != axes.len()
                {
                    return Err(ScheduleLegalityError::InvalidPermutation(*phase));
                }
                (
                    vec![
                        SchedulePrecondition::PhaseExists(*phase),
                        SchedulePrecondition::AxisPermutation,
                    ],
                    vec![*phase],
                )
            }
            T::Map { phase, axis, .. } => {
                self.require_axis(*phase, *axis)?;
                (
                    vec![
                        SchedulePrecondition::PhaseExists(*phase),
                        SchedulePrecondition::AxisExists(*axis),
                    ],
                    vec![*phase],
                )
            }
            T::SetWorkgroup { phase, shape } => {
                self.require_phase(*phase)?;
                if shape.contains(&0) {
                    return Err(ScheduleLegalityError::Zero("workgroup shape"));
                }
                (
                    vec![
                        SchedulePrecondition::PhaseExists(*phase),
                        SchedulePrecondition::NonZero(ScheduleBoundKind::Factor),
                    ],
                    vec![*phase],
                )
            }
            T::PlaceMemory {
                phase,
                placement,
                bytes,
                ..
            } => {
                self.require_phase(*phase)?;
                match placement {
                    MemoryPlacement::Workgroup => bounds.shared_bytes = *bytes,
                    MemoryPlacement::Invocation => bounds.private_bytes = *bytes,
                    MemoryPlacement::Device | MemoryPlacement::Retained => {}
                }
                (
                    vec![
                        SchedulePrecondition::PhaseExists(*phase),
                        SchedulePrecondition::BoundedResource(ScheduleBoundKind::Bytes),
                    ],
                    vec![*phase],
                )
            }
            T::Prefetch {
                phase,
                distance,
                bytes,
                ..
            } => {
                self.require_phase(*phase)?;
                if *distance == 0 {
                    return Err(ScheduleLegalityError::Zero("prefetch distance"));
                }
                bounds.private_bytes = *bytes;
                (
                    vec![
                        SchedulePrecondition::PhaseExists(*phase),
                        SchedulePrecondition::NonZero(ScheduleBoundKind::PrefetchDistance),
                        SchedulePrecondition::BoundedResource(ScheduleBoundKind::Bytes),
                    ],
                    vec![*phase],
                )
            }
            T::Pipeline {
                producer,
                consumer,
                ring_slots,
                roles,
            } => {
                let phases = self.distinct_phases(&[*producer, *consumer], 2)?;
                if *ring_slots == 0 {
                    return Err(ScheduleLegalityError::Zero("pipeline ring"));
                }
                if roles.is_empty()
                    || roles.iter().any(|role| role.workers == 0)
                    || !roles.iter().any(|role| role.role == PipelineRole::Producer)
                    || !roles.iter().any(|role| role.role == PipelineRole::Consumer)
                {
                    return Err(ScheduleLegalityError::InvalidPipelineRoles);
                }
                self.require_forward_edge(*producer, *consumer)?;
                bounds.pipeline_slots = *ring_slots;
                (
                    vec![
                        SchedulePrecondition::DistinctPhases(phases.clone()),
                        SchedulePrecondition::NonZero(ScheduleBoundKind::PipelineRing),
                        SchedulePrecondition::BoundedResource(ScheduleBoundKind::PipelineRing),
                        SchedulePrecondition::Acyclic,
                    ],
                    phases,
                )
            }
            T::Recompute { phase, values } => {
                self.require_phase(*phase)?;
                if values.is_empty() {
                    return Err(ScheduleLegalityError::Empty("recomputed values"));
                }
                (
                    vec![SchedulePrecondition::PhaseExists(*phase)],
                    vec![*phase],
                )
            }
            T::PersistentQueue { phase, capacity } => {
                self.require_phase(*phase)?;
                if *capacity == 0 {
                    return Err(ScheduleLegalityError::Zero("queue capacity"));
                }
                bounds.queue_capacity = *capacity;
                (
                    vec![
                        SchedulePrecondition::PhaseExists(*phase),
                        SchedulePrecondition::NonZero(ScheduleBoundKind::QueueCapacity),
                        SchedulePrecondition::BoundedResource(ScheduleBoundKind::QueueCapacity),
                    ],
                    vec![*phase],
                )
            }
            T::SpatialPartition {
                phase,
                partitions,
                level,
            } => {
                self.require_phase(*phase)?;
                if *partitions == 0 {
                    return Err(ScheduleLegalityError::Zero("partition count"));
                }
                if !matches!(
                    level,
                    MappingLevel::ComputeUnitPartition | MappingLevel::DevicePartition
                ) {
                    return Err(ScheduleLegalityError::InvalidPartitionLevel(*level));
                }
                (
                    vec![
                        SchedulePrecondition::PhaseExists(*phase),
                        SchedulePrecondition::NonZero(ScheduleBoundKind::PartitionCount),
                    ],
                    vec![*phase],
                )
            }
            T::DispatchCut { before, after } => {
                let phases = self.distinct_phases(&[*before, *after], 2)?;
                self.require_forward_edge(*before, *after)?;
                (
                    vec![
                        SchedulePrecondition::DistinctPhases(phases.clone()),
                        SchedulePrecondition::Acyclic,
                    ],
                    phases,
                )
            }
            T::Synchronize { phases, .. } => {
                let phases = self.distinct_phases(phases, 1)?;
                (
                    vec![SchedulePrecondition::DistinctPhases(phases.clone())],
                    phases,
                )
            }
            T::AsymmetricJoin {
                producers,
                consumer,
            } => {
                let producers = self.distinct_phases(producers, 2)?;
                self.require_phase(*consumer)?;
                if producers.contains(consumer) {
                    return Err(ScheduleLegalityError::DuplicateTransformPhase);
                }
                for producer in &producers {
                    self.require_forward_edge(*producer, *consumer)?;
                }
                let mut phases = producers;
                phases.push(*consumer);
                (
                    vec![
                        SchedulePrecondition::DistinctPhases(phases.clone()),
                        SchedulePrecondition::Acyclic,
                    ],
                    phases,
                )
            }
        };
        Ok((result.0, result.1, bounds))
    }

    pub(super) fn apply_checked(
        &mut self,
        transform: &ScheduleTransform,
        resource_bounds: ScheduleResourceBounds,
    ) -> Result<(), ScheduleLegalityError> {
        use ScheduleTransform as T;
        match transform {
            T::PhaseFission {
                phase,
                split_after_region,
            } => {
                let new_id = SchedulePhaseId(
                    self.phases
                        .iter()
                        .map(|item| item.id.0)
                        .max()
                        .unwrap_or(0)
                        .checked_add(1)
                        .ok_or(ScheduleLegalityError::PhaseIdOverflow)?,
                );
                let selected = self
                    .phase_mut(*phase)
                    .ok_or(ScheduleLegalityError::MissingPhase(*phase))?;
                let split = selected
                    .source_regions
                    .iter()
                    .position(|region| region == split_after_region)
                    .ok_or(ScheduleLegalityError::MissingRegion(*split_after_region))?
                    + 1;
                let second_regions = selected.source_regions.split_off(split);
                let second_region_set = second_regions.iter().copied().collect::<BTreeSet<_>>();
                let second_axes = selected
                    .axes
                    .iter()
                    .copied()
                    .filter(|axis| second_region_set.contains(&axis.region))
                    .collect::<Vec<_>>();
                selected
                    .axes
                    .retain(|axis| !second_region_set.contains(&axis.region));
                let mut second = selected.clone();
                second.id = new_id;
                second.source_regions = second_regions;
                second.axes = second_axes;
                second.predecessors = vec![*phase];
                self.phases.push(second);
            }
            T::Fuse { phases } => {
                let mut ids = phases.clone();
                ids.sort_unstable();
                let target = ids[0];
                let phase_set = ids.iter().copied().collect::<BTreeSet<_>>();
                let mut merged_regions = Vec::new();
                let mut merged_axes = Vec::new();
                let mut merged_predecessors = Vec::new();
                let mut merged_resources = ScheduleResourceBounds::default();
                for phase in self
                    .phases
                    .iter()
                    .filter(|phase| phase_set.contains(&phase.id))
                {
                    merged_regions.extend(&phase.source_regions);
                    merged_axes.extend(&phase.axes);
                    merged_predecessors.extend(&phase.predecessors);
                    merged_resources = merged_resources.checked_join(phase.resources)?;
                }
                merged_regions.sort_unstable();
                merged_regions.dedup();
                merged_axes.sort_unstable();
                merged_axes.dedup();
                merged_predecessors.retain(|phase| !phase_set.contains(phase));
                merged_predecessors.sort_unstable();
                merged_predecessors.dedup();
                self.phases.retain(|phase| !phase_set.contains(&phase.id));
                self.phases.push(SchedulePhase {
                    id: target,
                    source_regions: merged_regions,
                    axes: merged_axes,
                    grid: [merged_resources.logical_points.max(1), 1, 1],
                    workgroup: [1, 1, 1],
                    vector_width: 1,
                    mappings: Vec::new(),
                    predecessors: merged_predecessors,
                    resources: merged_resources,
                });
                for phase in &mut self.phases {
                    for predecessor in &mut phase.predecessors {
                        if phase_set.contains(predecessor) {
                            *predecessor = target;
                        }
                    }
                    phase.predecessors.sort_unstable();
                    phase.predecessors.dedup();
                    phase
                        .predecessors
                        .retain(|predecessor| *predecessor != phase.id);
                }
            }
            T::Reorder { phase, axes } => {
                self.phase_mut(*phase)
                    .ok_or(ScheduleLegalityError::MissingPhase(*phase))?
                    .axes = axes.clone();
            }
            T::Vectorize { phase, width, .. } => {
                self.phase_mut(*phase)
                    .ok_or(ScheduleLegalityError::MissingPhase(*phase))?
                    .vector_width = *width;
            }
            T::Map { phase, axis, level } => {
                let selected = self
                    .phase_mut(*phase)
                    .ok_or(ScheduleLegalityError::MissingPhase(*phase))?;
                selected.mappings.retain(|mapping| mapping.axis != *axis);
                selected.mappings.push(AxisMapping {
                    axis: *axis,
                    level: *level,
                });
            }
            T::SetWorkgroup { phase, shape } => {
                self.phase_mut(*phase)
                    .ok_or(ScheduleLegalityError::MissingPhase(*phase))?
                    .workgroup = *shape;
            }
            T::Pipeline {
                producer, consumer, ..
            }
            | T::DispatchCut {
                before: producer,
                after: consumer,
            } => {
                let selected = self
                    .phase_mut(*consumer)
                    .ok_or(ScheduleLegalityError::MissingPhase(*consumer))?;
                selected.predecessors.push(*producer);
                selected.predecessors.sort_unstable();
                selected.predecessors.dedup();
            }
            T::AsymmetricJoin {
                producers,
                consumer,
            } => {
                let selected = self
                    .phase_mut(*consumer)
                    .ok_or(ScheduleLegalityError::MissingPhase(*consumer))?;
                selected.predecessors.extend(producers);
                selected.predecessors.sort_unstable();
                selected.predecessors.dedup();
            }
            T::PersistentQueue { phase, capacity } => {
                self.phase_mut(*phase)
                    .ok_or(ScheduleLegalityError::MissingPhase(*phase))?
                    .resources
                    .queue_capacity = *capacity;
            }
            T::SpatialPartition {
                phase, partitions, ..
            } => {
                self.phase_mut(*phase)
                    .ok_or(ScheduleLegalityError::MissingPhase(*phase))?
                    .grid[0] = u64::from(*partitions);
            }
            T::PlaceMemory { phase, .. } | T::Prefetch { phase, .. } => {
                let selected = self
                    .phase_mut(*phase)
                    .ok_or(ScheduleLegalityError::MissingPhase(*phase))?;
                selected.resources = selected.resources.checked_join(resource_bounds)?;
            }
            T::Tile { .. } | T::Split { .. } | T::Recompute { .. } | T::Synchronize { .. } => {}
        }
        Ok(())
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
