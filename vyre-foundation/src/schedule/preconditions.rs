//! What one schedule transform must prove before it is applied.
//!
//! Every arm derives the typed preconditions the transform depends on, the
//! phases it touches, and the resource ceiling it joins, without mutating the
//! schedule. Application is a separate step in
//! [`legality`](super::legality), so a rejected transform leaves no partial
//! rewrite behind.

use std::collections::BTreeSet;

use super::{
    MappingLevel, MemoryPlacement, PipelineRole, ScheduleBoundKind, ScheduleLegalityError,
    SchedulePhaseId, SchedulePrecondition, ScheduleResourceBounds, ScheduleTransform,
    SelectedSchedule,
};

impl SelectedSchedule {
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
}
