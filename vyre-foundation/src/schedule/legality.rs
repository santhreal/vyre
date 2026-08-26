//! Phase lookup and the checked rewrite behind schedule transform application.
//!
//! `SelectedSchedule::apply` and `SelectedSchedule::validate` state what a
//! transform must prove. Precondition derivation is in
//! [`preconditions`](super::preconditions); what remains here is phase and axis
//! lookup and the rewrite that runs once a transform has proved itself.

use std::collections::BTreeSet;

use super::{
    AxisMapping, ScheduleAxis, ScheduleLegalityError, SchedulePhase, SchedulePhaseId,
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
            T::Tile { phase, tiles } => {
                let selected = self
                    .phase_mut(*phase)
                    .ok_or(ScheduleLegalityError::MissingPhase(*phase))?;
                for (axis, factor) in tiles {
                    Self::split_phase_axis(selected, *axis, *factor)?;
                }
            }
            T::Split {
                phase,
                axis,
                factor,
            } => {
                let selected = self
                    .phase_mut(*phase)
                    .ok_or(ScheduleLegalityError::MissingPhase(*phase))?;
                Self::split_phase_axis(selected, *axis, *factor)?;
            }
            T::Recompute { .. } | T::Synchronize { .. } => {}
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

    /// Replace one axis with an outer axis of the quotient extent, followed by
    /// an inner axis of the factor.
    ///
    /// `require_factor` proved the factor divides the extent before the rewrite
    /// runs. The inner axis takes the next free index in the same logical
    /// region, so two tiles of one region never collide, and nest position
    /// records which loop is outer.
    fn split_phase_axis(
        phase: &mut SchedulePhase,
        axis: ScheduleAxis,
        factor: u32,
    ) -> Result<(), ScheduleLegalityError> {
        let position = phase.axes.iter().position(|held| *held == axis).ok_or(
            ScheduleLegalityError::MissingAxis {
                phase: phase.id,
                axis,
            },
        )?;
        let outer = axis.extent / u64::from(factor);
        if outer == 0 {
            return Err(ScheduleLegalityError::Zero("tiled axis extent"));
        }
        let inner = phase
            .axes
            .iter()
            .filter(|held| held.region == axis.region)
            .map(|held| held.axis)
            .max()
            .unwrap_or(axis.axis)
            .checked_add(1)
            .ok_or(ScheduleLegalityError::AxisIndexOverflow(phase.id))?;
        phase.axes[position] = ScheduleAxis {
            region: axis.region,
            axis: axis.axis,
            extent: outer,
        };
        phase.axes.insert(
            position + 1,
            ScheduleAxis {
                region: axis.region,
                axis: inner,
                extent: u64::from(factor),
            },
        );
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
