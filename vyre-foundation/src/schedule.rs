//! Versioned backend-neutral selected schedule representation.
//!
//! A logical program states what points and dependencies exist. This module
//! records how those regions are transformed and assigned to execution and
//! storage scopes before physical-kernel lowering. Concrete target names and
//! instruction choices are not part of this schema.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::logical::{LogicalExtent, LogicalProgramGraph};

/// Current backend-neutral schedule schema and identity version.
pub const SCHEDULE_IR_VERSION: u16 = 1;

/// Stable identity of one selected schedule phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SchedulePhaseId(pub u32);

/// One logical axis retained through schedule transformation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ScheduleAxis {
    /// Source logical region.
    pub region: u32,
    /// Source logical axis.
    pub axis: u32,
    /// Validated upper bound on this axis.
    pub extent: u64,
}

/// Backend-neutral physical hierarchy available to an axis mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MappingLevel {
    /// One invocation lane.
    Lane,
    /// A target-provided subgroup.
    Subgroup,
    /// One cooperative workgroup.
    Workgroup,
    /// A neutral partition of the target's compute capacity.
    ComputeUnitPartition,
    /// One device partition in a multi-device schedule.
    DevicePartition,
}

/// Backend-neutral storage placement selected for a graph value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryPlacement {
    /// Invocation-private storage.
    Invocation,
    /// Workgroup-shared storage.
    Workgroup,
    /// Device-visible storage.
    Device,
    /// Storage retained across submissions.
    Retained,
}

/// Scope crossed by an explicit selected-schedule synchronization phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SynchronizationScope {
    /// Synchronize a subgroup.
    Subgroup,
    /// Synchronize a workgroup.
    Workgroup,
    /// Synchronize all cooperating workgroups on one device.
    Device,
}

/// Role assigned to a bounded producer/consumer pipeline group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PipelineRole {
    /// Produces values into the pipeline ring.
    Producer,
    /// Consumes values from the pipeline ring.
    Consumer,
}

/// One bounded role group in a producer/consumer pipeline.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PipelineRoleGroup {
    /// Role performed by this group.
    pub role: PipelineRole,
    /// Nonzero number of logical workers assigned to the role.
    pub workers: u32,
}

/// Checked resource ceiling attached to a phase or transform.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ScheduleResourceBounds {
    /// Maximum logical points covered by the phase.
    pub logical_points: u64,
    /// Maximum workgroup-shared bytes required by the phase.
    pub shared_bytes: u64,
    /// Maximum invocation-private bytes required by the phase.
    pub private_bytes: u64,
    /// Maximum scalar register slots required by one invocation.
    pub registers_per_invocation: u32,
    /// Maximum slots in a bounded asynchronous pipeline ring.
    pub pipeline_slots: u32,
    /// Maximum entries in a persistent work queue.
    pub queue_capacity: u32,
}

impl ScheduleResourceBounds {
    fn checked_join(self, other: Self) -> Result<Self, ScheduleLegalityError> {
        Ok(Self {
            logical_points: self
                .logical_points
                .checked_add(other.logical_points)
                .ok_or(ScheduleLegalityError::ResourceOverflow("logical_points"))?,
            shared_bytes: self
                .shared_bytes
                .checked_add(other.shared_bytes)
                .ok_or(ScheduleLegalityError::ResourceOverflow("shared_bytes"))?,
            private_bytes: self
                .private_bytes
                .checked_add(other.private_bytes)
                .ok_or(ScheduleLegalityError::ResourceOverflow("private_bytes"))?,
            registers_per_invocation: self
                .registers_per_invocation
                .checked_add(other.registers_per_invocation)
                .ok_or(ScheduleLegalityError::ResourceOverflow(
                    "registers_per_invocation",
                ))?,
            pipeline_slots: self
                .pipeline_slots
                .checked_add(other.pipeline_slots)
                .ok_or(ScheduleLegalityError::ResourceOverflow("pipeline_slots"))?,
            queue_capacity: self
                .queue_capacity
                .checked_add(other.queue_capacity)
                .ok_or(ScheduleLegalityError::ResourceOverflow("queue_capacity"))?,
        })
    }
}

/// One selected logical-axis mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AxisMapping {
    /// Mapped logical axis.
    pub axis: ScheduleAxis,
    /// Selected hierarchy level.
    pub level: MappingLevel,
}

/// One phase of a selected schedule.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SchedulePhase {
    /// Stable phase identity.
    pub id: SchedulePhaseId,
    /// Source logical regions covered by this phase.
    pub source_regions: Vec<u32>,
    /// Axes available to transforms in this phase.
    pub axes: Vec<ScheduleAxis>,
    /// Exact logical coverage selected for this phase.
    pub grid: [u64; 3],
    /// Exact workgroup shape selected for this phase.
    pub workgroup: [u32; 3],
    /// Selected vector width.
    pub vector_width: u32,
    /// Selected axis mappings.
    pub mappings: Vec<AxisMapping>,
    /// Preceding selected phases.
    pub predecessors: Vec<SchedulePhaseId>,
    /// Checked resource ceiling.
    pub resources: ScheduleResourceBounds,
}

/// Stable kind of a nonzero or bounded schedule operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScheduleBoundKind {
    /// Split, tile, or vector factor.
    Factor,
    /// Byte resource bound.
    Bytes,
    /// Prefetch distance.
    PrefetchDistance,
    /// Pipeline ring slot count.
    PipelineRing,
    /// Persistent queue capacity.
    QueueCapacity,
    /// Spatial partition count.
    PartitionCount,
}

/// Typed precondition authenticated with one applied transform.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchedulePrecondition {
    /// The referenced phase exists.
    PhaseExists(SchedulePhaseId),
    /// The referenced axis exists in the phase.
    AxisExists(ScheduleAxis),
    /// A factor or bound is nonzero.
    NonZero(ScheduleBoundKind),
    /// An axis extent is divisible by the transform factor.
    Divisible {
        /// Source axis extent.
        extent: u64,
        /// Selected exact factor.
        factor: u32,
    },
    /// The listed phases are pairwise distinct.
    DistinctPhases(Vec<SchedulePhaseId>),
    /// A resource increase remains representable.
    BoundedResource(ScheduleBoundKind),
    /// Adding the edge preserves acyclic phase order.
    Acyclic,
    /// The new axis order is a permutation of the existing axes.
    AxisPermutation,
}

/// Inverse provenance for an applied transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ScheduleInverse {
    /// Identity of the complete selected schedule before this transform.
    pub previous_identity: [u8; 32],
}

/// Source and inverse provenance for one applied transform.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ScheduleTransformProvenance {
    /// Source logical regions read by the transform.
    pub source_regions: Vec<u32>,
    /// Source selected phases read by the transform.
    pub source_phases: Vec<SchedulePhaseId>,
    /// Exact inverse checkpoint.
    pub inverse: ScheduleInverse,
}

/// One backend-neutral schedule transform.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScheduleTransform {
    /// Split one phase after a source logical region.
    PhaseFission {
        /// Phase to split.
        phase: SchedulePhaseId,
        /// Last source region retained by the first phase.
        split_after_region: u32,
    },
    /// Fuse two or more phases.
    Fuse {
        /// Phases to fuse.
        phases: Vec<SchedulePhaseId>,
    },
    /// Tile one or more logical axes.
    Tile {
        /// Phase containing the axes.
        phase: SchedulePhaseId,
        /// Axes and nonzero tile factors.
        tiles: Vec<(ScheduleAxis, u32)>,
    },
    /// Split one logical axis by a nonzero exact factor.
    Split {
        /// Phase containing the axis.
        phase: SchedulePhaseId,
        /// Axis to split.
        axis: ScheduleAxis,
        /// Exact split factor.
        factor: u32,
    },
    /// Reorder every axis of one phase.
    Reorder {
        /// Phase to reorder.
        phase: SchedulePhaseId,
        /// Complete axis permutation.
        axes: Vec<ScheduleAxis>,
    },
    /// Select vector execution for one logical axis.
    Vectorize {
        /// Phase containing the axis.
        phase: SchedulePhaseId,
        /// Axis to vectorize.
        axis: ScheduleAxis,
        /// Nonzero exact vector width.
        width: u32,
    },
    /// Map a logical axis to a backend-neutral hierarchy level.
    Map {
        /// Phase containing the axis.
        phase: SchedulePhaseId,
        /// Axis to map.
        axis: ScheduleAxis,
        /// Selected hierarchy level.
        level: MappingLevel,
    },
    /// Select the exact workgroup shape for one phase.
    SetWorkgroup {
        /// Phase whose physical workgroup is selected.
        phase: SchedulePhaseId,
        /// Nonzero selected shape.
        shape: [u32; 3],
    },
    /// Place one graph value in a backend-neutral memory class.
    PlaceMemory {
        /// Phase using the value.
        phase: SchedulePhaseId,
        /// Graph value identity.
        value: u32,
        /// Selected memory class.
        placement: MemoryPlacement,
        /// Checked byte bound.
        bytes: u64,
    },
    /// Prefetch one graph value a bounded number of phases ahead.
    Prefetch {
        /// Phase issuing the prefetch.
        phase: SchedulePhaseId,
        /// Graph value identity.
        value: u32,
        /// Nonzero bounded prefetch distance.
        distance: u32,
        /// Checked byte bound.
        bytes: u64,
    },
    /// Form a bounded asynchronous producer/consumer pipeline.
    Pipeline {
        /// Producer phase.
        producer: SchedulePhaseId,
        /// Consumer phase.
        consumer: SchedulePhaseId,
        /// Nonzero ring slot count.
        ring_slots: u32,
        /// Explicit nonempty producer and consumer role groups.
        roles: Vec<PipelineRoleGroup>,
    },
    /// Recompute graph values instead of retaining them.
    Recompute {
        /// Phase that performs recomputation.
        phase: SchedulePhaseId,
        /// Nonempty graph value set.
        values: Vec<u32>,
    },
    /// Execute one phase through a bounded persistent queue.
    PersistentQueue {
        /// Persistent phase.
        phase: SchedulePhaseId,
        /// Nonzero queue capacity.
        capacity: u32,
    },
    /// Partition one phase spatially across neutral compute partitions.
    SpatialPartition {
        /// Phase to partition.
        phase: SchedulePhaseId,
        /// Nonzero partition count.
        partitions: u32,
        /// Neutral partition level.
        level: MappingLevel,
    },
    /// Force a submission boundary before one phase.
    DispatchCut {
        /// Preceding phase.
        before: SchedulePhaseId,
        /// Following phase.
        after: SchedulePhaseId,
    },
    /// Add an explicit synchronization phase boundary.
    Synchronize {
        /// Nonempty synchronized phases.
        phases: Vec<SchedulePhaseId>,
        /// Selected synchronization scope.
        scope: SynchronizationScope,
    },
    /// Join two or more producer phases into one consumer phase.
    AsymmetricJoin {
        /// Distinct producer phases.
        producers: Vec<SchedulePhaseId>,
        /// Consumer phase.
        consumer: SchedulePhaseId,
    },
}

/// One validated application of a schedule transform.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ScheduleTransformRecord {
    /// Applied transform.
    pub transform: ScheduleTransform,
    /// Typed preconditions proven before application.
    pub preconditions: Vec<SchedulePrecondition>,
    /// Source and inverse provenance.
    pub provenance: ScheduleTransformProvenance,
    /// Checked resource increase introduced by the transform.
    pub resource_bounds: ScheduleResourceBounds,
}

/// Versioned backend-neutral selected schedule.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SelectedSchedule {
    /// Schedule schema version.
    pub version: u16,
    /// Identity of the validated logical algorithm this schedule transforms.
    pub logical_identity: [u8; 32],
    /// Immutable schedule-free phase set used to replay transform proofs.
    pub source_phases: Vec<SchedulePhase>,
    /// Immutable resource bound before schedule transforms.
    pub source_resources: ScheduleResourceBounds,
    /// Selected phases in stable identity order.
    pub phases: Vec<SchedulePhase>,
    /// Applied transforms in source order.
    pub transforms: Vec<ScheduleTransformRecord>,
    /// Checked whole-schedule resource ceiling.
    pub resources: ScheduleResourceBounds,
}

impl SelectedSchedule {
    /// Construct the unfused, unmapped schedule for a validated logical graph.
    #[must_use]
    pub fn from_logical(logical: &LogicalProgramGraph<'_>) -> Self {
        let logical_identity = *blake3::hash(logical.semantic_wire()).as_bytes();
        let phases = logical
            .regions()
            .iter()
            .enumerate()
            .map(|(index, region)| {
                let axes = region
                    .extents
                    .iter()
                    .enumerate()
                    .map(|(axis, extent)| ScheduleAxis {
                        region: region.node.0,
                        axis: u32::try_from(axis).unwrap_or(u32::MAX),
                        extent: match extent {
                            LogicalExtent::Static(value)
                            | LogicalExtent::GraphValue { bound: value, .. } => *value,
                        },
                    })
                    .collect();
                SchedulePhase {
                    id: SchedulePhaseId(u32::try_from(index).unwrap_or(u32::MAX)),
                    source_regions: vec![region.node.0],
                    axes,
                    grid: [region.max_points, 1, 1],
                    workgroup: [1, 1, 1],
                    vector_width: 1,
                    mappings: Vec::new(),
                    predecessors: region
                        .dependencies
                        .iter()
                        .map(|dependency| SchedulePhaseId(dependency.predecessor.0))
                        .collect(),
                    resources: ScheduleResourceBounds {
                        logical_points: region.max_points,
                        ..ScheduleResourceBounds::default()
                    },
                }
            })
            .collect::<Vec<_>>();
        let resources = phases
            .iter()
            .fold(ScheduleResourceBounds::default(), |total, phase| {
                ScheduleResourceBounds {
                    logical_points: total
                        .logical_points
                        .saturating_add(phase.resources.logical_points),
                    ..total
                }
            });
        Self {
            version: SCHEDULE_IR_VERSION,
            logical_identity,
            source_phases: phases.clone(),
            source_resources: resources,
            phases,
            transforms: Vec::new(),
            resources,
        }
    }

    /// Construct a synthetic baseline used by bounded planner unit fixtures.
    #[must_use]
    pub fn synthetic(region_count: usize) -> Self {
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
        Self {
            version: SCHEDULE_IR_VERSION,
            logical_identity: *blake3::hash(&source).as_bytes(),
            source_phases: phases.clone(),
            source_resources: ScheduleResourceBounds {
                logical_points: region_count as u64,
                ..ScheduleResourceBounds::default()
            },
            phases,
            transforms: Vec::new(),
            resources: ScheduleResourceBounds {
                logical_points: region_count as u64,
                ..ScheduleResourceBounds::default()
            },
        }
    }

    /// Canonical schedule identity bytes.
    pub fn canonical_wire(&self) -> Result<Vec<u8>, ScheduleLegalityError> {
        serde_json::to_vec(self).map_err(|error| ScheduleLegalityError::Identity(error.to_string()))
    }

    /// Deterministic content identity of the complete selected schedule.
    pub fn identity(&self) -> Result<[u8; 32], ScheduleLegalityError> {
        Ok(*blake3::hash(&self.canonical_wire()?).as_bytes())
    }

    /// Return the phase containing one source logical region.
    #[must_use]
    pub fn phase_for_region(&self, region: u32) -> Option<&SchedulePhase> {
        self.phases
            .iter()
            .find(|phase| phase.source_regions.contains(&region))
    }

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

    /// Validate every persisted phase, transform proof, dependency, and resource bound.
    pub fn validate(&self) -> Result<(), ScheduleLegalityError> {
        if self.version != SCHEDULE_IR_VERSION {
            return Err(ScheduleLegalityError::UnsupportedVersion {
                found: self.version,
                expected: SCHEDULE_IR_VERSION,
            });
        }
        if self.phases.is_empty() {
            return Err(ScheduleLegalityError::Empty("phases"));
        }
        let mut ids = BTreeSet::new();
        let mut regions = BTreeSet::new();
        for phase in &self.phases {
            if !ids.insert(phase.id) {
                return Err(ScheduleLegalityError::DuplicatePhase(phase.id));
            }
            if phase.source_regions.is_empty() {
                return Err(ScheduleLegalityError::Empty("phase.source_regions"));
            }
            for region in &phase.source_regions {
                if !regions.insert(*region) {
                    return Err(ScheduleLegalityError::DuplicateRegion(*region));
                }
            }
            if phase.grid.contains(&0) || phase.workgroup.contains(&0) || phase.vector_width == 0 {
                return Err(ScheduleLegalityError::Zero("phase geometry"));
            }
            if phase.axes.iter().any(|axis| axis.extent == 0) {
                return Err(ScheduleLegalityError::Zero("axis extent"));
            }
            for predecessor in &phase.predecessors {
                if *predecessor == phase.id
                    || !self.phases.iter().any(|item| item.id == *predecessor)
                {
                    return Err(ScheduleLegalityError::DependencyCycle {
                        from: *predecessor,
                        to: phase.id,
                    });
                }
            }
        }
        self.validate_acyclic()?;
        for record in &self.transforms {
            if record.preconditions.is_empty()
                || record.provenance.source_phases.is_empty()
                || record.provenance.source_regions.is_empty()
            {
                return Err(ScheduleLegalityError::MissingProvenance);
            }
        }
        let mut replay = Self {
            version: self.version,
            logical_identity: self.logical_identity,
            source_phases: self.source_phases.clone(),
            source_resources: self.source_resources,
            phases: self.source_phases.clone(),
            transforms: Vec::new(),
            resources: self.source_resources,
        };
        for (index, record) in self.transforms.iter().enumerate() {
            replay.apply(record.transform.clone())?;
            if replay.transforms.last() != Some(record) {
                return Err(ScheduleLegalityError::InvalidTransformProof(index));
            }
        }
        if replay.phases != self.phases || replay.resources != self.resources {
            return Err(ScheduleLegalityError::ReplayMismatch);
        }
        let _ = self.identity()?;
        Ok(())
    }

    fn phase(&self, id: SchedulePhaseId) -> Option<&SchedulePhase> {
        self.phases.iter().find(|phase| phase.id == id)
    }

    fn phase_mut(&mut self, id: SchedulePhaseId) -> Option<&mut SchedulePhase> {
        self.phases.iter_mut().find(|phase| phase.id == id)
    }

    fn require_phase(&self, id: SchedulePhaseId) -> Result<&SchedulePhase, ScheduleLegalityError> {
        self.phase(id)
            .ok_or(ScheduleLegalityError::MissingPhase(id))
    }

    fn require_axis(
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

    fn distinct_phases(
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

    fn check_transform(
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

    fn apply_checked(
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

    fn require_factor(factor: u32, extent: u64) -> Result<(), ScheduleLegalityError> {
        if factor == 0 {
            return Err(ScheduleLegalityError::Zero("transform factor"));
        }
        if extent % u64::from(factor) != 0 {
            return Err(ScheduleLegalityError::NonDivisible { extent, factor });
        }
        Ok(())
    }

    fn require_forward_edge(
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

    fn validate_acyclic(&self) -> Result<(), ScheduleLegalityError> {
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

    fn canonicalize(&mut self) {
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

/// Stable legality failure for a backend-neutral schedule transform.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum ScheduleLegalityError {
    /// The persisted schema version is unsupported.
    #[error("schedule schema {found} is unsupported; expected {expected}")]
    UnsupportedVersion {
        /// Persisted version.
        found: u16,
        /// Current version.
        expected: u16,
    },
    /// A required collection is empty.
    #[error("schedule requires a nonempty {0}")]
    Empty(&'static str),
    /// A required dimension or bound is zero.
    #[error("schedule requires a nonzero {0}")]
    Zero(&'static str),
    /// A selected phase does not exist.
    #[error("schedule phase {0:?} does not exist")]
    MissingPhase(SchedulePhaseId),
    /// A logical region does not exist in the selected phase.
    #[error("schedule region {0} does not exist in the selected phase")]
    MissingRegion(u32),
    /// A selected axis does not exist in the selected phase.
    #[error("schedule axis {axis:?} does not exist in phase {phase:?}")]
    MissingAxis {
        /// Referenced phase.
        phase: SchedulePhaseId,
        /// Referenced axis.
        axis: ScheduleAxis,
    },
    /// A split or vector width does not divide its source extent.
    #[error("schedule factor {factor} does not divide extent {extent}")]
    NonDivisible {
        /// Source extent.
        extent: u64,
        /// Requested factor.
        factor: u32,
    },
    /// A phase identity occurs more than once.
    #[error("schedule phase {0:?} occurs more than once")]
    DuplicatePhase(SchedulePhaseId),
    /// A logical region occurs in more than one selected phase.
    #[error("schedule region {0} occurs in more than one phase")]
    DuplicateRegion(u32),
    /// A transform repeats one of its phase operands.
    #[error("schedule transform phase operands must be distinct")]
    DuplicateTransformPhase,
    /// Phase fission would leave an empty phase.
    #[error("schedule fission of phase {0:?} would leave an empty phase")]
    InvalidFission(SchedulePhaseId),
    /// Reorder is not a complete axis permutation.
    #[error("schedule reorder is not a permutation of phase {0:?}")]
    InvalidPermutation(SchedulePhaseId),
    /// A pipeline does not contain positive producer and consumer roles.
    #[error("schedule pipeline requires nonzero producer and consumer role groups")]
    InvalidPipelineRoles,
    /// Spatial partitioning used an invocation-level mapping.
    #[error("schedule spatial partition cannot use {0:?}")]
    InvalidPartitionLevel(MappingLevel),
    /// A resource bound overflowed its representation.
    #[error("schedule resource bound `{0}` overflowed")]
    ResourceOverflow(&'static str),
    /// Allocating a new phase identity overflowed.
    #[error("schedule phase identity overflowed")]
    PhaseIdOverflow,
    /// A selected dependency introduces a cycle.
    #[error("schedule dependency {from:?} -> {to:?} is cyclic")]
    DependencyCycle {
        /// Source phase.
        from: SchedulePhaseId,
        /// Destination phase.
        to: SchedulePhaseId,
    },
    /// An applied transform lacks typed source or inverse provenance.
    #[error("schedule transform lacks typed source or inverse provenance")]
    MissingProvenance,
    /// A persisted transform's typed proof differs from deterministic replay.
    #[error("schedule transform {0} has invalid precondition or provenance evidence")]
    InvalidTransformProof(usize),
    /// Persisted final phases or resources differ from deterministic replay.
    #[error("schedule final state differs from deterministic transform replay")]
    ReplayMismatch,
    /// Canonical schedule identity serialization failed.
    #[error("schedule identity encoding failed: {0}")]
    Identity(String),
}
