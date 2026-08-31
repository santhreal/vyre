//! The versioned schedule grammar bounded candidate search derives plans from.
//!
//! A production names one class of schedule transform. Search expands a
//! candidate by asking every production what it can propose over that
//! candidate's own phases, so the candidate set is derived from the schedule in
//! front of the compiler instead of listed from kernel shapes someone
//! remembered. Operands come from the phase axes, the planning facts, and the
//! authenticated device facts. No target name, no source template, and no
//! per-operation exception participates.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use vyre_foundation::schedule::{
    MappingLevel, MemoryPlacement, PipelineRole, PipelineRoleGroup, ScheduleAxis, SchedulePhase,
    SchedulePhaseId, ScheduleTransform, SelectedSchedule, SynchronizationScope,
};

use crate::facts::PlanningFacts;

/// Version of the production set search derives candidates from.
///
/// A production added, removed, or given different operands moves this version,
/// so a recorded search certificate states which grammar produced it.
pub const SCHEDULE_GRAMMAR_VERSION: u32 = 1;

/// Exact factors every divisible production draws its operands from.
///
/// A factor is proposed only when it divides the axis extent exactly, so the
/// ladder is a bound on how many operands one axis contributes, never a claim
/// that a factor applies.
const FACTOR_LADDER: &[u32] = &[2, 4, 8, 16, 32, 64, 128, 256];

/// Launch widths the launch-width production proposes.
///
/// The set stops at 32 and 256 on purpose. Below 32 a workgroup cannot fill a
/// subgroup on any supported device, and 256 is the widest group any recorded
/// `vyre-bench` case uses (`foundation.reduce.sum.1m` tiles at 256).
pub(crate) const WORKGROUP_WIDTHS: &[u32] = &[32, 64, 128, 256];

/// Vector widths the vectorizing production proposes.
const VECTOR_WIDTHS: &[u32] = &[2, 4];

/// Ring depths the pipelining production proposes.
const RING_SLOTS: &[u32] = &[2, 3];

/// Producer and consumer worker counts the pipelining production proposes.
const ROLE_SPLITS: &[(u32, u32)] = &[(1, 1), (1, 2), (2, 1)];

/// Partition and queue cardinalities the resident productions propose.
pub(crate) const PARTITION_COUNTS: &[u32] = &[2, 4];

/// Distance the prefetching production proposes.
const PREFETCH_DISTANCE: u32 = 1;

/// Axes one phase contributes operands for.
const AXIS_OPERAND_LIMIT: usize = 2;

/// Values one phase contributes operands for.
const VALUE_OPERAND_LIMIT: usize = 4;

/// One production family of the schedule grammar.
///
/// Exactly one family derives each [`ScheduleTransform`] variant, so a transform
/// added to the schedule IR is a compile error here until a family derives it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScheduleProduction {
    /// Contract a producer and a consumer phase into one generated kernel.
    Fusion,
    /// Split one phase into separate phases at a source region boundary.
    Fission,
    /// Select the launch width of every phase that tolerates one.
    LaunchWidth,
    /// Execute one phase across neutral compute partitions.
    SpatialPartition,
    /// Execute one phase through a bounded persistent queue.
    PersistentQueue,
    /// Overlap a producer and a consumer phase through a bounded ring.
    Pipeline,
    /// Force a submission boundary between two phases.
    DispatchCut,
    /// Join several producer phases into one consumer phase.
    AsymmetricJoin,
    /// Place an explicit synchronization boundary across phases.
    Synchronization,
    /// Place one value in a neutral memory class.
    MemoryPlacement,
    /// Fetch one value a bounded distance ahead of its use.
    Prefetch,
    /// Recompute a value instead of materializing it.
    Recomputation,
    /// Tile logical axes of one phase.
    Tiling,
    /// Split one logical axis by an exact factor.
    AxisSplit,
    /// Select vector execution for one logical axis.
    Vectorization,
    /// Map one logical axis onto a neutral hierarchy level.
    AxisMapping,
    /// Reorder the logical axes of one phase.
    AxisReorder,
}

impl ScheduleProduction {
    /// Every production, in the order search expands them.
    ///
    /// The order is the order in which a production can change what the device
    /// executes: kernel organization first, then concurrency and ordering, then
    /// storage, then the intra-phase loop shape. A bounded search that stops
    /// early has spent its budget on the families that change the most.
    pub const ALL: &'static [Self] = &[
        Self::Fusion,
        Self::Fission,
        Self::LaunchWidth,
        Self::SpatialPartition,
        Self::PersistentQueue,
        Self::Pipeline,
        Self::DispatchCut,
        Self::AsymmetricJoin,
        Self::Synchronization,
        Self::MemoryPlacement,
        Self::Prefetch,
        Self::Recomputation,
        Self::Tiling,
        Self::AxisSplit,
        Self::Vectorization,
        Self::AxisMapping,
        Self::AxisReorder,
    ];

    /// The production family that derives `transform`.
    ///
    /// The match is exhaustive with no catch-all arm, which is what closes the
    /// class: a schedule transform the IR gains cannot compile until a
    /// production derives it.
    #[must_use]
    pub fn deriving(transform: &ScheduleTransform) -> Self {
        match transform {
            ScheduleTransform::Fuse { .. } => Self::Fusion,
            ScheduleTransform::PhaseFission { .. } => Self::Fission,
            ScheduleTransform::SetWorkgroup { .. } => Self::LaunchWidth,
            ScheduleTransform::SpatialPartition { .. } => Self::SpatialPartition,
            ScheduleTransform::PersistentQueue { .. } => Self::PersistentQueue,
            ScheduleTransform::Pipeline { .. } => Self::Pipeline,
            ScheduleTransform::DispatchCut { .. } => Self::DispatchCut,
            ScheduleTransform::AsymmetricJoin { .. } => Self::AsymmetricJoin,
            ScheduleTransform::Synchronize { .. } => Self::Synchronization,
            ScheduleTransform::PlaceMemory { .. } => Self::MemoryPlacement,
            ScheduleTransform::Prefetch { .. } => Self::Prefetch,
            ScheduleTransform::Recompute { .. } => Self::Recomputation,
            ScheduleTransform::Tile { .. } => Self::Tiling,
            ScheduleTransform::Split { .. } => Self::AxisSplit,
            ScheduleTransform::Vectorize { .. } => Self::Vectorization,
            ScheduleTransform::Map { .. } => Self::AxisMapping,
            ScheduleTransform::Reorder { .. } => Self::AxisReorder,
        }
    }

    /// Stable machine-readable production code.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::Fusion => "MKP001_FUSION",
            Self::Fission => "MKP002_FISSION",
            Self::LaunchWidth => "MKP003_LAUNCH_WIDTH",
            Self::SpatialPartition => "MKP004_SPATIAL_PARTITION",
            Self::PersistentQueue => "MKP005_PERSISTENT_QUEUE",
            Self::Pipeline => "MKP006_PIPELINE",
            Self::DispatchCut => "MKP007_DISPATCH_CUT",
            Self::AsymmetricJoin => "MKP008_ASYMMETRIC_JOIN",
            Self::Synchronization => "MKP009_SYNCHRONIZATION",
            Self::MemoryPlacement => "MKP010_MEMORY_PLACEMENT",
            Self::Prefetch => "MKP011_PREFETCH",
            Self::Recomputation => "MKP012_RECOMPUTATION",
            Self::Tiling => "MKP013_TILING",
            Self::AxisSplit => "MKP014_AXIS_SPLIT",
            Self::Vectorization => "MKP015_VECTORIZATION",
            Self::AxisMapping => "MKP016_AXIS_MAPPING",
            Self::AxisReorder => "MKP017_AXIS_REORDER",
        }
    }
}

/// One application of one production, with the transforms it applied.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DerivationStep {
    /// Production that proposed this step.
    pub production: ScheduleProduction,
    /// Transforms the step applies, in application order.
    pub transforms: Vec<ScheduleTransform>,
}

/// Facts every production reads to bound its operands.
///
/// Device capabilities are absent on purpose. A production proposes the
/// structures the schedule IR can express, and constraint propagation
/// eliminates the ones the authenticated target facts do not grant, so a
/// certificate can state that a family was considered and why it disappeared.
pub(crate) struct GrammarContext<'a> {
    /// Per-node and per-value planning measurements.
    pub(crate) facts: &'a PlanningFacts,
}

/// Every step `production` can propose over `schedule`.
///
/// The result is deterministic: phases are visited in identity order, operands
/// in ascending order, and nothing reads a hash map iteration order.
pub(crate) fn propose(
    production: ScheduleProduction,
    schedule: &SelectedSchedule,
    context: &GrammarContext<'_>,
) -> Vec<DerivationStep> {
    let transforms = match production {
        ScheduleProduction::Fusion => fusion(schedule),
        ScheduleProduction::Fission => fission(schedule),
        ScheduleProduction::LaunchWidth => launch_width(schedule, context),
        ScheduleProduction::SpatialPartition => spatial_partition(schedule),
        ScheduleProduction::PersistentQueue => persistent_queue(schedule),
        ScheduleProduction::Pipeline => pipeline(schedule),
        ScheduleProduction::DispatchCut => dispatch_cut(schedule),
        ScheduleProduction::AsymmetricJoin => asymmetric_join(schedule),
        ScheduleProduction::Synchronization => synchronization(schedule),
        ScheduleProduction::MemoryPlacement => memory_placement(schedule, context),
        ScheduleProduction::Prefetch => prefetch(schedule, context),
        ScheduleProduction::Recomputation => recomputation(schedule, context),
        ScheduleProduction::Tiling => tiling(schedule),
        ScheduleProduction::AxisSplit => axis_split(schedule),
        ScheduleProduction::Vectorization => vectorization(schedule),
        ScheduleProduction::AxisMapping => axis_mapping(schedule),
        ScheduleProduction::AxisReorder => axis_reorder(schedule),
    };
    transforms
        .into_iter()
        .map(|transforms| DerivationStep {
            production,
            transforms,
        })
        .collect()
}

/// Phases in stable identity order.
///
/// Applying a transform appends the phase it produced, so the stored order is
/// application order and only the identity order is stable across expansions.
fn ordered_phases(schedule: &SelectedSchedule) -> Vec<&SchedulePhase> {
    let mut phases: Vec<&SchedulePhase> = schedule.phases.iter().collect();
    phases.sort_by_key(|phase| phase.id);
    phases
}

/// Producer-consumer phase pairs in ascending order.
fn phase_edges(schedule: &SelectedSchedule) -> Vec<(SchedulePhaseId, SchedulePhaseId)> {
    let mut edges = BTreeSet::new();
    for phase in &schedule.phases {
        for predecessor in &phase.predecessors {
            edges.insert((*predecessor, phase.id));
        }
    }
    edges.into_iter().collect()
}

/// Submission boundaries the schedule already forced.
fn recorded_cuts(schedule: &SelectedSchedule) -> BTreeSet<(SchedulePhaseId, SchedulePhaseId)> {
    schedule
        .transforms
        .iter()
        .filter_map(|record| match record.transform {
            ScheduleTransform::DispatchCut { before, after } => Some((before, after)),
            _ => None,
        })
        .collect()
}

/// Graph values each phase reads or writes, in ascending value order.
fn phase_values(schedule: &SelectedSchedule, facts: &PlanningFacts) -> BTreeMap<u32, Vec<u32>> {
    let mut values = BTreeMap::<u32, BTreeSet<u32>>::new();
    for phase in &schedule.phases {
        let regions = phase
            .source_regions
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let mut touched = BTreeSet::new();
        for edge in &facts.dataflow {
            if regions.contains(&edge.from.0) || regions.contains(&edge.to.0) {
                touched.insert(edge.value.0);
            }
        }
        values.insert(phase.id.0, touched.iter().copied().collect());
    }
    values
        .into_iter()
        .map(|(phase, set)| (phase, set.into_iter().collect()))
        .collect()
}

/// Contract each producer-consumer pair the schedule has not cut.
fn fusion(schedule: &SelectedSchedule) -> Vec<Vec<ScheduleTransform>> {
    let cuts = recorded_cuts(schedule);
    phase_edges(schedule)
        .into_iter()
        .filter(|edge| !cuts.contains(edge))
        .map(|(producer, consumer)| {
            vec![ScheduleTransform::Fuse {
                phases: vec![producer, consumer],
            }]
        })
        .collect()
}

/// Split each phase that covers more than one source region.
fn fission(schedule: &SelectedSchedule) -> Vec<Vec<ScheduleTransform>> {
    ordered_phases(schedule)
        .into_iter()
        .flat_map(|phase| {
            let boundaries = phase.source_regions.len().saturating_sub(1);
            phase
                .source_regions
                .iter()
                .take(boundaries)
                .map(|region| {
                    vec![ScheduleTransform::PhaseFission {
                        phase: phase.id,
                        split_after_region: *region,
                    }]
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Select one launch width for every phase whose nodes tolerate one.
///
/// A node that reasons about the size of its own workgroup holds its phase at
/// the declared shape, so the step covers only the phases the width moves. A
/// width the device cannot issue is eliminated by the launch constraint.
fn launch_width(
    schedule: &SelectedSchedule,
    context: &GrammarContext<'_>,
) -> Vec<Vec<ScheduleTransform>> {
    WORKGROUP_WIDTHS
        .iter()
        .filter_map(|width| {
            let shape = [*width, 1, 1];
            let transforms = ordered_phases(schedule)
                .into_iter()
                .filter(|phase| phase.workgroup != shape)
                .filter(|phase| phase_accepts_width(phase, context.facts))
                .map(|phase| ScheduleTransform::SetWorkgroup {
                    phase: phase.id,
                    shape,
                })
                .collect::<Vec<_>>();
            (!transforms.is_empty()).then_some(transforms)
        })
        .collect()
}

/// Whether every node the phase covers stays correct at another launch width.
fn phase_accepts_width(phase: &SchedulePhase, facts: &PlanningFacts) -> bool {
    phase.source_regions.iter().all(|region| {
        facts
            .node_accepts_width
            .get(*region as usize)
            .copied()
            .unwrap_or(false)
    })
}

/// Spread one phase across neutral compute partitions.
///
/// A device that grants neither the partitioning nor the units is answered by
/// the target-fact constraint, not by refusing to propose the family.
fn spatial_partition(schedule: &SelectedSchedule) -> Vec<Vec<ScheduleTransform>> {
    ordered_phases(schedule)
        .into_iter()
        .flat_map(|phase| {
            PARTITION_COUNTS
                .iter()
                .map(move |partitions| {
                    vec![ScheduleTransform::SpatialPartition {
                        phase: phase.id,
                        partitions: *partitions,
                        level: MappingLevel::ComputeUnitPartition,
                    }]
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Run one phase from a bounded resident queue.
///
/// Forward progress is a device fact, so a device that cannot guarantee it
/// eliminates the family in constraint propagation.
fn persistent_queue(schedule: &SelectedSchedule) -> Vec<Vec<ScheduleTransform>> {
    ordered_phases(schedule)
        .into_iter()
        .flat_map(|phase| {
            PARTITION_COUNTS
                .iter()
                .map(move |capacity| {
                    vec![ScheduleTransform::PersistentQueue {
                        phase: phase.id,
                        capacity: *capacity,
                    }]
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Overlap a producer and a consumer through a bounded ring.
fn pipeline(schedule: &SelectedSchedule) -> Vec<Vec<ScheduleTransform>> {
    phase_edges(schedule)
        .into_iter()
        .filter(|(producer, consumer)| producer < consumer)
        .flat_map(|(producer, consumer)| {
            RING_SLOTS
                .iter()
                .flat_map(move |ring_slots| {
                    ROLE_SPLITS.iter().map(move |(producers, consumers)| {
                        vec![ScheduleTransform::Pipeline {
                            producer,
                            consumer,
                            ring_slots: *ring_slots,
                            roles: vec![
                                PipelineRoleGroup {
                                    role: PipelineRole::Producer,
                                    workers: *producers,
                                },
                                PipelineRoleGroup {
                                    role: PipelineRole::Consumer,
                                    workers: *consumers,
                                },
                            ],
                        }]
                    })
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Force a submission boundary the later fusion productions must respect.
fn dispatch_cut(schedule: &SelectedSchedule) -> Vec<Vec<ScheduleTransform>> {
    let cuts = recorded_cuts(schedule);
    phase_edges(schedule)
        .into_iter()
        .filter(|(producer, consumer)| producer < consumer)
        .filter(|edge| !cuts.contains(edge))
        .map(|(before, after)| vec![ScheduleTransform::DispatchCut { before, after }])
        .collect()
}

/// Join every producer of a phase that has more than one.
fn asymmetric_join(schedule: &SelectedSchedule) -> Vec<Vec<ScheduleTransform>> {
    ordered_phases(schedule)
        .into_iter()
        .filter_map(|phase| {
            let producers = phase
                .predecessors
                .iter()
                .copied()
                .filter(|producer| *producer < phase.id)
                .collect::<Vec<_>>();
            (producers.len() >= 2).then(|| {
                vec![ScheduleTransform::AsymmetricJoin {
                    producers,
                    consumer: phase.id,
                }]
            })
        })
        .collect()
}

/// Place an explicit synchronization boundary across a dependency edge.
fn synchronization(schedule: &SelectedSchedule) -> Vec<Vec<ScheduleTransform>> {
    let scopes = [
        SynchronizationScope::Subgroup,
        SynchronizationScope::Workgroup,
        SynchronizationScope::Device,
    ];
    phase_edges(schedule)
        .into_iter()
        .flat_map(|(producer, consumer)| {
            scopes
                .into_iter()
                .map(move |scope| {
                    vec![ScheduleTransform::Synchronize {
                        phases: vec![producer, consumer],
                        scope,
                    }]
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Place one value the phase touches in a neutral memory class.
///
/// Workgroup-shared storage is proposed for every value the phase touches. A
/// value larger than the shared budget the device reports is eliminated by the
/// storage constraint.
fn memory_placement(
    schedule: &SelectedSchedule,
    context: &GrammarContext<'_>,
) -> Vec<Vec<ScheduleTransform>> {
    let values = phase_values(schedule, context.facts);
    ordered_phases(schedule)
        .into_iter()
        .flat_map(|phase| {
            values
                .get(&phase.id.0)
                .into_iter()
                .flatten()
                .take(VALUE_OPERAND_LIMIT)
                .filter_map(|value| {
                    let bytes = context.facts.value_bytes.get(value).copied()?;
                    (bytes > 0).then_some(vec![ScheduleTransform::PlaceMemory {
                        phase: phase.id,
                        value: *value,
                        placement: MemoryPlacement::Workgroup,
                        bytes,
                    }])
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Fetch one value the phase reads from a predecessor a bounded distance ahead.
fn prefetch(
    schedule: &SelectedSchedule,
    context: &GrammarContext<'_>,
) -> Vec<Vec<ScheduleTransform>> {
    let values = phase_values(schedule, context.facts);
    ordered_phases(schedule)
        .into_iter()
        .filter(|phase| !phase.predecessors.is_empty())
        .flat_map(|phase| {
            values
                .get(&phase.id.0)
                .into_iter()
                .flatten()
                .take(VALUE_OPERAND_LIMIT)
                .filter_map(|value| {
                    let bytes = context.facts.value_bytes.get(value).copied()?;
                    (bytes > 0).then_some(vec![ScheduleTransform::Prefetch {
                        phase: phase.id,
                        value: *value,
                        distance: PREFETCH_DISTANCE,
                        bytes,
                    }])
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Recompute a value the phase reads instead of materializing it.
fn recomputation(
    schedule: &SelectedSchedule,
    context: &GrammarContext<'_>,
) -> Vec<Vec<ScheduleTransform>> {
    let values = phase_values(schedule, context.facts);
    ordered_phases(schedule)
        .into_iter()
        .filter(|phase| !phase.predecessors.is_empty())
        .flat_map(|phase| {
            values
                .get(&phase.id.0)
                .into_iter()
                .flatten()
                .take(VALUE_OPERAND_LIMIT)
                .map(|value| {
                    vec![ScheduleTransform::Recompute {
                        phase: phase.id,
                        values: vec![*value],
                    }]
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Tile the leading axes of one phase by an exact factor.
fn tiling(schedule: &SelectedSchedule) -> Vec<Vec<ScheduleTransform>> {
    ordered_phases(schedule)
        .into_iter()
        .flat_map(|phase| {
            operand_axes(phase)
                .flat_map(move |axis| {
                    exact_factors(axis).map(move |factor| {
                        vec![ScheduleTransform::Tile {
                            phase: phase.id,
                            tiles: vec![(axis, factor)],
                        }]
                    })
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Split one logical axis of a phase by an exact factor.
fn axis_split(schedule: &SelectedSchedule) -> Vec<Vec<ScheduleTransform>> {
    ordered_phases(schedule)
        .into_iter()
        .flat_map(|phase| {
            operand_axes(phase)
                .flat_map(move |axis| {
                    exact_factors(axis).map(move |factor| {
                        vec![ScheduleTransform::Split {
                            phase: phase.id,
                            axis,
                            factor,
                        }]
                    })
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Select vector execution for one logical axis.
fn vectorization(schedule: &SelectedSchedule) -> Vec<Vec<ScheduleTransform>> {
    ordered_phases(schedule)
        .into_iter()
        .flat_map(|phase| {
            operand_axes(phase)
                .flat_map(move |axis| {
                    VECTOR_WIDTHS
                        .iter()
                        .filter(move |width| axis.extent % u64::from(**width) == 0)
                        .filter(move |width| phase.vector_width != **width)
                        .map(move |width| {
                            vec![ScheduleTransform::Vectorize {
                                phase: phase.id,
                                axis,
                                width: *width,
                            }]
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Map one logical axis onto one level of the execution hierarchy.
///
/// A level the device facts do not grant is eliminated by the target-fact
/// constraint, which is what lets a certificate report the level as considered.
fn axis_mapping(schedule: &SelectedSchedule) -> Vec<Vec<ScheduleTransform>> {
    let levels = MAPPING_LEVELS;
    ordered_phases(schedule)
        .into_iter()
        .flat_map(|phase| {
            operand_axes(phase)
                .flat_map(|axis| {
                    levels
                        .iter()
                        .filter(move |level| {
                            !phase
                                .mappings
                                .iter()
                                .any(|mapping| mapping.axis == axis && mapping.level == **level)
                        })
                        .map(move |level| {
                            vec![ScheduleTransform::Map {
                                phase: phase.id,
                                axis,
                                level: *level,
                            }]
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Rotate the logical axes of one phase.
fn axis_reorder(schedule: &SelectedSchedule) -> Vec<Vec<ScheduleTransform>> {
    ordered_phases(schedule)
        .into_iter()
        .filter(|phase| phase.axes.len() >= 2)
        .map(|phase| {
            let mut axes = phase.axes.clone();
            axes.rotate_left(1);
            vec![ScheduleTransform::Reorder {
                phase: phase.id,
                axes,
            }]
        })
        .collect()
}

/// Axes of one phase that contribute transform operands.
fn operand_axes(phase: &SchedulePhase) -> impl Iterator<Item = ScheduleAxis> + '_ {
    let mut axes = phase.axes.clone();
    axes.sort_unstable();
    axes.into_iter().take(AXIS_OPERAND_LIMIT)
}

/// Ladder factors that divide the axis extent exactly and leave a remainder.
fn exact_factors(axis: ScheduleAxis) -> impl Iterator<Item = u32> {
    let extent = axis.extent;
    FACTOR_LADDER
        .iter()
        .copied()
        .filter(move |factor| u64::from(*factor) < extent)
        .filter(move |factor| extent % u64::from(*factor) == 0)
}

/// Every hierarchy level a mapping production proposes.
pub(crate) const MAPPING_LEVELS: &[MappingLevel] = &[
    MappingLevel::Lane,
    MappingLevel::Subgroup,
    MappingLevel::Workgroup,
    MappingLevel::ComputeUnitPartition,
    MappingLevel::DevicePartition,
];

/// Position of one level in `MAPPING_LEVELS`.
///
/// The match has no catch-all arm, so a level added to the schedule IR fails to
/// compile here until the production records a decision for it.
const fn mapping_level_index(level: MappingLevel) -> usize {
    match level {
        MappingLevel::Lane => 0,
        MappingLevel::Subgroup => 1,
        MappingLevel::Workgroup => 2,
        MappingLevel::ComputeUnitPartition => 3,
        MappingLevel::DevicePartition => 4,
    }
}

/// `MAPPING_LEVELS` holds every level the index names, once, in order.
const _: () = {
    let mut index = 0;
    while index < MAPPING_LEVELS.len() {
        assert!(mapping_level_index(MAPPING_LEVELS[index]) == index);
        index += 1;
    }
    assert!(MAPPING_LEVELS.len() == mapping_level_index(MappingLevel::DevicePartition) + 1);
};
