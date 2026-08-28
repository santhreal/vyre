//! Constraint propagation over derived candidates.
//!
//! Every candidate the grammar derives passes through here before anything is
//! ranked, compiled, or measured. A candidate that survives is legal on the
//! authenticated target facts; a candidate that does not carries one stable
//! reason, so a certificate can state why a family disappeared instead of
//! reporting a smaller search.
//!
//! Illegality is rejected here, never priced. A budget the device exceeds by
//! running more resident passes is a cost and belongs to the cost model; a
//! launch the device cannot issue, a ring that cannot be held, or a plan the
//! artifact cannot represent is eliminated.

use vyre_foundation::{
    algebraic_reordering::ReorderingClass,
    ir::ProgramGraph,
    schedule::{CombineOrder, MappingLevel, SchedulePhaseId, ScheduleTransform},
};

use crate::{
    candidate::CandidatePlan,
    certificate::PruneReason,
    facts::PlanningFacts,
    legality::{
        analyze_fusion_pair, analyze_topology_legality, FusionDecision, FusionRejectionReason,
        TopologyDecision, TopologyRejectionReason,
    },
    DependencyEdge, DeviceFacts, FusionGroupId,
};

/// Everything the constraint stage reads about one compilation.
pub(crate) struct ConstraintContext<'a> {
    /// Validated graph the candidate schedules.
    pub(crate) graph: &'a ProgramGraph,
    /// Per-node and per-value planning measurements.
    pub(crate) facts: &'a PlanningFacts,
    /// Canonical dependency edges of the graph.
    pub(crate) dependencies: &'a [DependencyEdge],
    /// Authenticated capabilities of the compile target.
    pub(crate) device: DeviceFacts,
}

/// Whether one derived candidate is legal, and why not when it is not.
pub(crate) fn admit(
    candidate: &CandidatePlan,
    context: &ConstraintContext<'_>,
) -> Result<(), PruneReason> {
    schedule_legality(candidate)?;
    dependence(candidate, context)?;
    fusion(candidate, context)?;
    topology(candidate, context)?;
    numerical(candidate, context)?;
    reordering(candidate, context)?;
    launch(candidate, context)?;
    storage(candidate, context)?;
    pipeline(candidate, context)?;
    progress(candidate, context)?;
    target_facts(candidate, context)?;
    representation(candidate)?;
    workspace(candidate, context)
}

/// A transform whose preconditions failed leaves the candidate illegal.
fn schedule_legality(candidate: &CandidatePlan) -> Result<(), PruneReason> {
    match candidate.schedule_error() {
        Some(_) => Err(PruneReason::ScheduleLegality),
        None => Ok(()),
    }
}

/// The grouping must keep the graph's dependence order executable in stages.
fn dependence(
    candidate: &CandidatePlan,
    context: &ConstraintContext<'_>,
) -> Result<(), PruneReason> {
    let groups = candidate
        .node_groups
        .iter()
        .copied()
        .map(FusionGroupId)
        .collect::<Vec<_>>();
    crate::group_stages(candidate.group_count(), context.dependencies, &groups)
        .map(|_| ())
        .map_err(|_| PruneReason::Dependence)
}

/// Every internalized dataflow edge must be legal to internalize.
fn fusion(candidate: &CandidatePlan, context: &ConstraintContext<'_>) -> Result<(), PruneReason> {
    for edge in &candidate.fused_edges {
        if let FusionDecision::Rejected(reason) =
            analyze_fusion_pair(context.graph, edge.from, edge.to, edge.value)
        {
            return Err(fusion_reason(reason));
        }
    }
    Ok(())
}

/// Stable constraint class of one fusion rejection.
fn fusion_reason(reason: FusionRejectionReason) -> PruneReason {
    match reason {
        FusionRejectionReason::UnknownGraphMember
        | FusionRejectionReason::NotProducerConsumer
        | FusionRejectionReason::DependencyCycle => PruneReason::Dependence,
        FusionRejectionReason::LifecycleBoundary => PruneReason::AliasOrEffect,
        FusionRejectionReason::MultipleConsumers => PruneReason::Representation,
        FusionRejectionReason::WorkgroupMismatch
        | FusionRejectionReason::SynchronizationBoundary => PruneReason::BarrierVisibility,
    }
}

/// The proposed execution topology must be legal on the graph and the device.
fn topology(candidate: &CandidatePlan, context: &ConstraintContext<'_>) -> Result<(), PruneReason> {
    match analyze_topology_legality(
        candidate,
        context.graph,
        context.facts,
        context.dependencies,
        context.device,
    ) {
        TopologyDecision::Legal => Ok(()),
        TopologyDecision::Rejected(reason) => Err(topology_reason(reason)),
    }
}

/// Stable constraint class of one topology rejection.
fn topology_reason(reason: TopologyRejectionReason) -> PruneReason {
    match reason {
        TopologyRejectionReason::InsufficientConcurrentQueues
        | TopologyRejectionReason::InsufficientComputeUnits
        | TopologyRejectionReason::UnenforceableSpatialMasking => PruneReason::TargetFacts,
        TopologyRejectionReason::RequiresCooperativeLaunch => PruneReason::Progress,
        TopologyRejectionReason::ResourceConflict
        | TopologyRejectionReason::ControlDependencyOrEffect => PruneReason::AliasOrEffect,
        TopologyRejectionReason::IllegalAsymmetricJoin => PruneReason::BarrierVisibility,
        TopologyRejectionReason::NoIndependentConcurrency => PruneReason::Representation,
        TopologyRejectionReason::OccupancyExceeded => PruneReason::Occupancy,
    }
}

/// A transform that reshapes execution must not reshape the result.
///
/// A program that reasons about the size of its own workgroup observes the shape
/// it runs at, so vectorizing, tiling, splitting, or rewidening its phase would
/// compute something else. The semantic IR owner classifies that observability
/// and every stage reads the same classification.
fn numerical(
    candidate: &CandidatePlan,
    context: &ConstraintContext<'_>,
) -> Result<(), PruneReason> {
    for record in &candidate.schedule.transforms {
        let phase = match &record.transform {
            ScheduleTransform::Vectorize { phase, .. }
            | ScheduleTransform::Tile { phase, .. }
            | ScheduleTransform::Split { phase, .. } => *phase,
            ScheduleTransform::SetWorkgroup { phase, shape } => {
                if reshapes_any_node(candidate, context, *phase, *shape) {
                    *phase
                } else {
                    continue;
                }
            }
            _ => continue,
        };
        if !phase_tolerates_reshape(candidate, context, phase) {
            return Err(PruneReason::Numerical);
        }
    }
    Ok(())
}

/// A transform that changes combine order must combine reorderably.
///
/// A work queue, a spatial partition, a pipeline, and an asymmetric join let
/// invocations reach a shared accumulator in an order the schedule does not fix,
/// and splitting or remapping an axis changes which invocations reach it at all.
/// Over a rounding accumulation that computes a different number from the one
/// the graph states, and the difference is data-dependent, so it is eliminated
/// here rather than reported as an accuracy result.
fn reordering(
    candidate: &CandidatePlan,
    context: &ConstraintContext<'_>,
) -> Result<(), PruneReason> {
    for record in &candidate.schedule.transforms {
        let phases = record.provenance.source_phases.as_slice();
        match record.transform.combine_order() {
            CombineOrder::Preserved => continue,
            CombineOrder::ChangedWhenReshaped(shape) => {
                if !phases
                    .iter()
                    .any(|phase| reshapes_any_node(candidate, context, *phase, shape))
                {
                    continue;
                }
            }
            CombineOrder::Changed => {}
        }
        for phase in phases {
            if !phase_permits_reordering(candidate, context, *phase) {
                return Err(PruneReason::Numerical);
            }
        }
    }
    Ok(())
}

/// Whether every node one phase covers combines reorderably.
///
/// A phase covering no known region permits reordering: there is no combine to
/// reorder. A region the facts do not describe does not, because an unclassified
/// program is not a proof.
fn phase_permits_reordering(
    candidate: &CandidatePlan,
    context: &ConstraintContext<'_>,
    phase: SchedulePhaseId,
) -> bool {
    let Some(regions) = covered_regions(candidate, phase) else {
        return true;
    };
    regions.iter().all(|region| {
        context
            .facts
            .node_reordering
            .get(*region as usize)
            .copied()
            .unwrap_or(ReorderingClass::Ordered)
            .permits_reordering()
    })
}

/// Whether a selected shape differs from a shape some covered node declared.
///
/// Freezing a phase at the shape its own nodes declared reshapes nothing, so it
/// is not a numerical question. Only a shape a node did not declare is.
fn reshapes_any_node(
    candidate: &CandidatePlan,
    context: &ConstraintContext<'_>,
    phase: SchedulePhaseId,
    shape: [u32; 3],
) -> bool {
    covered_regions(candidate, phase).is_some_and(|regions| {
        regions.iter().any(|region| {
            context
                .facts
                .node_declared_workgroup
                .get(*region as usize)
                .copied()
                .is_some_and(|declared| declared != shape)
        })
    })
}

/// Whether every node one phase covers stays correct under a reshape.
fn phase_tolerates_reshape(
    candidate: &CandidatePlan,
    context: &ConstraintContext<'_>,
    phase: SchedulePhaseId,
) -> bool {
    let Some(regions) = covered_regions(candidate, phase) else {
        return true;
    };
    regions.iter().all(|region| {
        context
            .facts
            .node_accepts_width
            .get(*region as usize)
            .copied()
            .unwrap_or(false)
    })
}

/// Source regions one selected phase covers.
fn covered_regions<'a>(candidate: &'a CandidatePlan, phase: SchedulePhaseId) -> Option<&'a [u32]> {
    candidate
        .schedule
        .phases
        .iter()
        .find(|item| item.id == phase)
        .map(|item| item.source_regions.as_slice())
}

/// Every selected phase must be launchable at the shape it selected.
fn launch(candidate: &CandidatePlan, context: &ConstraintContext<'_>) -> Result<(), PruneReason> {
    let limit = u64::from(context.device.max_invocations_per_workgroup());
    if limit == 0 {
        return Ok(());
    }
    for phase in &candidate.schedule.phases {
        let invocations = u64::from(phase.workgroup[0])
            .saturating_mul(u64::from(phase.workgroup[1]))
            .saturating_mul(u64::from(phase.workgroup[2]));
        if invocations > limit {
            return Err(PruneReason::Occupancy);
        }
    }
    Ok(())
}

/// Storage a phase reserves must fit the storage the device grants.
fn storage(candidate: &CandidatePlan, context: &ConstraintContext<'_>) -> Result<(), PruneReason> {
    let granted = u64::from(context.device.shared_scratch_bytes_per_workgroup());
    if granted == 0 {
        return Ok(());
    }
    for phase in &candidate.schedule.phases {
        if phase.resources.shared_bytes > granted {
            return Err(PruneReason::Scratch);
        }
    }
    Ok(())
}

/// A bounded ring must fit in the storage that holds it.
fn pipeline(candidate: &CandidatePlan, context: &ConstraintContext<'_>) -> Result<(), PruneReason> {
    let granted = u64::from(context.device.shared_scratch_bytes_per_workgroup());
    if granted == 0 {
        return Ok(());
    }
    for record in &candidate.schedule.transforms {
        let ScheduleTransform::Pipeline {
            producer,
            ring_slots,
            ..
        } = &record.transform
        else {
            continue;
        };
        let slot_bytes = phase_value_bytes(candidate, context, producer.0);
        let held = slot_bytes.saturating_mul(u64::from(*ring_slots));
        if held > granted {
            return Err(PruneReason::PipelineCapacity);
        }
    }
    Ok(())
}

/// Largest value one phase moves, in bytes.
fn phase_value_bytes(
    candidate: &CandidatePlan,
    context: &ConstraintContext<'_>,
    phase: u32,
) -> u64 {
    let Some(selected) = candidate
        .schedule
        .phases
        .iter()
        .find(|item| item.id.0 == phase)
    else {
        return 0;
    };
    context
        .facts
        .dataflow
        .iter()
        .filter(|edge| selected.source_regions.contains(&edge.from.0))
        .filter_map(|edge| context.facts.value_bytes.get(&edge.value.0).copied())
        .max()
        .unwrap_or(0)
}

/// A resident schedule needs a device that guarantees forward progress.
fn progress(candidate: &CandidatePlan, context: &ConstraintContext<'_>) -> Result<(), PruneReason> {
    if context.device.supports_cooperative_launch() {
        return Ok(());
    }
    let resident = candidate
        .schedule
        .transforms
        .iter()
        .any(|record| matches!(record.transform, ScheduleTransform::PersistentQueue { .. }));
    if resident {
        return Err(PruneReason::Progress);
    }
    Ok(())
}

/// A capability no authenticated fact reports is not available to a plan.
///
/// Partitioning is absent here on purpose. A schedule that partitions or runs
/// resident reports a resident topology, and topology legality proves the
/// partition count and the masking capability against the device facts, so a
/// second copy of that proof here could only certify what it never checked.
fn target_facts(
    candidate: &CandidatePlan,
    context: &ConstraintContext<'_>,
) -> Result<(), PruneReason> {
    for record in &candidate.schedule.transforms {
        if let ScheduleTransform::Map { level, .. } = &record.transform {
            mapping_level_granted(*level, context.device)?;
        }
    }
    Ok(())
}

/// Whether the device facts grant the hierarchy level a mapping selected.
///
/// A device partition is never granted: the artifact describes one device, so an
/// axis mapped across devices has no launch to be recorded in.
pub(crate) fn mapping_level_granted(
    level: MappingLevel,
    device: DeviceFacts,
) -> Result<(), PruneReason> {
    let granted = match level {
        MappingLevel::Lane | MappingLevel::Workgroup => true,
        MappingLevel::Subgroup => {
            device.capabilities().supports_subgroup_ops && device.subgroup_size() > 0
        }
        MappingLevel::ComputeUnitPartition => device.compute_units() >= 2,
        MappingLevel::DevicePartition => false,
    };
    if granted {
        Ok(())
    } else {
        Err(PruneReason::TargetFacts)
    }
}

/// The artifact must be able to record what the candidate selected.
///
/// Recomputing a value means the node that produces it runs inside more than one
/// generated kernel. An artifact assigns each node to exactly one fusion group,
/// so a recomputing candidate has no artifact to be recorded in and is
/// eliminated here rather than recorded as a transform nothing performs.
fn representation(candidate: &CandidatePlan) -> Result<(), PruneReason> {
    let recomputes = candidate
        .schedule
        .transforms
        .iter()
        .any(|record| matches!(record.transform, ScheduleTransform::Recompute { .. }));
    if recomputes {
        return Err(PruneReason::Representation);
    }
    Ok(())
}

/// Values crossing generated kernels must be addressable together.
fn workspace(
    candidate: &CandidatePlan,
    context: &ConstraintContext<'_>,
) -> Result<(), PruneReason> {
    let mut total = 0_u64;
    for edge in &context.facts.dataflow {
        let producer = candidate.node_groups.get(edge.from.0 as usize);
        let consumer = candidate.node_groups.get(edge.to.0 as usize);
        if producer == consumer {
            continue;
        }
        let bytes = context
            .facts
            .value_bytes
            .get(&edge.value.0)
            .copied()
            .unwrap_or(0);
        total = total
            .checked_next_multiple_of(crate::allocation::REGION_ALIGNMENT)
            .and_then(|aligned| aligned.checked_add(bytes))
            .ok_or(PruneReason::Workspace)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use vyre_foundation::{schedule::MappingLevel, validate::BackendCapabilities};

    use super::mapping_level_granted;
    use crate::{certificate::PruneReason, grammar::MAPPING_LEVELS, DeviceFacts};

    /// WHY: every hierarchy level is answered by its own device fact, and the
    /// reason for refusing one must not be produced by the arm that answers
    /// another. The level space is `MAPPING_LEVELS` and the expectation is an
    /// exhaustive match with no catch-all arm, so a level added to the schedule
    /// IR turns this red until a fact is recorded for it.
    #[test]
    fn every_mapping_level_is_answered_by_its_own_device_fact() {
        let bare = DeviceFacts::new(BackendCapabilities::default(), 256);
        let subgroup = DeviceFacts::new(
            BackendCapabilities {
                supports_subgroup_ops: true,
                ..BackendCapabilities::default()
            },
            256,
        )
        .with_subgroup_size(32);
        let units = DeviceFacts::new(BackendCapabilities::default(), 256).with_compute_units(8);
        for device in [bare, subgroup, units] {
            for level in MAPPING_LEVELS {
                let expected = match level {
                    MappingLevel::Lane | MappingLevel::Workgroup => true,
                    MappingLevel::Subgroup => {
                        device.capabilities().supports_subgroup_ops && device.subgroup_size() > 0
                    }
                    MappingLevel::ComputeUnitPartition => device.compute_units() >= 2,
                    MappingLevel::DevicePartition => false,
                };
                let verdict = mapping_level_granted(*level, device);
                assert_eq!(
                    verdict.is_ok(),
                    expected,
                    "level {level:?} on a device with {} units and subgroup width {}",
                    device.compute_units(),
                    device.subgroup_size()
                );
                if !expected {
                    assert_eq!(verdict, Err(PruneReason::TargetFacts));
                }
            }
        }
    }
}
