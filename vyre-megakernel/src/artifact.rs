use std::collections::{BTreeMap, BTreeSet};

use vyre_foundation::logical::LogicalProgramGraph;
use vyre_foundation::schedule::{
    PipelineRoleGroup, SchedulePhase, SchedulePhaseId, ScheduleTransform, SelectedSchedule,
};

use crate::{
    build_barriers, build_materializations, certificate::SearchCertificate, domain_digest,
    facts::PlanningFacts, failure, group_stages, select::Selection, ArtifactNodeId,
    ArtifactValueId, BarrierPhaseRecord, CompileError, CompilerFailureKind, DependencyEdge,
    DependencyEndpoint, DeviceFacts, EntryAbiRecord, EntryPersistence, ExecutionMode,
    ExternalFacts, FusionGroupId, FusionRecord, FusionRejection, GeometryRecord,
    LaunchResourceIntent, PlanMeasurement, ResourceLifetime, ResourceRecord, SearchBudget,
    SearchWork, SelectedPlan, WorkspacePlan, WorkspaceRegion, WORKSPACE_REGION_ALIGNMENT,
};

const LEGALITY_DIGEST_DOMAIN: &[u8] = b"VYRE_FUSION_LEGALITY_V1\0";

pub(crate) struct ArtifactPlan {
    pub(crate) node_groups: Vec<FusionGroupId>,
    pub(crate) stages: Vec<u32>,
    pub(crate) geometry: Vec<GeometryRecord>,
    pub(crate) selected_plan: SelectedPlan,
}

/// Everything one candidate needs to become a recorded plan.
pub(crate) struct PlanInputs<'a, 'graph> {
    pub(crate) logical: &'a LogicalProgramGraph<'graph>,
    pub(crate) dependencies: &'a [DependencyEdge],
    pub(crate) facts: &'a PlanningFacts,
    pub(crate) selection: &'a Selection,
    pub(crate) pruned_fusions: &'a [FusionRejection],
    pub(crate) certificate: &'a SearchCertificate,
    pub(crate) external: &'a ExternalFacts,
    pub(crate) device: DeviceFacts,
    pub(crate) budget: SearchBudget,
    pub(crate) work: SearchWork,
    pub(crate) measurement: PlanMeasurement,
    pub(crate) pareto_frontier: u32,
}

pub(crate) fn plan(inputs: PlanInputs<'_, '_>) -> Result<ArtifactPlan, CompileError> {
    let PlanInputs {
        logical,
        dependencies,
        facts,
        selection,
        pruned_fusions,
        certificate,
        external,
        device,
        budget,
        work,
        measurement,
        pareto_frontier,
    } = inputs;
    let graph = logical.graph();
    let candidate = &selection.candidate;
    let node_groups: Vec<FusionGroupId> = candidate
        .node_groups
        .iter()
        .copied()
        .map(FusionGroupId)
        .collect();
    if node_groups.len() != graph.nodes().len() {
        return Err(failure(
            CompilerFailureKind::InvalidProgram,
            "planner.node_groups",
            "planner did not assign every graph node",
            "report the compiler defect",
        ));
    }
    let group_count = candidate.group_count();
    let stages = group_stages(group_count, dependencies, &node_groups)?;
    let fusion = (0..group_count)
        .map(|group| {
            let nodes: Vec<ArtifactNodeId> = node_groups
                .iter()
                .enumerate()
                .filter(|(_, node_group)| node_group.0 as usize == group)
                .map(|(node, _)| ArtifactNodeId(node as u32))
                .collect();
            let accepted_edges = candidate
                .fused_edges
                .iter()
                .filter(|edge| {
                    node_groups.get(edge.from.0 as usize).copied()
                        == Some(FusionGroupId(group as u32))
                        && node_groups.get(edge.to.0 as usize).copied()
                            == Some(FusionGroupId(group as u32))
                })
                .count();
            let evidence = if accepted_edges == 0 {
                b"MKL000_SINGLE_NODE_GROUP".as_slice()
            } else {
                b"MKL000_LEGAL_DATAFLOW".as_slice()
            };
            FusionRecord {
                id: FusionGroupId(group as u32),
                members: nodes,
                stage: stages[group],
                legality: vec![domain_digest(LEGALITY_DIGEST_DOMAIN, evidence)],
            }
        })
        .collect();
    let barriers = build_barriers(dependencies, &node_groups, &stages)?;
    let materializations = build_materializations(graph, &node_groups, &stages);
    let execution = execution_mode(device, external, selection.cost.launches);
    let mut schedule = candidate.selected_schedule(facts).map_err(|error| {
        failure(
            CompilerFailureKind::InvalidProgram,
            "planner.schedule",
            error.to_string(),
            "repair schedule candidate generation before selecting an artifact",
        )
    })?;
    // A persistent route is a schedule decision, so it is applied to the
    // schedule rather than recorded beside it. Recording the mode alone left
    // the queue capacity nowhere, and a consumer that needed one sized it.
    if let ExecutionMode::Persistent { .. } = execution {
        let capacity = u32::try_from(selection.cost.launches.max(1)).unwrap_or(u32::MAX);
        let phases: Vec<SchedulePhaseId> = schedule.phases.iter().map(|phase| phase.id).collect();
        for phase in phases {
            schedule
                .apply(ScheduleTransform::PersistentQueue { phase, capacity })
                .map_err(|error| {
                    failure(
                        CompilerFailureKind::InvalidProgram,
                        "planner.schedule.persistent_queue",
                        error.to_string(),
                        "select a static route when the schedule cannot carry a bounded queue",
                    )
                })?;
        }
    }
    let predecessors = entry_predecessors(dependencies);
    let geometry = graph
        .nodes()
        .iter()
        .map(|node| {
            let node_id = ArtifactNodeId(node.id.0);
            geometry_record(
                node_id,
                &schedule,
                predecessors.get(&node_id).cloned().unwrap_or_default(),
            )
        })
        .collect::<Result<Vec<_>, CompileError>>()?;
    let selected_plan = SelectedPlan {
        topology: candidate.topology(),
        schedule,
        derivation: candidate.derivation.clone(),
        certificate: certificate.clone(),
        fusion,
        barriers,
        materializations,
        candidates_explored: work.candidates_explored,
        pareto_frontier,
        search_budget: budget,
        search_work: work,
        selection_cost: selection.cost,
        pruned_fusions: pruned_fusions.to_vec(),
        execution,
        measurement,
    };
    selected_plan.validate()?;
    Ok(ArtifactPlan {
        node_groups,
        stages,
        geometry,
        selected_plan,
    })
}

/// Decide how the runtime executes this plan.
///
/// One resident kernel polling a device-side work queue replaces the launches a
/// submission batch would otherwise issue: it pays the setup cost once and saves
/// one launch overhead per launch it removes. The trade is profitable only when
/// the overhead removed exceeds the setup paid, and only a device that can hold
/// the whole grid resident can run a kernel that waits on other workgroups, so
/// cooperative launch is a precondition. An unmeasured launch overhead leaves
/// nothing to amortize and selects static execution rather than a guess.
fn execution_mode(
    device: DeviceFacts,
    external: &ExternalFacts,
    launches_per_submission: u64,
) -> ExecutionMode {
    if !device.supports_cooperative_launch() || device.per_launch_overhead_ns() == 0 {
        return ExecutionMode::Static;
    }
    let launches = u128::from(external.expected_launch_batch)
        .saturating_mul(u128::from(launches_per_submission));
    if launches < 2 {
        return ExecutionMode::Static;
    }
    let removed = launches.saturating_mul(u128::from(device.per_launch_overhead_ns()));
    let setup = u128::from(device.persistent_setup_overhead_ns());
    if removed <= setup {
        return ExecutionMode::Static;
    }
    ExecutionMode::Persistent {
        saved_ns: u64::try_from(removed - setup).unwrap_or(u64::MAX),
    }
}

/// Entry-point predecessors implied by the canonical dependency edges.
///
/// Value endpoints are projected onto the nodes that carry them, because a
/// consumer submits entry points and never a value.
fn entry_predecessors(
    dependencies: &[DependencyEdge],
) -> BTreeMap<ArtifactNodeId, Vec<ArtifactNodeId>> {
    let mut predecessors = BTreeMap::<ArtifactNodeId, BTreeSet<ArtifactNodeId>>::new();
    for edge in dependencies {
        if let (DependencyEndpoint::Node(from), DependencyEndpoint::Node(to)) = (edge.from, edge.to)
        {
            if from != to {
                predecessors.entry(to).or_default().insert(from);
            }
        }
    }
    predecessors
        .into_iter()
        .map(|(node, set)| (node, set.into_iter().collect()))
        .collect()
}

/// Project the selected schedule phase covering one node onto its launch record.
fn geometry_record(
    node: ArtifactNodeId,
    schedule: &SelectedSchedule,
    predecessors: Vec<ArtifactNodeId>,
) -> Result<GeometryRecord, CompileError> {
    let phase = schedule.phase_for_region(node.0).ok_or_else(|| {
        failure(
            CompilerFailureKind::InvalidProgram,
            format!("planner.geometry[{}]", node.0),
            "no selected schedule phase covers this node",
            "select a schedule whose phases cover every graph node",
        )
    })?;
    let (roles, ring_slots) = pipeline_assignment(schedule, phase.id);
    let record = GeometryRecord {
        node,
        phase: phase.id,
        predecessors,
        logical_coverage: phase.grid,
        grid: GeometryRecord::covering_grid(phase.grid, phase.workgroup)?,
        workgroup_size: phase.workgroup,
        vector_width: phase.vector_width,
        roles,
        ring_slots,
        barrier_phases: barrier_phases(schedule, phase.id),
        dynamic_shared_bytes: shared_bytes(node, phase)?,
        launch_intent: LaunchResourceIntent {
            private_bytes: phase.resources.private_bytes,
            registers_per_invocation: phase.resources.registers_per_invocation,
        },
        persistence: persistence(schedule, phase.id),
    };
    record.validate()?;
    Ok(record)
}

/// Workgroup-shared bytes the phase reserves, as a launch reads them.
fn shared_bytes(node: ArtifactNodeId, phase: &SchedulePhase) -> Result<u32, CompileError> {
    u32::try_from(phase.resources.shared_bytes).map_err(|_| {
        failure(
            CompilerFailureKind::ResourceOverflow,
            format!("planner.geometry[{}].dynamic_shared_bytes", node.0),
            "selected workgroup-shared bytes exceed the launch limit",
            "place the value in device memory instead of workgroup-shared storage",
        )
    })
}

/// Pipeline roles and ring depth the schedule assigned to one phase.
fn pipeline_assignment(
    schedule: &SelectedSchedule,
    phase: SchedulePhaseId,
) -> (Vec<PipelineRoleGroup>, u32) {
    for record in &schedule.transforms {
        if let ScheduleTransform::Pipeline {
            producer,
            consumer,
            ring_slots,
            roles,
        } = &record.transform
        {
            if *producer == phase || *consumer == phase {
                return (roles.clone(), *ring_slots);
            }
        }
    }
    (Vec::new(), 0)
}

/// Synchronization boundaries the schedule placed across one phase.
fn barrier_phases(schedule: &SelectedSchedule, phase: SchedulePhaseId) -> Vec<BarrierPhaseRecord> {
    schedule
        .transforms
        .iter()
        .filter_map(|record| match &record.transform {
            ScheduleTransform::Synchronize { phases, scope } if phases.contains(&phase) => {
                Some(BarrierPhaseRecord {
                    scope: *scope,
                    phases: phases.clone(),
                })
            }
            _ => None,
        })
        .collect()
}

/// Persistence the schedule selected for one phase.
fn persistence(schedule: &SelectedSchedule, phase: SchedulePhaseId) -> EntryPersistence {
    for record in &schedule.transforms {
        if let ScheduleTransform::PersistentQueue {
            phase: persistent,
            capacity,
        } = record.transform
        {
            if persistent == phase {
                return EntryPersistence::Persistent {
                    queue_capacity: capacity,
                };
            }
        }
    }
    EntryPersistence::Static
}

/// Assign one workspace offset per value the artifact produces inside itself.
///
/// A region is storage the runtime allocates because nothing outside the
/// artifact owns it: some entry point writes the value, and a later entry point
/// or a retained successor reads it. Constants and graph inputs are bound by the
/// caller and outputs are read back by the caller, so none of them is workspace
/// even when their lifetime class matches. The live range is the span of entry
/// points that bind the value.
pub(crate) fn workspace_plan(
    resources: &[ResourceRecord],
    entries: &[EntryAbiRecord],
) -> Result<WorkspacePlan, CompileError> {
    let mut produced = BTreeSet::<ArtifactValueId>::new();
    let mut span = BTreeMap::<ArtifactValueId, (ArtifactNodeId, ArtifactNodeId)>::new();
    for entry in entries {
        produced.extend(entry.outputs.iter().copied());
        for value in entry.inputs.iter().chain(entry.outputs.iter()) {
            span.entry(*value)
                .and_modify(|range| {
                    range.0 = range.0.min(entry.node);
                    range.1 = range.1.max(entry.node);
                })
                .or_insert((entry.node, entry.node));
        }
    }
    let mut plan = WorkspacePlan::default();
    for resource in resources {
        if !matches!(
            resource.lifetime,
            ResourceLifetime::Invocation | ResourceLifetime::Retained
        ) || !produced.contains(&resource.value)
        {
            continue;
        }
        let Some((first_entry, last_entry)) = span.get(&resource.value).copied() else {
            continue;
        };
        if resource.byte_count == 0 {
            continue;
        }
        let offset = plan
            .total_bytes
            .checked_next_multiple_of(WORKSPACE_REGION_ALIGNMENT)
            .ok_or_else(|| {
                failure(
                    CompilerFailureKind::ResourceOverflow,
                    "planner.workspace.offset",
                    "workspace offset exceeds the addressable range",
                    "reduce the retained working set",
                )
            })?;
        plan.total_bytes = offset.checked_add(resource.byte_count).ok_or_else(|| {
            failure(
                CompilerFailureKind::ResourceOverflow,
                "planner.workspace.total_bytes",
                "workspace allocation exceeds the addressable range",
                "reduce the retained working set",
            )
        })?;
        plan.regions.push(WorkspaceRegion {
            value: resource.value,
            offset,
            bytes: resource.byte_count,
            lifetime: resource.lifetime,
            first_entry,
            last_entry,
        });
    }
    plan.validate()?;
    Ok(plan)
}
