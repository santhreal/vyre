//! The compile seam: rank the legal candidates, assemble the winner, and the
//! measured path that times finalists on the live device.

use std::time::Instant;

use crate::certificate::{PruneReason, SearchCertificate};
use crate::envelope::TargetPayload;
use crate::error::{failure, overflow, serialization_failure, CompileError, CompilerFailureKind};
use crate::grammar::ScheduleProduction;
use crate::identity::domain_digest;
use crate::identity::{ArtifactNodeId, DependencyEdge, Digest};
use crate::measure::{
    self, CandidateMeasurement, MeasurementEnvironment, MeasurementProtocol, MeasurementRecord,
    ReplacementVerdict, SampleEstimate,
};
use crate::request::{SearchWork, ValidatedCompileRequest};
use crate::request_identity::{
    RequestIdentity, REQUEST_DIGEST_DOMAIN, SEMANTIC_DIGEST_DOMAIN, SOURCE_DIGEST_DOMAIN,
};
use crate::resource_records::{build_abi, build_resources};
use crate::schema::encode_payload;
use crate::schema::{
    Artifact, ArtifactPayload, FusionRejection, GeometryRecord, NodeRecord, PlanMeasurement,
    Provenance, ARTIFACT_SCHEMA_VERSION,
};
use crate::target::{TargetCompileError, TargetCompiler};
use crate::{allocation, artifact, candidate, cost, facts, mesh, normalize, search, select};

/// Everything one compilation derives once and every finalist reuses.
struct CompileContext<'a> {
    logical: vyre_foundation::logical::LogicalProgramGraph<'a>,
    source_graph: Digest,
    semantic_graph: Digest,
    nodes: Vec<NodeRecord>,
    dependencies: Vec<DependencyEdge>,
    facts: facts::PlanningFacts,
    ranked: Vec<select::Selection>,
    /// Legal candidates no other candidate dominates on the metrics the
    /// objective orders by.
    pareto_frontier: u32,
    pruned_fusions: Vec<FusionRejection>,
    certificate: SearchCertificate,
    /// Every placement of this program on the authenticated mesh, single device
    /// first and never pruned.
    placements: Vec<mesh::MeshTopologyPlan>,
    work: SearchWork,
}

/// Rank every legal candidate for one validated request.
fn prepare(request: &ValidatedCompileRequest) -> Result<CompileContext<'_>, CompileError> {
    let logical = vyre_foundation::logical::LogicalProgramGraph::validate(
        &request.graph,
        &request.facts.symbolic_bindings,
    )
    .map_err(|error| {
        failure(
            CompilerFailureKind::InvalidProgram,
            "request.logical",
            error.to_string(),
            "supply a graph with bounded, compatible logical domains",
        )
    })?;
    let source_graph = domain_digest(SOURCE_DIGEST_DOMAIN, logical.semantic_wire());
    let semantic_graph = domain_digest(
        SEMANTIC_DIGEST_DOMAIN,
        &logical.graph().to_wire().map_err(|error| {
            failure(
                CompilerFailureKind::InvalidProgram,
                "request.graph",
                error.to_string(),
                "supply a graph whose values and contracts serialize canonically",
            )
        })?,
    );
    let nodes = logical
        .graph()
        .nodes()
        .iter()
        .map(|node| {
            let program = node.program.canonical_wire_bytes().map_err(|error| {
                failure(
                    CompilerFailureKind::InvalidProgram,
                    format!("request.graph.nodes[{}].program", node.id.0),
                    error.to_string(),
                    "supply canonical-wire-compatible typed IR",
                )
            })?;
            Ok(NodeRecord {
                id: ArtifactNodeId(node.id.0),
                name: node.name.clone(),
                program,
            })
        })
        .collect::<Result<Vec<_>, CompileError>>()?;
    let normalized = normalize::normalize(logical.graph())?;
    let dependencies = normalized.dependencies;
    let planning_facts = facts::derive(&logical, &dependencies, &request.facts.symbolic_bindings)?;
    let placements = mesh::candidates(&logical, request.mesh())?;
    let search = search::explore(
        &logical,
        &planning_facts,
        &dependencies,
        request.search_budget,
        request.device,
        &request.objective,
    );
    let mut certificate = search.certificate;
    let ranked = select::rank(
        search.candidates,
        &planning_facts,
        &dependencies,
        request.device,
        &request.objective,
        &mut certificate,
    );
    certificate.canonicalize();
    if let Some(violation) = ranked.refused {
        return Err(failure(
            CompilerFailureKind::ObjectiveBoundViolated,
            "request.objective.bounds",
            violation.statement(),
            "raise the bound the objective states, state a different primary metric, or reduce the source graph",
        ));
    }
    if ranked.admitted.is_empty() {
        return Err(failure(
            CompilerFailureKind::InvalidSearchBudget,
            "search.candidates",
            "schedule search scored no candidate plan",
            "raise the candidate bound so the unfused baseline plan is explored",
        ));
    }
    let pruned_fusions = search
        .rejected
        .into_iter()
        .map(|rejection| FusionRejection {
            from: rejection.edge.from,
            to: rejection.edge.to,
            value: rejection.edge.value,
            reason: rejection.reason,
        })
        .collect();
    Ok(CompileContext {
        logical,
        source_graph,
        semantic_graph,
        nodes,
        dependencies,
        facts: planning_facts,
        pareto_frontier: frontier_width(&ranked.admitted),
        ranked: ranked.admitted,
        pruned_fusions,
        certificate,
        placements,
        work: search.work,
    })
}

/// Legal candidates in one ranking no other candidate dominates.
///
/// Recorded in the artifact, so a reader can tell a selection the objective had
/// to order from one the legal set decided on its own: a frontier of one means
/// no other legal plan traded a metric for another, and a wide frontier means
/// the tie breakers and bounds are what chose.
fn frontier_width(ranked: &[select::Selection]) -> u32 {
    let width = ranked
        .iter()
        .filter(|selection| selection.on_frontier)
        .count();
    u32::try_from(width).unwrap_or(u32::MAX)
}

/// Turn one ranked candidate into a complete canonical artifact.
fn assemble(
    request: &ValidatedCompileRequest,
    context: &CompileContext<'_>,
    selection: &select::Selection,
    certificate: &SearchCertificate,
    work: SearchWork,
    measurement: PlanMeasurement,
) -> Result<Artifact, CompileError> {
    let artifact::ArtifactPlan {
        node_groups,
        stages,
        geometry,
        selected_plan,
    } = artifact::plan(artifact::PlanInputs {
        logical: &context.logical,
        dependencies: &context.dependencies,
        facts: &context.facts,
        selection,
        pruned_fusions: &context.pruned_fusions,
        certificate,
        external: &request.facts,
        device: request.device,
        budget: request.search_budget,
        work,
        measurement,
        pareto_frontier: context.pareto_frontier,
    })?;
    let (resources, resource_envelope) = build_resources(
        &request.graph,
        &request.facts.symbolic_bindings,
        &node_groups,
        &stages,
    )?;
    let abi = build_abi(&request.graph)?;
    let ranked_peak = selected_plan.selection_cost.planned_peak_bytes;
    let placement = mesh::choose(
        &context.placements,
        &request.objective,
        selected_plan.selection_cost.total,
        ranked_peak,
    );
    let topology = context.placements[placement].clone();
    let allocation = allocation::plan(
        &allocation::value_facts(
            &context.logical,
            &resources,
            &request.facts.symbolic_bindings,
        )?,
        request.device,
        &topology,
    )?;
    let single_device = topology.devices().len() <= 1;
    if single_device && allocation.aggregate_peak_bytes != ranked_peak {
        return Err(failure(
            CompilerFailureKind::InvalidAllocationPlan,
            "artifact.allocation.aggregate_peak_bytes",
            format!(
                "the assembled plan holds {} bytes and ranking priced {ranked_peak}",
                allocation.aggregate_peak_bytes
            ),
            "price peak memory from the same liveness the plan is packed against",
        ));
    }
    // A partition distributes the same bytes over more devices, so the mesh
    // never holds more than one device would. Holding more means a share was
    // invented rather than cut.
    if !single_device && allocation.aggregate_peak_bytes > ranked_peak {
        return Err(failure(
            CompilerFailureKind::InvalidAllocationPlan,
            "artifact.allocation.aggregate_peak_bytes",
            format!(
                "the mesh holds {} bytes and one device would hold {ranked_peak}",
                allocation.aggregate_peak_bytes
            ),
            "cut every value's bytes across its shards instead of copying them",
        ));
    }
    topology.verify_capacity(request.mesh(), &allocation.device_peaks)?;
    let nodes = frozen_nodes(&context.nodes, &geometry)?;
    let request_bytes =
        serde_json::to_vec(&RequestIdentity::from(request)).map_err(serialization_failure)?;
    let provenance = Provenance {
        source_graph: context.source_graph,
        semantic_graph: context.semantic_graph,
        request: domain_digest(REQUEST_DIGEST_DOMAIN, &request_bytes),
        objective: request.objective,
        compiler_version: env!("CARGO_PKG_VERSION").to_string(),
    };
    let payload = ArtifactPayload {
        schema_version: ARTIFACT_SCHEMA_VERSION,
        nodes,
        dependencies: context.dependencies.clone(),
        selected_plan,
        abi,
        resources,
        resource_envelope,
        geometry,
        allocation,
        topology,
        provenance,
    };
    let framed = encode_payload(&payload)?;
    let byte_len = u64::try_from(framed.bytes.len())
        .map_err(|_| overflow("artifact", "artifact length exceeds u64"))?;
    let artifact_bytes_limit = request.max_artifact_bytes();
    if byte_len > artifact_bytes_limit {
        return Err(failure(
            CompilerFailureKind::ArtifactLimit,
            "artifact",
            format!("canonical artifact is {byte_len} bytes; limit is {artifact_bytes_limit}"),
            "raise the artifact-byte bound the objective states or reduce the source graph",
        ));
    }
    Ok(Artifact {
        payload,
        digest: Digest(framed.digest),
    })
}

/// Re-encode every node program at the geometry the search selected.
///
/// The workgroup a source program declares is an input to the search. Leaving
/// the declared shape in the artifact made target compilation rewrite it during
/// emission, so the bytes the artifact authenticated and the bytes the device
/// ran disagreed on the one field a launch cannot recover from.
fn frozen_nodes(
    nodes: &[NodeRecord],
    geometry: &[GeometryRecord],
) -> Result<Vec<NodeRecord>, CompileError> {
    nodes
        .iter()
        .map(|record| {
            let selected = geometry
                .iter()
                .find(|entry| entry.node == record.id)
                .ok_or_else(|| {
                    failure(
                        CompilerFailureKind::InvalidProgram,
                        format!("planner.geometry[{}]", record.id.0),
                        "node has no selected launch geometry",
                        "report the compiler defect",
                    )
                })?;
            let mut program =
                vyre_foundation::ir::Program::from_wire(&record.program).map_err(|error| {
                    failure(
                        CompilerFailureKind::InvalidProgram,
                        format!("planner.nodes[{}].program", record.id.0),
                        error.to_string(),
                        "report the compiler defect",
                    )
                })?;
            if program.workgroup_size == selected.workgroup_size {
                return Ok(record.clone());
            }
            program.set_workgroup_size(selected.workgroup_size);
            let program = program.canonical_wire_bytes().map_err(|error| {
                failure(
                    CompilerFailureKind::InvalidProgram,
                    format!("planner.nodes[{}].program", record.id.0),
                    error.to_string(),
                    "report the compiler defect",
                )
            })?;
            Ok(NodeRecord {
                id: record.id,
                name: record.name.clone(),
                program,
            })
        })
        .collect()
}

/// Compile one validated typed graph into a canonical backend-neutral artifact.
///
/// This path ranks candidates with the open cost model alone and records the
/// winner as [`PlanMeasurement::Unbudgeted`]. A request that budgets on-device
/// measurements is rejected here rather than compiled without spending them:
/// [`compile_measured`] is the only path that can honour that budget.
pub fn compile(request: &ValidatedCompileRequest) -> Result<Artifact, CompileError> {
    if request.search_budget.max_measurements > 0 {
        return Err(failure(
            CompilerFailureKind::InvalidSearchBudget,
            "request.search_budget.max_measurements",
            "analytic compilation cannot spend an on-device measurement budget",
            "compile through compile_measured with a finalist evaluator, or set max_measurements to zero",
        ));
    }
    require_single_artifact(request)?;
    let context = prepare(request)?;
    let selection = first_ranked(&context)?;
    assemble(
        request,
        &context,
        selection,
        &context.certificate,
        context.work,
        PlanMeasurement::Unbudgeted,
    )
}

/// Registers, spill and shared bytes one emitted entry point allocates.
///
/// A target compiler assigns physical registers and decides what spills; a
/// device reports what the loaded module holds. Both figures are measurements of
/// the entry point the compiler is about to time, and both outrank the estimate
/// candidate search derived from the IR. Zero means the backend reported nothing
/// for that term, and the estimate stands for it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EmittedResources {
    /// Registers the entry point allocates per invocation.
    pub registers_per_invocation: u32,
    /// Local-memory spill bytes per invocation.
    pub spill_bytes_per_invocation: u32,
    /// Statically declared workgroup-scoped bytes.
    pub shared_memory_bytes: u32,
    /// Device bytes the loaded module and its bound storage hold while this
    /// entry point runs.
    ///
    /// The selected allocation plan states the bytes the artifact requires to be
    /// resident at once. A device holding fewer than that is not running the plan
    /// the compiler selected, so the figure is reconciled before a measurement
    /// decides anything. Zero means the backend has no memory query and the
    /// planned figure stands unreconciled.
    pub resident_device_bytes: u64,
}

/// Device access the compiler borrows to time its finalists.
///
/// The compiler owns which plans are finalists and how their times are compared.
/// The caller owns the device: it supplies the target compiler that turns one
/// artifact into loadable bytes, the resources each emitted entry point turned
/// out to need, and a launch that returns the device time of one execution.
/// Nothing here acquires a device, so a caller without one calls [`compile`]
/// instead.
pub trait FinalistEvaluator {
    /// Target compiler that turns one candidate artifact into target bytes.
    fn target_compiler(&self) -> &dyn TargetCompiler;

    /// What each entry point of `payload` allocates, in payload entry order.
    ///
    /// The compiler re-ranks its emitted finalists on these figures before it
    /// spends a measurement, so a plan whose real register allocation costs
    /// occupancy is measured after one whose does not. A backend whose API
    /// reports none of it returns one default record per entry, which leaves
    /// every candidate ranked on the analytic estimate.
    ///
    /// # Errors
    ///
    /// Returns an error when the payload cannot be inspected on this device.
    fn resources(
        &self,
        artifact: &Artifact,
        payload: &TargetPayload,
    ) -> Result<Vec<EmittedResources>, TargetCompileError>;

    /// Launch `payload` once and return the device time of that launch in
    /// nanoseconds. The time must come from the device, not the host clock.
    ///
    /// The launch must be complete before this returns. The protocol counts one
    /// sample per call and compares samples across candidates, so a call that
    /// returned while the device was still running would attribute one
    /// candidate's work to whichever candidate the round measured next.
    ///
    /// # Errors
    ///
    /// Returns an error when the payload cannot be launched or timed on this
    /// device.
    fn measure(
        &self,
        artifact: &Artifact,
        payload: &TargetPayload,
    ) -> Result<u64, TargetCompileError>;

    /// Clock, thermal and power state the device reports as the session starts.
    ///
    /// Retained beside the samples so a reader can tell a slow candidate from a
    /// throttled device. A backend whose API reports none of it returns
    /// [`DeviceState::unreported`](crate::measure::DeviceState::unreported), and
    /// the drift the session observes across its own rounds still holds.
    fn device_state(&self) -> measure::DeviceState;
}

/// Compile with the ranked finalists emitted for the target and timed on the
/// device under the versioned measurement protocol.
///
/// Evaluation is a ladder of rising fidelity and rising cost. The symbolic bound
/// eliminates a candidate no descendant can bring under the incumbent, the
/// analytic cost model ranks what survives, emission answers what the target
/// compiler accepts, and device measurement decides among the finalists that
/// emitted. The top `max_target_compilations` ranked plans are emitted; a plan
/// the target compiler rejects is eliminated with
/// [`PruneReason::Emission`](crate::PruneReason::Emission) and the ladder
/// continues, so one unemittable plan no longer ends the compilation.
///
/// Measurement runs [`MeasurementProtocol::V1`] fitted to `max_measurements`,
/// which bounds the launches one candidate receives including its warmup. Every
/// candidate is sampled in every round, in an order rotated per round, and
/// sampling ends when every estimate settles inside the protocol's uncertainty
/// target or the rounds run out. The winner is the lowest trimmed-median
/// estimate, and a later candidate takes the selection from an earlier one only
/// by clearing the equivalence band, so a re-run of the same search on the same
/// device selects the same artifact. Passing the previous artifact's evidence to
/// [`CompileRequest::with_recorded_measurement`](crate::CompileRequest::with_recorded_measurement)
/// extends that guarantee across compilations.
///
/// [`SearchWork::target_compilations`] and [`SearchWork::measurements`] carry the
/// counts actually spent, and the recorded [`PlanMeasurement`] states whether a
/// measurement decided the plan at all: a zero measurement budget records
/// [`PlanMeasurement::Unbudgeted`] and a device with no launch timestamps records
/// [`PlanMeasurement::UntimedDevice`], neither of which is reported as a measured
/// selection. A compilation where no finalist emitted fails with the last
/// emission error rather than returning a plan the target cannot build.
pub fn compile_measured(
    request: &ValidatedCompileRequest,
    evaluator: &dyn FinalistEvaluator,
) -> Result<Artifact, CompileError> {
    require_single_artifact(request)?;
    let context = prepare(request)?;
    let budget = request.search_budget;
    if budget.max_measurements == 0 || budget.max_target_compilations == 0 {
        return assemble(
            request,
            &context,
            first_ranked(&context)?,
            &context.certificate,
            context.work,
            PlanMeasurement::Unbudgeted,
        );
    }
    if !request.device.supports_device_timestamps() {
        return assemble(
            request,
            &context,
            first_ranked(&context)?,
            &context.certificate,
            context.work,
            PlanMeasurement::UntimedDevice,
        );
    }

    let finalists = context
        .ranked
        .len()
        .min(budget.max_target_compilations as usize);
    let started = Instant::now();
    let mut work = context.work;
    let mut certificate = context.certificate.clone();
    let mut emitted: Vec<(usize, Artifact, TargetPayload)> = Vec::new();
    let mut rejection = None;
    for index in 0..finalists {
        if spent(started) >= budget.max_elapsed_ns {
            break;
        }
        let provisional = assemble(
            request,
            &context,
            &context.ranked[index],
            &context.certificate,
            context.work,
            PlanMeasurement::Unbudgeted,
        )?;
        work.target_compilations = work.target_compilations.saturating_add(1);
        match evaluator.target_compiler().compile(&provisional) {
            Ok(payload) => emitted.push((index, provisional, payload)),
            Err(error) => {
                eliminate(&mut certificate, &context.ranked[index]);
                rejection = Some(finalist_failure(index, &error));
            }
        }
    }
    // Emission and load turn every register and shared-byte estimate into a
    // measurement. Re-rank on those before spending a device measurement, so the
    // plan measured first is the plan the reported allocation favours rather than
    // the one the IR estimate favoured.
    let mut reranked: Vec<(u64, usize, usize)> = Vec::with_capacity(emitted.len());
    let mut over_ceiling: Vec<usize> = Vec::new();
    for (position, (index, provisional, payload)) in emitted.iter().enumerate() {
        let reported = evaluator
            .resources(provisional, payload)
            .map_err(|error| finalist_failure(*index, &error))?;
        reconcile_resident_bytes(provisional, &reported)?;
        let candidate = &context.ranked[*index].candidate;
        let groups = reported_groups(&reported, payload, candidate);
        let ceiling = request.device.hardware_registers_per_invocation();
        if ceiling > 0
            && groups
                .iter()
                .any(|group| group.registers_per_invocation > ceiling)
        {
            over_ceiling.push(position);
            continue;
        }
        let cost = cost::evaluate_reported(
            candidate,
            &context.facts,
            &context.dependencies,
            request.device,
            &groups,
        );
        reranked.push((cost.total, *index, position));
    }
    for position in over_ceiling {
        let (index, ..) = &emitted[position];
        eliminate(&mut certificate, &context.ranked[*index]);
    }
    // Total first, then the analytic rank, so a tie on the reported cost keeps
    // the order a caller can reproduce from the certificate.
    reranked.sort_unstable();
    certificate.canonicalize();
    let mut session: Vec<Sampling> = reranked
        .into_iter()
        .map(|(predicted_ns, index, position)| Sampling {
            index,
            position,
            predicted_ns,
            samples: Vec::new(),
        })
        .collect();

    let protocol = MeasurementProtocol::V1.fitted(budget.max_measurements);
    // Warmup is charged against the budget like any counted launch. Its samples
    // are discarded: a first launch measures module load and cold allocation,
    // which is not what distinguishes two schedules.
    for entry in &session {
        if spent(started) >= budget.max_elapsed_ns {
            break;
        }
        let (_, provisional, payload) = &emitted[entry.position];
        for _ in 0..protocol.warmup_launches {
            evaluator
                .measure(provisional, payload)
                .map_err(|error| finalist_failure(entry.index, &error))?;
            work.measurements = work.measurements.saturating_add(1);
        }
    }

    let mut rounds = 0_u32;
    let mut first_round_ns = 0_u64;
    let mut last_round_ns = 0_u64;
    while !protocol.rounds_exhausted(rounds) && !session.is_empty() {
        if spent(started) >= budget.max_elapsed_ns {
            break;
        }
        let mut round = Vec::with_capacity(session.len());
        // Rotate the visit order every round. Measuring the same candidate first
        // every time charges it for whatever the device does at the start of a
        // round, which is how a ranking becomes an artefact of position.
        for offset in 0..session.len() {
            let slot = (rounds as usize + offset) % session.len();
            let entry = &mut session[slot];
            let (_, provisional, payload) = &emitted[entry.position];
            for _ in 0..protocol.repetitions_per_round {
                let sample = evaluator
                    .measure(provisional, payload)
                    .map_err(|error| finalist_failure(entry.index, &error))?;
                entry.samples.push(sample);
                round.push(sample);
                work.measurements = work.measurements.saturating_add(1);
            }
        }
        rounds = rounds.saturating_add(1);
        round.sort_unstable();
        last_round_ns = round.get(round.len() / 2).copied().unwrap_or(0);
        if rounds == 1 {
            first_round_ns = last_round_ns;
        }
        if protocol.rounds_sufficient(rounds)
            && session.iter().all(|entry| settled(entry, &protocol))
        {
            break;
        }
    }
    work.elapsed_ns = work.elapsed_ns.saturating_add(spent(started));

    let mut ranked_indices = Vec::with_capacity(session.len());
    let mut candidates = Vec::with_capacity(session.len());
    for entry in &session {
        let Some(estimate) = SampleEstimate::from_samples(&entry.samples, &protocol) else {
            continue;
        };
        let (_, provisional, _) = &emitted[entry.position];
        ranked_indices.push(entry.index);
        candidates.push(CandidateMeasurement {
            identity: provisional.digest(),
            analytic_rank: u32::try_from(entry.index).unwrap_or(u32::MAX),
            predicted_ns: entry.predicted_ns,
            samples: entry.samples.clone(),
            estimate,
        });
    }
    if candidates.is_empty() {
        return match rejection {
            Some(error) => Err(error),
            None => assemble(
                request,
                &context,
                first_ranked(&context)?,
                &certificate,
                work,
                PlanMeasurement::Unbudgeted,
            ),
        };
    }

    // Candidates are in reported-cost order, so the first is the canonical
    // lower-risk finalist. A later one takes the selection only by clearing the
    // equivalence band, which is what makes two runs of the same search on the
    // same device select the same artifact.
    let mut winner = 0_usize;
    for (slot, candidate) in candidates.iter().enumerate().skip(1) {
        if measure::improves(&candidates[winner].estimate, &candidate.estimate, &protocol) {
            winner = slot;
        }
    }
    let mut record = MeasurementRecord {
        protocol,
        environment: MeasurementEnvironment {
            warmup_launches: protocol.warmup_launches,
            facts_calibration_version: request.device().calibration_version(),
            first_round_ns,
            last_round_ns,
            state: evaluator.device_state(),
        },
        rounds,
        candidates,
        winner: u32::try_from(winner).unwrap_or(u32::MAX),
    };
    if let Some(incumbent) = request.recorded_measurement() {
        if record.verdict_against(incumbent) == ReplacementVerdict::Equivalent {
            if let Some(authenticated) = incumbent.winner().and_then(|authenticated| {
                record
                    .candidates
                    .iter()
                    .position(|candidate| candidate.identity == authenticated.identity)
            }) {
                record.winner = u32::try_from(authenticated).unwrap_or(u32::MAX);
                winner = authenticated;
            }
        }
    }
    assemble(
        request,
        &context,
        &context.ranked[ranked_indices[winner]],
        &certificate,
        work,
        PlanMeasurement::Measured(record),
    )
}

/// One finalist and every counted sample taken against it.
struct Sampling {
    /// Position in the analytic ranking.
    index: usize,
    /// Position in the emitted finalist list.
    position: usize,
    /// Analytic cost the reported-resource ranking predicted, in nanoseconds.
    predicted_ns: u64,
    /// Counted device times in measurement order.
    samples: Vec<u64>,
}

/// Whether this candidate's samples are precise enough to stop sampling it.
fn settled(entry: &Sampling, protocol: &MeasurementProtocol) -> bool {
    SampleEstimate::from_samples(&entry.samples, protocol)
        .is_some_and(|estimate| estimate.is_settled(protocol))
}

/// Nanoseconds the measured path has spent since it started.
fn spent(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// Project what each emitted entry reported onto the fusion groups the cost
/// model prices.
///
/// One entry implements one fusion group, so the entry's node names the group.
/// A report shorter than the entry list leaves the remaining groups at zero,
/// which keeps their analytic estimate. Spill is reported per invocation and
/// priced as traffic, so it is multiplied by the invocations the authenticated
/// geometry launches for that entry.
fn reported_groups(
    reported: &[EmittedResources],
    payload: &TargetPayload,
    candidate: &candidate::CandidatePlan,
) -> Vec<cost::ReportedGroup> {
    let mut groups = vec![cost::ReportedGroup::default(); candidate.group_count()];
    for (entry, resources) in payload.entries().iter().zip(reported) {
        let node = usize::try_from(entry.node.0).unwrap_or(usize::MAX);
        let Some(group) = candidate.node_groups.get(node).copied() else {
            continue;
        };
        let Some(slot) = groups.get_mut(usize::try_from(group).unwrap_or(usize::MAX)) else {
            continue;
        };
        let invocations = entry
            .grid_size
            .iter()
            .chain(entry.workgroup_size.iter())
            .fold(1_u64, |total, extent| {
                total.saturating_mul(u64::from(*extent))
            });
        slot.registers_per_invocation = slot
            .registers_per_invocation
            .max(resources.registers_per_invocation);
        slot.shared_memory_bytes = slot.shared_memory_bytes.max(resources.shared_memory_bytes);
        slot.spill_traffic_bytes = slot.spill_traffic_bytes.saturating_add(
            u64::from(resources.spill_bytes_per_invocation).saturating_mul(invocations),
        );
    }
    groups
}

/// Reconciles the planned resident peak against what the device reports holding.
///
/// The allocation plan states the bytes that must be resident at once for the
/// artifact to run. Every one of those bytes is on the device while an entry
/// point of that artifact runs, so a device reporting fewer bytes than the plan
/// requires is not running the selected plan, and a measurement taken there
/// would rank a schedule nobody compiled. A backend with no memory query reports
/// zero, which leaves the planned figure unreconciled rather than contradicted.
fn reconcile_resident_bytes(
    artifact: &Artifact,
    reported: &[EmittedResources],
) -> Result<(), CompileError> {
    let observed = reported
        .iter()
        .map(|entry| entry.resident_device_bytes)
        .max()
        .unwrap_or(0);
    let planned = artifact.allocation().aggregate_peak_bytes;
    if observed == 0 || observed >= planned {
        return Ok(());
    }
    Err(failure(
        CompilerFailureKind::UnreconciledResidentBytes,
        "measurement.resident_device_bytes",
        format!(
            "the device holds {observed} bytes while the selected allocation plan requires {planned}"
        ),
        "bind the allocation plan the artifact records before measuring it",
    ))
}

/// Record that emission eliminated the family that derived one ranked plan.
///
/// The elimination is charged to the production of the last step that derived
/// the plan, because that step is what made the plan unemittable. A plan with no
/// derivation is the baseline, which is charged to the fusion family it would
/// have been contracted by.
fn eliminate(certificate: &mut SearchCertificate, selection: &select::Selection) {
    let production = selection
        .candidate
        .derivation
        .last()
        .map_or(ScheduleProduction::Fusion, |step| step.production);
    certificate.pruned(production, PruneReason::Emission);
}

/// Refuse a compile that emits one artifact for an objective whose retained set
/// cannot be one artifact.
///
/// The coverage policy states how many artifacts the caller keeps. A path that
/// emits one used to accept a per-class policy and return a single member of the
/// set with nothing recording which member it was, so the refusal names the path
/// that retains a set instead.
fn require_single_artifact(request: &ValidatedCompileRequest) -> Result<(), CompileError> {
    let objective = request.objective();
    let classes = objective.workload().len();
    if objective.portfolio().admits(1, classes) {
        return Ok(());
    }
    Err(failure(
        CompilerFailureKind::PortfolioCoverageUnsatisfied,
        "request.objective.portfolio.coverage",
        format!(
            "coverage `{}` over {classes} workload classes retains {} artifacts and this path emits one",
            objective.portfolio().coverage().name(),
            objective.portfolio().coverage().minimum_variants(classes)
        ),
        "compile through compile_portfolio, or state a coverage policy one artifact satisfies",
    ))
}

fn first_ranked<'a>(
    context: &'a CompileContext<'_>,
) -> Result<&'a select::Selection, CompileError> {
    context.ranked.first().ok_or_else(|| {
        failure(
            CompilerFailureKind::InvalidSearchBudget,
            "search.candidates",
            "schedule search scored no candidate plan",
            "raise the candidate bound so the unfused baseline plan is explored",
        )
    })
}

fn finalist_failure(index: usize, error: &TargetCompileError) -> CompileError {
    failure(
        CompilerFailureKind::FinalistEvaluation,
        format!("search.finalists[{index}]"),
        error.to_string(),
        "supply a finalist evaluator whose target compiler and device accept every ranked plan",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ArtifactNodeId;
    use vyre_foundation::ir::Program;

    fn program(workgroup: [u32; 3]) -> Vec<u8> {
        Program::wrapped(Vec::new(), workgroup, Vec::new())
            .canonical_wire_bytes()
            .expect("fixture program encodes")
    }

    fn record(node: u32, workgroup: [u32; 3]) -> GeometryRecord {
        crate::geometry_fixtures::geometry(node, node, workgroup)
    }

    fn node(id: u32, workgroup: [u32; 3]) -> NodeRecord {
        NodeRecord {
            id: ArtifactNodeId(id),
            name: format!("n{id}"),
            program: program(workgroup),
        }
    }

    /// WHY: the workgroup a source program declares is an input to the search.
    /// Emission used to rewrite it while lowering, so the program the artifact
    /// authenticated and the module the device ran disagreed on the one field a
    /// launch cannot recover from. Freezing happens once, here, and the artifact
    /// carries the result.
    #[test]
    fn a_recorded_program_declares_the_workgroup_the_search_selected() {
        let frozen = frozen_nodes(
            &[node(0, [8, 1, 1]), node(1, [32, 1, 1])],
            &[record(0, [32, 1, 1]), record(1, [32, 1, 1])],
        )
        .expect("both nodes carry selected geometry");

        for record in &frozen {
            let program = Program::from_wire(&record.program).expect("a frozen program decodes");
            assert_eq!(program.workgroup_size, [32, 1, 1], "node {}", record.id.0);
        }
        assert_eq!(
            frozen[1].program,
            program([32, 1, 1]),
            "a program already at the selected shape is carried through unchanged"
        );
    }

    /// WHY: a node with no selected geometry has no shape to be frozen at, and
    /// re-encoding it at its declared shape would reintroduce exactly the
    /// disagreement freezing exists to end.
    #[test]
    fn a_node_without_selected_geometry_is_refused() {
        let error = frozen_nodes(&[node(0, [8, 1, 1])], &[record(1, [32, 1, 1])])
            .expect_err("a node with no geometry cannot be frozen");
        assert_eq!(
            error
                .diagnostic
                .location
                .as_ref()
                .and_then(|location| location.path.as_deref()),
            Some("planner.geometry[0]")
        );
    }
}
