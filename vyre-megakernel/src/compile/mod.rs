//! The compile seam: rank the legal candidates, assemble the winner, and the
//! measured path that times finalists on the live device.

mod measured;
#[cfg(test)]
mod tests;

pub use measured::{compile_measured, EmittedResources, FinalistEvaluator, LaunchObservation};

use crate::certificate::SearchCertificate;
use crate::error::{failure, overflow, serialization_failure, CompileError, CompilerFailureKind};
use crate::identity::domain_digest;
use crate::identity::{ArtifactNodeId, DependencyEdge, Digest};
use crate::law_candidates::LawDerivationError;
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
use crate::{allocation, artifact, facts, mesh, normalize, search, select};

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
        request.numeric,
    )
    .map_err(|error| {
        let (site, detail) = match error {
            LawDerivationError::Value(detail) => ("request.graph.nodes[].program", detail),
            LawDerivationError::Region(detail) => ("optimizer.registered_passes", detail),
        };
        failure(
            CompilerFailureKind::LawDerivationFailed,
            site,
            detail,
            "lower the law derivation budget, or repair the registered pass set the region laws cite",
        )
    })?;
    let mut certificate = search.certificate;
    let ranked = select::rank(
        search.candidates,
        &planning_facts,
        &dependencies,
        request.device,
        &request.objective,
        &mut certificate,
        request.required_schedule,
    );
    certificate.canonicalize();
    if let Some(required) = ranked.unreachable_schedule {
        return Err(failure(
            CompilerFailureKind::RequiredScheduleUnreachable,
            "request.required_schedule",
            format!(
                "no legal candidate plan exercises the required schedule family {}",
                required.code()
            ),
            "raise the candidate bound, state device facts that grant the family, or require a schedule the graph admits",
        ));
    }
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
        numeric: request.numeric,
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
