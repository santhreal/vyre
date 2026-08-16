//! The compile seam: rank the legal candidates, assemble the winner, and the
//! measured path that times finalists on the live device.

use std::time::Instant;

use crate::envelope::TargetPayload;
use crate::error::{failure, overflow, serialization_failure, CompileError, CompilerFailureKind};
use crate::identity::domain_digest;
use crate::identity::{ArtifactNodeId, DependencyEdge, Digest};
use crate::request::{SearchWork, ValidatedCompileRequest};
use crate::request_identity::{RequestIdentity, REQUEST_DIGEST_DOMAIN, SOURCE_DIGEST_DOMAIN};
use crate::resource_records::{build_abi, build_resources};
use crate::schema::encode_payload;
use crate::schema::{
    Artifact, ArtifactPayload, FusionRejection, NodeRecord, PlanMeasurement, Provenance,
    ARTIFACT_SCHEMA_VERSION,
};
use crate::target::{TargetCompileError, TargetCompiler};
use crate::{artifact, facts, normalize, search, select};

/// Everything one compilation derives once and every finalist reuses.
struct CompileContext {
    source_graph: Digest,
    nodes: Vec<NodeRecord>,
    dependencies: Vec<DependencyEdge>,
    facts: facts::PlanningFacts,
    ranked: Vec<select::Selection>,
    pruned_fusions: Vec<FusionRejection>,
    work: SearchWork,
}

/// Rank every legal candidate for one validated request.
fn prepare(request: &ValidatedCompileRequest) -> Result<CompileContext, CompileError> {
    let canonical_wire = request.graph.to_wire().map_err(|error| {
        failure(
            CompilerFailureKind::InvalidProgram,
            "request.graph",
            error.to_string(),
            "supply a graph representable by the canonical foundation wire format",
        )
    })?;
    let source_graph = domain_digest(SOURCE_DIGEST_DOMAIN, &canonical_wire);
    let nodes = request
        .graph
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
    let normalized = normalize::normalize(&request.graph)?;
    let dependencies = normalized.dependencies;
    let planning_facts = facts::derive(
        &request.graph,
        &dependencies,
        &request.facts.symbolic_bindings,
    )?;
    let search = search::explore(
        &request.graph,
        &planning_facts,
        &dependencies,
        request.search_budget,
        request.device,
    );
    let ranked = select::rank(
        search.candidates,
        &planning_facts,
        &dependencies,
        request.device,
    );
    if ranked.is_empty() {
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
        source_graph,
        nodes,
        dependencies,
        facts: planning_facts,
        ranked,
        pruned_fusions,
        work: search.work,
    })
}

/// Turn one ranked candidate into a complete canonical artifact.
fn assemble(
    request: &ValidatedCompileRequest,
    context: &CompileContext,
    selection: &select::Selection,
    work: SearchWork,
    measurement: PlanMeasurement,
) -> Result<Artifact, CompileError> {
    let artifact::ArtifactPlan {
        node_groups,
        stages,
        geometry,
        selected_plan,
    } = artifact::plan(artifact::PlanInputs {
        graph: &request.graph,
        dependencies: &context.dependencies,
        facts: &context.facts,
        selection,
        pruned_fusions: &context.pruned_fusions,
        external: &request.facts,
        device: request.device,
        budget: request.search_budget,
        work,
        measurement,
    })?;
    let (resources, resource_envelope) = build_resources(
        &request.graph,
        &request.facts.symbolic_bindings,
        &node_groups,
        &stages,
    )?;
    let abi = build_abi(&request.graph)?;
    let request_bytes =
        serde_json::to_vec(&RequestIdentity::from(request)).map_err(serialization_failure)?;
    let provenance = Provenance {
        source_graph: context.source_graph,
        request: domain_digest(REQUEST_DIGEST_DOMAIN, &request_bytes),
        compiler_version: env!("CARGO_PKG_VERSION").to_string(),
    };
    let payload = ArtifactPayload {
        schema_version: ARTIFACT_SCHEMA_VERSION,
        nodes: context.nodes.clone(),
        dependencies: context.dependencies.clone(),
        selected_plan,
        abi,
        resources,
        resource_envelope,
        geometry,
        provenance,
    };
    let framed = encode_payload(&payload)?;
    let byte_len = u64::try_from(framed.bytes.len())
        .map_err(|_| overflow("artifact", "artifact length exceeds u64"))?;
    if byte_len > request.max_artifact_bytes {
        return Err(failure(
            CompilerFailureKind::ArtifactLimit,
            "artifact",
            format!(
                "canonical artifact is {byte_len} bytes; limit is {}",
                request.max_artifact_bytes
            ),
            "raise the explicit artifact bound or reduce the source graph",
        ));
    }
    Ok(Artifact {
        payload,
        digest: Digest(framed.digest),
    })
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
    let context = prepare(request)?;
    let selection = first_ranked(&context)?;
    assemble(
        request,
        &context,
        selection,
        context.work,
        PlanMeasurement::Unbudgeted,
    )
}

/// Device access the compiler borrows to time its finalists.
///
/// The compiler owns which plans are finalists and how their times are compared.
/// The caller owns the device: it supplies the target compiler that turns one
/// artifact into loadable bytes, and a launch that returns the device time of
/// one execution. Nothing here acquires a device, so a caller without one calls
/// [`compile`] instead.
pub trait FinalistEvaluator {
    /// Target compiler that turns one candidate artifact into target bytes.
    fn target_compiler(&self) -> &dyn TargetCompiler;

    /// Launch `payload` once and return the device time of that launch in
    /// nanoseconds. The time must come from the device, not the host clock.
    fn measure(
        &self,
        artifact: &Artifact,
        payload: &TargetPayload,
    ) -> Result<u64, TargetCompileError>;
}

/// Compile with the ranked finalists compiled for the target and timed on the
/// device, selecting the plan with the lowest median device time.
///
/// The analytic ranking chooses which plans are worth a target compilation. The
/// top `max_target_compilations` of them are compiled and each launched
/// `max_measurements` times; the winner is the finalist with the lowest median.
/// [`SearchWork::target_compilations`] and [`SearchWork::measurements`] carry the
/// counts actually spent, and the recorded [`PlanMeasurement`] states whether a
/// measurement decided the plan at all: a zero measurement budget records
/// [`PlanMeasurement::Unbudgeted`] and a device with no launch timestamps records
/// [`PlanMeasurement::UntimedDevice`], neither of which is reported as a measured
/// selection.
pub fn compile_measured(
    request: &ValidatedCompileRequest,
    evaluator: &dyn FinalistEvaluator,
) -> Result<Artifact, CompileError> {
    let context = prepare(request)?;
    let budget = request.search_budget;
    if budget.max_measurements == 0 || budget.max_target_compilations == 0 {
        return assemble(
            request,
            &context,
            first_ranked(&context)?,
            context.work,
            PlanMeasurement::Unbudgeted,
        );
    }
    if !request.device.supports_device_timestamps() {
        return assemble(
            request,
            &context,
            first_ranked(&context)?,
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
    let mut winner: Option<(usize, u64, u32)> = None;
    for index in 0..finalists {
        let elapsed = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        if elapsed >= budget.max_elapsed_ns {
            break;
        }
        let provisional = assemble(
            request,
            &context,
            &context.ranked[index],
            context.work,
            PlanMeasurement::Unbudgeted,
        )?;
        let payload = evaluator
            .target_compiler()
            .compile(&provisional)
            .map_err(|error| finalist_failure(index, &error))?;
        work.target_compilations = work.target_compilations.saturating_add(1);
        let mut samples = Vec::with_capacity(budget.max_measurements as usize);
        for _ in 0..budget.max_measurements {
            let sample = evaluator
                .measure(&provisional, &payload)
                .map_err(|error| finalist_failure(index, &error))?;
            samples.push(sample);
            work.measurements = work.measurements.saturating_add(1);
        }
        samples.sort_unstable();
        let launches = u32::try_from(samples.len()).unwrap_or(u32::MAX);
        let Some(median) = samples.get(samples.len() / 2).copied() else {
            continue;
        };
        if winner.is_none_or(|(_, best, _)| median < best) {
            winner = Some((index, median, launches));
        }
    }
    work.elapsed_ns = work
        .elapsed_ns
        .saturating_add(u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX));
    match winner {
        Some((index, median_ns, launches)) => assemble(
            request,
            &context,
            &context.ranked[index],
            work,
            PlanMeasurement::Measured {
                launches,
                median_ns,
            },
        ),
        None => assemble(
            request,
            &context,
            first_ranked(&context)?,
            work,
            PlanMeasurement::Unbudgeted,
        ),
    }
}

fn first_ranked(context: &CompileContext) -> Result<&select::Selection, CompileError> {
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
