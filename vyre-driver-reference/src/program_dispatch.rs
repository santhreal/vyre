//! Reference-only semantic parity execution over compiler-admitted graph artifacts.

use std::collections::{BTreeMap, BTreeSet};

use vyre_driver::target_dialect::{EmittedDialectModule, TargetDialect};
use vyre_foundation::ir::{GraphValueId, ValueLifetime};
use vyre_megakernel::{
    ObjectiveMetric, SemanticExecutionError, SemanticExecutionOutput, SemanticExecutionRequest,
    SemanticExecutor, TargetCompileError, TargetCompiler, TargetProfile,
};
use vyre_reference::value::Value;

use crate::CPU_REF_BACKEND_ID;

pub(crate) const REFERENCE_TARGET_FORMAT: &str = "reference-graph";

const REFERENCE_DIALECT: TargetDialect = TargetDialect {
    backend_id: CPU_REF_BACKEND_ID,
    dialect: "reference graph",
    format: REFERENCE_TARGET_FORMAT,
    format_version: 1,
    generation: 1,
    max_workgroup_size: [1_024, 1, 1],
    max_invocations_per_workgroup: 1_024,
    max_dynamic_shared_bytes: 0,
    subgroup_size: 1,
    emit: emit_reference_module,
};

fn emit_reference_module(
    selected: &vyre_megakernel::SelectedLowering,
    _profile: &TargetProfile,
) -> Result<EmittedDialectModule, TargetCompileError> {
    let mut bytes = Vec::with_capacity(48 + selected.nodes.len() * 4);
    bytes.extend_from_slice(b"vyre-reference-graph-v1\0");
    bytes.extend_from_slice(&selected.artifact.0);
    bytes.extend_from_slice(&selected.group.0.to_le_bytes());
    bytes.extend_from_slice(&selected.stage.to_le_bytes());
    for node in &selected.nodes {
        bytes.extend_from_slice(&node.0.to_le_bytes());
    }
    Ok(EmittedDialectModule {
        entry_point: "reference_eval".to_string(),
        bytes,
    })
}

pub(crate) fn target_compiler_factory() -> Result<Box<dyn TargetCompiler>, vyre_driver::BackendError>
{
    REFERENCE_DIALECT.compiler()
}

/// Reference-only parity executor for validated logical graphs.
///
/// Neutral compilation and registered target attachment run before the graph is
/// interpreted. Artifact and payload identities therefore have the same
/// canonical provenance as device execution rather than test-only digests.
#[derive(Debug, Default, Clone, Copy)]
pub struct ReferenceSemanticExecutor;

impl SemanticExecutor for ReferenceSemanticExecutor {
    fn execute(
        &self,
        request: &SemanticExecutionRequest<'_>,
    ) -> Result<SemanticExecutionOutput, SemanticExecutionError> {
        if request.objective().primary() != ObjectiveMetric::Latency {
            return Err(SemanticExecutionError::InvalidRequest(format!(
                "reference parity execution ranks nothing, so it cannot honour a `{}` objective. Fix: state a latency objective for the reference oracle",
                request.objective().primary().name()
            )));
        }
        if request.budget().max_measurements != 0 {
            return Err(SemanticExecutionError::InvalidRequest(
                "reference parity execution cannot satisfy a device-measurement budget. Fix: use a zero-measurement policy for the reference oracle".to_string(),
            ));
        }
        let compile_request = request
            .compile_request()
            .validate()
            .map_err(SemanticExecutionError::Compile)?;
        let artifact =
            vyre_megakernel::compile(&compile_request).map_err(SemanticExecutionError::Compile)?;
        let registration = vyre_driver::backend_registration(CPU_REF_BACKEND_ID)
            .map_err(|error| SemanticExecutionError::Backend(error.to_string()))?;
        let compiler = registration.target_compiler().map_err(|error| {
            SemanticExecutionError::Target(TargetCompileError::Unsupported(error.to_string()))
        })?;
        let payload = compiler
            .compile(&artifact)
            .map_err(SemanticExecutionError::Target)?;

        let graph = request.logical().graph();
        let external_values = graph
            .values()
            .iter()
            .filter(|value| value.producer.is_none())
            .map(|value| value.id)
            .collect::<BTreeSet<_>>();
        let supplied_values = request.inputs().keys().copied().collect::<BTreeSet<_>>();
        if external_values != supplied_values {
            let missing = external_values.difference(&supplied_values).count();
            let undeclared = supplied_values.difference(&external_values).count();
            return Err(SemanticExecutionError::InvalidRequest(format!(
                "graph-value inputs disagree with the logical graph: {missing} missing and {undeclared} undeclared value(s). Fix: key every external graph value exactly once"
            )));
        }

        let mut values = request
            .inputs()
            .iter()
            .map(|(value, bytes)| (*value, bytes.to_vec()))
            .collect::<BTreeMap<_, _>>();
        for node in graph.nodes() {
            let mut inputs = Vec::with_capacity(node.inputs.len());
            for input in &node.inputs {
                let bytes = values.get(&input.value).ok_or_else(|| {
                    SemanticExecutionError::InvalidRequest(format!(
                        "graph node `{}` is missing value {}. Fix: preserve graph dependency order and bind every external input",
                        node.name, input.value.0
                    ))
                })?;
                inputs.push(Value::from(bytes.clone()));
            }
            let outputs = vyre_reference::reference_eval(&node.program, &inputs).map_err(|error| {
                SemanticExecutionError::Backend(format!(
                    "reference graph node `{}` failed: {error}. Fix: validate the node Program and graph-value ABI",
                    node.name
                ))
            })?;
            let written = vyre_megakernel::writable_graph_values(node);
            let mut output_values = Vec::with_capacity(written.len());
            for value in written {
                let replaces_retained = node.inputs.iter().any(|port| port.value == value);
                output_values.push((value, replaces_retained));
            }
            if outputs.len() != output_values.len() {
                return Err(SemanticExecutionError::Backend(format!(
                    "reference graph node `{}` returned {} output(s) for {} writable graph value(s). Fix: keep graph ports aligned with the reference output ABI",
                    node.name,
                    outputs.len(),
                    output_values.len()
                )));
            }
            for ((value, replaces_retained), output) in output_values.into_iter().zip(outputs) {
                let prior = values.insert(value, output.to_bytes());
                if prior.is_some() != replaces_retained {
                    return Err(SemanticExecutionError::Backend(format!(
                        "reference graph node `{}` violated graph value {} replacement semantics. Fix: only read-write inputs may replace retained state",
                        node.name, value.0
                    )));
                }
            }
        }

        let mut outputs = BTreeMap::new();
        for value in graph.values().iter().filter(|value| {
            matches!(value.contract.lifetime, ValueLifetime::Output) && value.producer.is_some()
                || matches!(value.contract.lifetime, ValueLifetime::Retained)
        }) {
            let bytes = values.remove(&value.id).ok_or_else(|| {
                SemanticExecutionError::Backend(format!(
                    "reference graph omitted declared result value {}. Fix: execute every logical graph node before collecting outputs",
                    value.id.0
                ))
            })?;
            outputs.insert(GraphValueId(value.id.0), bytes);
        }

        Ok(SemanticExecutionOutput {
            artifact: artifact.digest(),
            payload: payload.digest(),
            outputs,
        })
    }
}
