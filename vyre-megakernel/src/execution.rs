//! Semantic compile-and-execute boundary over validated logical graphs.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use vyre_foundation::ir::{BufferAccess, GraphValueId, Program, ProgramGraph};
use vyre_foundation::logical::LogicalProgramGraph;

use crate::error::CompileError;
use crate::request::CompileRequest;
use crate::target::TargetCompileError;
use crate::{DeviceFacts, Digest, ExternalFacts, SearchBudget};

/// Compiler objective applied while ranking legal schedules.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompileObjective {
    /// Minimize measured or modeled end-to-end device latency.
    MinimizeLatency,
}
/// Immutable non-geometric policy supplied to semantic compilation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticExecutionPolicy {
    external_facts: ExternalFacts,
    target_facts: DeviceFacts,
    objective: CompileObjective,
    budget: SearchBudget,
    max_artifact_bytes: u64,
}

impl SemanticExecutionPolicy {
    /// Construct an explicit semantic compilation policy.
    #[must_use]
    pub const fn new(
        external_facts: ExternalFacts,
        target_facts: DeviceFacts,
        objective: CompileObjective,
        budget: SearchBudget,
        max_artifact_bytes: u64,
    ) -> Self {
        Self {
            external_facts,
            target_facts,
            objective,
            budget,
            max_artifact_bytes,
        }
    }

    /// Borrow immutable semantic facts outside graph topology.
    #[must_use]
    pub const fn external_facts(&self) -> &ExternalFacts {
        &self.external_facts
    }

    /// Return the live target facts used for legality and ranking.
    #[must_use]
    pub const fn target_facts(&self) -> DeviceFacts {
        self.target_facts
    }

    /// Return the explicit ranking objective.
    #[must_use]
    pub const fn objective(&self) -> CompileObjective {
        self.objective
    }

    /// Return the bounded compiler search budget.
    #[must_use]
    pub const fn budget(&self) -> SearchBudget {
        self.budget
    }

    /// Return the maximum admitted neutral artifact size.
    #[must_use]
    pub const fn max_artifact_bytes(&self) -> u64 {
        self.max_artifact_bytes
    }
}

/// One semantic execution request with no caller-controlled launch geometry.
#[derive(Debug)]
pub struct SemanticExecutionRequest<'a> {
    logical: &'a LogicalProgramGraph<'a>,
    inputs: BTreeMap<GraphValueId, &'a [u8]>,
    external_facts: ExternalFacts,
    target_facts: DeviceFacts,
    objective: CompileObjective,
    budget: SearchBudget,
    max_artifact_bytes: u64,
}

impl<'a> SemanticExecutionRequest<'a> {
    /// Construct a request from a validated logical graph and explicit compiler policy.
    ///
    /// # Errors
    ///
    /// Rejects missing canonical graph inputs and values that are not external
    /// inputs of the validated graph.
    pub fn new(
        logical: &'a LogicalProgramGraph<'a>,
        inputs: BTreeMap<GraphValueId, &'a [u8]>,
        external_facts: ExternalFacts,
        target_facts: DeviceFacts,
        objective: CompileObjective,
        budget: SearchBudget,
        max_artifact_bytes: u64,
    ) -> Result<Self, SemanticExecutionError> {
        let expected = logical
            .graph()
            .nodes()
            .iter()
            .flat_map(|node| node.inputs.iter().map(|port| port.value))
            .filter(|value| {
                logical.graph().values()[value.0 as usize]
                    .producer
                    .is_none()
            })
            .collect::<BTreeSet<_>>();
        if let Some(value) = expected.iter().find(|value| !inputs.contains_key(value)) {
            return Err(SemanticExecutionError::InvalidRequest(format!(
                "request omitted canonical input graph value {}. Fix: bind every external graph input exactly once",
                value.0
            )));
        }
        if let Some(value) = inputs.keys().find(|value| !expected.contains(value)) {
            return Err(SemanticExecutionError::InvalidRequest(format!(
                "request supplied undeclared input graph value {}. Fix: bind only external graph inputs",
                value.0
            )));
        }
        Ok(Self {
            logical,
            inputs,
            external_facts,
            target_facts,
            objective,
            budget,
            max_artifact_bytes,
        })
    }

    /// Borrow the validated schedule-free graph.
    #[must_use]
    pub const fn logical(&self) -> &LogicalProgramGraph<'a> {
        self.logical
    }

    /// Borrow exact host inputs keyed by graph value identity.
    #[must_use]
    pub const fn inputs(&self) -> &BTreeMap<GraphValueId, &'a [u8]> {
        &self.inputs
    }

    /// Borrow immutable semantic facts outside graph topology.
    #[must_use]
    pub const fn external_facts(&self) -> &ExternalFacts {
        &self.external_facts
    }

    /// Return the live target facts used for legality and ranking.
    #[must_use]
    pub const fn target_facts(&self) -> DeviceFacts {
        self.target_facts
    }

    /// Return the explicit ranking objective.
    #[must_use]
    pub const fn objective(&self) -> CompileObjective {
        self.objective
    }

    /// Return the bounded compiler search budget.
    #[must_use]
    pub const fn budget(&self) -> SearchBudget {
        self.budget
    }

    /// Return the maximum admitted neutral artifact size.
    #[must_use]
    pub const fn max_artifact_bytes(&self) -> u64 {
        self.max_artifact_bytes
    }

    /// Project the semantic request into the canonical neutral compiler request.
    ///
    /// Representative inputs are copied only when the explicit measurement
    /// budget permits device measurement.
    #[must_use]
    pub fn compile_request(&self) -> CompileRequest {
        let mut request = CompileRequest::new(
            self.logical.graph().clone(),
            self.external_facts.clone(),
            self.target_facts,
            self.budget,
            self.max_artifact_bytes,
        )
        .with_objective(self.objective);
        if self.budget.max_measurements > 0 {
            let representative_inputs = self
                .inputs
                .iter()
                .map(|(value, bytes)| (*value, bytes.to_vec()))
                .collect();
            request = request.with_representative_inputs(representative_inputs);
        }
        request
    }
}

/// Typed result of compiling, admitting, and executing one semantic request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticExecutionOutput {
    /// Neutral artifact identity selected by the compiler.
    pub artifact: Digest,
    /// Authenticated target payload identity submitted to the device.
    pub payload: Digest,
    /// Canonical graph outputs keyed by graph value identity.
    pub outputs: BTreeMap<GraphValueId, Vec<u8>>,
}
/// Ordered outputs of a single-program semantic execution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SingleProgramExecutionOutput {
    /// Neutral artifact identity selected by the compiler.
    pub artifact: Digest,
    /// Authenticated target payload identity submitted to the device.
    pub payload: Digest,
    /// Program outputs in canonical declaration order.
    pub outputs: Vec<Vec<u8>>,
    /// Program buffer name of each entry in `outputs`, same order.
    pub output_buffers: Vec<String>,
}

impl SingleProgramExecutionOutput {
    /// Bytes written to one named Program buffer.
    ///
    /// A program that declares read-write working storage writes more buffers
    /// than the one a wrapper reads back, so a positional guess lands on
    /// whichever buffer the declaration order put first. Select by name.
    #[must_use]
    pub fn buffer(&self, name: &str) -> Option<&[u8]> {
        self.output_buffers
            .iter()
            .position(|candidate| candidate == name)
            .map(|index| self.outputs[index].as_slice())
    }
}

/// Failure at the semantic execution boundary.
#[derive(Debug, Error)]
pub enum SemanticExecutionError {
    /// Request inputs or immutable compiler policy are invalid.
    #[error("semantic execution request is invalid: {0}")]
    InvalidRequest(String),
    /// Neutral compilation or target attachment failed.
    #[error(transparent)]
    Compile(#[from] CompileError),
    /// Target payload compilation or attachment failed.
    #[error(transparent)]
    Target(#[from] TargetCompileError),
    /// Artifact admission, materialization, or submission failed.
    #[error("semantic artifact execution failed: {0}")]
    Backend(String),
}

/// Graph values one node writes, paired with the Program buffer that carries
/// them, in Program buffer declaration order.
///
/// A backend-allocated output buffer resolves through the node's output ports;
/// a read-write buffer resolves through the input port that carries its
/// retained value. Workgroup scratch and read-only buffers write nothing and
/// are absent.
///
/// This is the one answer to "which graph values does this node write, and in
/// what order". Every executor returns results in it, and a second derivation
/// is how a retained read-write value came to be dropped from one executor's
/// result list while the program still wrote it.
#[must_use]
pub fn writable_graph_value_buffers(
    node: &vyre_foundation::ir::ProgramGraphNode,
) -> Vec<(GraphValueId, String)> {
    let mut order = Vec::new();
    for buffer in node.program.buffers() {
        if let Some(value) = node
            .output_ports
            .iter()
            .zip(node.outputs.iter().copied())
            .find_map(|(port, value)| (port.buffer == buffer.name()).then_some(value))
        {
            order.push((value, buffer.name().to_string()));
            continue;
        }
        if let Some(value) = node.inputs.iter().find_map(|port| {
            (port.buffer == buffer.name() && port.contract.access == BufferAccess::ReadWrite)
                .then_some(port.value)
        }) {
            order.push((value, buffer.name().to_string()));
        }
    }
    order
}

/// Graph values one node writes, in Program buffer declaration order.
#[must_use]
pub fn writable_graph_values(node: &vyre_foundation::ir::ProgramGraphNode) -> Vec<GraphValueId> {
    writable_graph_value_buffers(node)
        .into_iter()
        .map(|(value, _)| value)
        .collect()
}

/// Validate one schedule-free program as a graph and execute it through the
/// canonical semantic boundary.
///
/// This helper maps canonical graph input ports to caller bytes and returns one
/// buffer per written Program buffer, in declaration order. It never accepts or
/// derives physical launch geometry; schedule search owns that work.
pub fn execute_single_program(
    executor: &dyn SemanticExecutor,
    node_name: &str,
    program: Program,
    inputs: &[Vec<u8>],
    policy: &SemanticExecutionPolicy,
) -> Result<SingleProgramExecutionOutput, SemanticExecutionError> {
    let graph = ProgramGraph::from_program(node_name, program).map_err(|error| {
        SemanticExecutionError::InvalidRequest(format!(
            "program graph validation failed: {error}. Fix: supply a bounded schedule-free program"
        ))
    })?;
    let logical = LogicalProgramGraph::validate(&graph, &policy.external_facts.symbolic_bindings)
        .map_err(|error| {
        SemanticExecutionError::InvalidRequest(format!(
            "logical graph validation failed: {error}. Fix: supply exact dynamic extent bindings"
        ))
    })?;
    let node = graph.nodes().first().ok_or_else(|| {
        SemanticExecutionError::InvalidRequest(
            "single-program graph has no node. Fix: supply one executable program".to_string(),
        )
    })?;
    if node.inputs.len() != inputs.len() {
        return Err(SemanticExecutionError::InvalidRequest(format!(
            "graph requires {} input value(s), received {}. Fix: supply one byte buffer per canonical graph input",
            node.inputs.len(),
            inputs.len()
        )));
    }
    let output_order = writable_graph_value_buffers(node);
    let inputs = node
        .inputs
        .iter()
        .zip(inputs)
        .map(|(port, bytes)| (port.value, bytes.as_slice()))
        .collect();
    let request = SemanticExecutionRequest::new(
        &logical,
        inputs,
        policy.external_facts.clone(),
        policy.target_facts,
        policy.objective,
        policy.budget,
        policy.max_artifact_bytes,
    )?;
    let output = executor.execute(&request)?;
    let SemanticExecutionOutput {
        artifact,
        payload,
        mut outputs,
    } = output;
    let mut ordered = Vec::with_capacity(output_order.len());
    let mut output_buffers = Vec::with_capacity(output_order.len());
    for (value, buffer) in output_order {
        let bytes = outputs.remove(&value).ok_or_else(|| {
            SemanticExecutionError::Backend(format!(
                "executor omitted canonical output value {}. Fix: return every graph output exactly once",
                value.0
            ))
        })?;
        ordered.push(bytes);
        output_buffers.push(buffer);
    }
    if !outputs.is_empty() {
        return Err(SemanticExecutionError::Backend(format!(
            "executor returned {} undeclared output value(s). Fix: return only canonical graph outputs",
            outputs.len()
        )));
    }
    Ok(SingleProgramExecutionOutput {
        artifact,
        payload,
        outputs: ordered,
        output_buffers,
    })
}

/// Compile and execute validated logical graphs through admitted artifacts.
///
/// Implementations may cache immutable artifacts or materialized instances, but
/// every cache key includes the request facts, objective, and budget. Submission
/// cannot accept a grid, workgroup, persistence, or route override.
pub trait SemanticExecutor: Send + Sync {
    /// Compile, admit, submit, and return canonical graph outputs.
    fn execute(
        &self,
        request: &SemanticExecutionRequest<'_>,
    ) -> Result<SemanticExecutionOutput, SemanticExecutionError>;
}
#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node};

    use super::*;

    const BUDGET: SearchBudget = SearchBudget::new(8, 64, 1, 0, 1_000);
    fn policy() -> SemanticExecutionPolicy {
        SemanticExecutionPolicy::new(
            ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
            DeviceFacts::unknown(),
            CompileObjective::MinimizeLatency,
            BUDGET,
            1_000_000,
        )
    }

    fn semantic_program() -> Program {
        Program::wrapped(
            vec![
                BufferDecl::read("src", 0, DataType::U32).with_count(1),
                BufferDecl::output("out", 1, DataType::U32).with_count(1),
            ],
            [1, 1, 1],
            vec![Node::store(
                "out",
                Expr::logical_index(0),
                Expr::load("src", Expr::logical_index(0)),
            )],
        )
    }

    struct RecordingExecutor {
        calls: AtomicUsize,
    }

    impl SemanticExecutor for RecordingExecutor {
        fn execute(
            &self,
            request: &SemanticExecutionRequest<'_>,
        ) -> Result<SemanticExecutionOutput, SemanticExecutionError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            assert_eq!(request.objective(), CompileObjective::MinimizeLatency);
            assert_eq!(request.budget(), BUDGET);
            assert_eq!(request.target_facts(), DeviceFacts::unknown());
            assert_eq!(request.max_artifact_bytes(), 1_000_000);
            let projected = request
                .compile_request()
                .validate()
                .expect("semantic request must project to one valid compiler request");
            assert_eq!(projected.objective(), CompileObjective::MinimizeLatency);
            let node = &request.logical().graph().nodes()[0];
            assert_eq!(request.inputs()[&node.inputs[0].value], [7, 0, 0, 0]);
            Ok(SemanticExecutionOutput {
                artifact: Digest([1; 32]),
                payload: Digest([2; 32]),
                outputs: BTreeMap::from([(node.outputs[0], vec![7, 0, 0, 0])]),
            })
        }
    }

    /// WHY: wrapper migration must preserve typed graph inputs while making
    /// physical launch geometry impossible to supply at this boundary.
    #[test]
    fn single_program_helper_crosses_the_validated_semantic_boundary() {
        let executor = RecordingExecutor {
            calls: AtomicUsize::new(0),
        };
        let output = execute_single_program(
            &executor,
            "copy",
            semantic_program(),
            &[vec![7, 0, 0, 0]],
            &policy(),
        )
        .expect("semantic execution must succeed");

        assert_eq!(executor.calls.load(Ordering::Relaxed), 1);
        assert_eq!(output.artifact, Digest([1; 32]));
        assert_eq!(output.payload, Digest([2; 32]));
        assert_eq!(output.outputs.len(), 1);
    }

    struct NeverExecutor;

    impl SemanticExecutor for NeverExecutor {
        fn execute(
            &self,
            _request: &SemanticExecutionRequest<'_>,
        ) -> Result<SemanticExecutionOutput, SemanticExecutionError> {
            panic!("invalid input arity reached the execution backend")
        }
    }

    #[test]
    fn single_program_helper_rejects_missing_canonical_inputs() {
        let error =
            execute_single_program(&NeverExecutor, "copy", semantic_program(), &[], &policy())
                .expect_err("missing input must fail before execution");

        assert!(error.to_string().contains("requires 1 input value"));
        assert!(error.to_string().contains("Fix:"));
    }
    #[derive(Clone, Copy)]
    enum OutputFault {
        Missing,
        Extra,
    }

    struct FaultyOutputExecutor(OutputFault);

    impl SemanticExecutor for FaultyOutputExecutor {
        fn execute(
            &self,
            request: &SemanticExecutionRequest<'_>,
        ) -> Result<SemanticExecutionOutput, SemanticExecutionError> {
            let output = request.logical().graph().nodes()[0].outputs[0];
            let outputs = match self.0 {
                OutputFault::Missing => BTreeMap::new(),
                OutputFault::Extra => BTreeMap::from([
                    (output, vec![7, 0, 0, 0]),
                    (GraphValueId(u32::MAX), vec![0; 4]),
                ]),
            };
            Ok(SemanticExecutionOutput {
                artifact: Digest([1; 32]),
                payload: Digest([2; 32]),
                outputs,
            })
        }
    }

    /// WHY: an executor must account for the complete graph-output set; an
    /// omitted or invented value cannot be hidden by ordered byte projection.
    #[test]
    fn single_program_helper_rejects_incomplete_and_extra_output_sets() {
        for (fault, expected) in [
            (OutputFault::Missing, "omitted canonical output value"),
            (OutputFault::Extra, "undeclared output value"),
        ] {
            let error = execute_single_program(
                &FaultyOutputExecutor(fault),
                "copy",
                semantic_program(),
                &[vec![7, 0, 0, 0]],
                &policy(),
            )
            .expect_err("invalid output set must fail");
            assert!(error.to_string().contains(expected), "{error}");
            assert!(error.to_string().contains("Fix:"), "{error}");
        }
    }
}
