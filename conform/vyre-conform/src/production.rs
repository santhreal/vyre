//! Production semantic compilation, artifact admission, and submission route.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use thiserror::Error;
use vyre_driver::{BackendRegistration, BindingPlan};
use vyre_foundation::ir::{BufferDecl, GraphValueId, Program, ProgramGraph};
use vyre_foundation::logical::LogicalProgramGraph;
use vyre_megakernel::{
    CompileObjective, Digest, ExternalFacts, SearchBudget, SemanticExecutionError,
    SemanticExecutionOutput, SemanticExecutionPolicy, SemanticExecutionRequest, SemanticExecutor,
};
use vyre_runtime::RegisteredSemanticExecutor;

const MAX_ARTIFACT_BYTES: u64 = 64 * 1024 * 1024;
const CONFORMANCE_SEARCH_BUDGET: SearchBudget =
    SearchBudget::new(256, 100_000, 1, 1, 1_000_000_000);

/// Ceiling on one bounded step against a registered backend.
pub const PRODUCTION_STEP_DEADLINE: Duration = Duration::from_secs(120);

/// Operation identity a bounded step reports for a program carrying none.
const UNNAMED_OP_ID: &str = "<program with no entry op id>";

/// Failure in the production conformance route.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ProductionError {
    /// Program-to-graph adaptation or compiler validation failed.
    #[error("production conformance compilation failed: {0}")]
    Compile(String),
    /// Semantic compilation, admission, submission, or output projection failed.
    #[error(transparent)]
    Semantic(#[from] SemanticExecutionError),
    /// Backend acquisition failed before compiler target facts could be recorded.
    #[error("backend dispatch route failed: {0}")]
    Dispatch(String),
    /// A bounded step did not finish inside [`PRODUCTION_STEP_DEADLINE`].
    #[error(
        "{step} of `{op_id}` on backend `{backend}` did not finish within {deadline:?}. \
         Fix: the step is bounded on purpose; diagnose the backend call that does not return \
         for this operation instead of waiting on it."
    )]
    Deadline {
        /// Which bounded step exceeded its ceiling.
        step: &'static str,
        /// Operation the step was running.
        op_id: String,
        /// Backend the step was running on.
        backend: &'static str,
        /// Ceiling the step exceeded.
        deadline: Duration,
    },
    /// A bounded step panicked instead of returning a result.
    #[error(
        "{step} of `{op_id}` on backend `{backend}` panicked. \
         Fix: repair the panicking backend path; a conformance step must return a typed error."
    )]
    Panicked {
        /// Which bounded step panicked.
        step: &'static str,
        /// Operation the step was running.
        op_id: String,
        /// Backend the bounded step was running on.
        backend: &'static str,
    },
    /// A bounded step was abandoned on expiry and the executor refuses more work.
    #[error(
        "`{op_id}` on backend `{backend}` abandoned a step that exceeded its deadline. \
         Fix: the abandoned step may still be inside a driver call against this \
         artifact; construct a fresh executor instead of reusing this one."
    )]
    Abandoned {
        /// Operation whose step was abandoned.
        op_id: String,
        /// Backend the abandoned step was running on.
        backend: &'static str,
    },
}

/// Run one backend step under `deadline`, naming `op_id` and `backend` on expiry.
///
/// The step runs on its own thread. A call already blocked inside a device driver
/// cannot be cancelled from outside it, so an expired step is abandoned rather
/// than joined.
pub fn run_bounded_step<T: Send + 'static>(
    step: &'static str,
    op_id: &str,
    backend: &'static str,
    deadline: Duration,
    work: impl FnOnce() -> Result<T, ProductionError> + Send + 'static,
) -> Result<T, ProductionError> {
    let (sender, receiver) = mpsc::sync_channel(1);
    let thread_name = format!("vyre-conform-{step}-{backend}");
    thread::Builder::new()
        .name(thread_name)
        .spawn(move || {
            let _ = sender.send(work());
        })
        .map_err(|error| {
            ProductionError::Dispatch(format!(
                "could not start a bounded {step} thread for `{op_id}` on `{backend}`: {error}. \
                 Fix: raise the process thread limit before running conformance."
            ))
        })?;
    match receiver.recv_timeout(deadline) {
        Ok(result) => result,
        Err(RecvTimeoutError::Timeout) => Err(ProductionError::Deadline {
            step,
            op_id: op_id.to_string(),
            backend,
            deadline,
        }),
        Err(RecvTimeoutError::Disconnected) => Err(ProductionError::Panicked {
            step,
            op_id: op_id.to_string(),
            backend,
        }),
    }
}

/// Compiler-selected artifact and payload identities with canonical program outputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProductionExecution {
    /// Neutral admitted artifact identity.
    pub artifact: Digest,
    /// Authenticated target payload identity submitted to the device.
    pub payload: Digest,
    /// Writable program buffers in declaration order.
    pub outputs: Vec<Vec<u8>>,
}

/// Reusable semantic execution boundary for one conformance program.
///
/// Every submission crosses [`SemanticExecutor`]. The executor compiles the
/// schedule-free program, admits its target payload, submits only its frozen
/// entry geometry, and returns the identities of the admitted bytes.
pub struct ProductionSession {
    executor: Arc<dyn SemanticExecutor>,
    policy: SemanticExecutionPolicy,
    program: Arc<Program>,
    op_id: String,
    backend: &'static str,
    abandoned: AtomicBool,
}

impl ProductionSession {
    /// Construct a semantic executor and explicit policy from registered target facts.
    pub fn from_registration(
        program: &Program,
        registration: &'static BackendRegistration,
    ) -> Result<Self, ProductionError> {
        let policy = semantic_policy_for_registration(registration)?;
        Ok(Self::with_executor(
            program,
            Arc::new(RegisteredSemanticExecutor::new(registration)),
            policy,
            registration.id,
        ))
    }

    /// Bind a caller-supplied semantic executor and explicit compiler policy.
    #[must_use]
    pub fn with_executor(
        program: &Program,
        executor: Arc<dyn SemanticExecutor>,
        policy: SemanticExecutionPolicy,
        backend: &'static str,
    ) -> Self {
        Self {
            executor,
            policy,
            program: Arc::new(program.clone()),
            op_id: program.entry_op_id().unwrap_or(UNNAMED_OP_ID).to_string(),
            backend,
            abandoned: AtomicBool::new(false),
        }
    }

    /// Execute caller inputs through semantic compilation and admitted artifact submission.
    pub fn submit(&self, inputs: &[&[u8]]) -> Result<ProductionExecution, ProductionError> {
        if self.abandoned.load(Ordering::Acquire) {
            return Err(ProductionError::Abandoned {
                op_id: self.op_id.clone(),
                backend: self.backend,
            });
        }
        let owned_inputs = inputs
            .iter()
            .map(|bytes| bytes.to_vec())
            .collect::<Vec<_>>();
        let executor = Arc::clone(&self.executor);
        let policy = self.policy.clone();
        let program = Arc::clone(&self.program);
        let outcome = run_bounded_step(
            "semantic execution",
            &self.op_id,
            self.backend,
            PRODUCTION_STEP_DEADLINE,
            move || execute_program(executor.as_ref(), &policy, &program, &owned_inputs),
        );
        if matches!(outcome, Err(ProductionError::Deadline { .. })) {
            self.abandoned.store(true, Ordering::Release);
        }
        outcome
    }

    /// What a passing case on this route proves, in the words a report records.
    #[must_use]
    pub const fn proof(&self) -> &'static str {
        "through canonical semantic artifact submission"
    }
}

/// Construct the explicit compiler policy from immutable registered target facts.
pub fn semantic_policy_for_registration(
    registration: &'static BackendRegistration,
) -> Result<SemanticExecutionPolicy, ProductionError> {
    let target_facts = registration
        .acquire()
        .map_err(|error| ProductionError::Dispatch(error.to_string()))?
        .device_profile()
        .compile_facts();
    Ok(SemanticExecutionPolicy::new(
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        target_facts,
        CompileObjective::MinimizeLatency,
        CONFORMANCE_SEARCH_BUDGET,
        MAX_ARTIFACT_BYTES,
    ))
}

fn execute_program(
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    program: &Program,
    inputs: &[Vec<u8>],
) -> Result<ProductionExecution, ProductionError> {
    let borrowed_inputs = inputs.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let binding_plan = BindingPlan::from_borrowed_inputs(program, &borrowed_inputs)
        .map_err(|error| ProductionError::Compile(error.to_string()))?;
    let mut runtime_counts = BTreeMap::new();
    for (&buffer_idx, bytes) in binding_plan.input_indices.iter().zip(inputs) {
        let buffer = &program.buffers()[buffer_idx];
        if buffer.count() == 0 {
            runtime_counts.insert(
                buffer.name().to_string(),
                runtime_element_count(buffer, bytes)?,
            );
        }
    }
    let graph =
        ProgramGraph::from_program_with_runtime_counts("main", program.clone(), &runtime_counts)
            .map_err(|error| ProductionError::Compile(error.to_string()))?;
    let logical = LogicalProgramGraph::validate(&graph, &policy.external_facts().symbolic_bindings)
        .map_err(|error| {
            ProductionError::Semantic(SemanticExecutionError::InvalidRequest(format!(
                "logical graph validation failed: {error}. Fix: supply exact dynamic extent bindings"
            )))
        })?;
    let node = graph.nodes().first().ok_or_else(|| {
        ProductionError::Semantic(SemanticExecutionError::InvalidRequest(
            "single-program graph has no node. Fix: supply one executable program".to_string(),
        ))
    })?;
    if node.inputs.len() != inputs.len() {
        return Err(ProductionError::Semantic(
            SemanticExecutionError::InvalidRequest(format!(
                "graph requires {} input value(s), received {}. Fix: supply one byte buffer per canonical graph input",
                node.inputs.len(),
                inputs.len()
            )),
        ));
    }
    let output_order = node.outputs.clone();
    let request_inputs = node
        .inputs
        .iter()
        .zip(inputs)
        .map(|(port, bytes)| (port.value, bytes.as_slice()))
        .collect::<BTreeMap<GraphValueId, &[u8]>>();
    let request = SemanticExecutionRequest::new(
        &logical,
        request_inputs,
        policy.external_facts().clone(),
        policy.target_facts(),
        policy.objective(),
        policy.budget(),
        policy.max_artifact_bytes(),
    )?;
    let SemanticExecutionOutput {
        artifact,
        payload,
        mut outputs,
    } = executor.execute(&request)?;
    let mut ordered = Vec::with_capacity(output_order.len());
    for value in output_order {
        let bytes = outputs.remove(&value).ok_or_else(|| {
            ProductionError::Semantic(SemanticExecutionError::Backend(format!(
                "executor omitted canonical output value {}. Fix: return every graph output exactly once",
                value.0
            )))
        })?;
        ordered.push(bytes);
    }
    if !outputs.is_empty() {
        return Err(ProductionError::Semantic(SemanticExecutionError::Backend(
            format!(
                "executor returned {} undeclared output value(s). Fix: return only canonical graph outputs",
                outputs.len()
            ),
        )));
    }
    Ok(ProductionExecution {
        artifact,
        payload,
        outputs: ordered,
    })
}

fn runtime_element_count(buffer: &BufferDecl, bytes: &[u8]) -> Result<u64, ProductionError> {
    let element_bits = if let Some(element_bytes) = buffer.element().size_bytes() {
        element_bytes.checked_mul(8).ok_or_else(|| {
            ProductionError::Compile(format!(
                "runtime-sized input `{}` element width overflowed host addressing",
                buffer.name()
            ))
        })?
    } else {
        return Err(ProductionError::Compile(format!(
            "runtime-sized input `{}` has variable-width element type `{}`; supply a fixed-width element contract before artifact compilation",
            buffer.name(),
            buffer.element()
        )));
    };
    if element_bits == 0 {
        return Err(ProductionError::Compile(format!(
            "runtime-sized input `{}` has zero-width element type `{}`",
            buffer.name(),
            buffer.element()
        )));
    }
    let total_bits = bytes.len().checked_mul(8).ok_or_else(|| {
        ProductionError::Compile(format!(
            "runtime-sized input `{}` byte length overflowed bit-count arithmetic",
            buffer.name()
        ))
    })?;
    if total_bits % element_bits != 0 {
        return Err(ProductionError::Compile(format!(
            "runtime-sized input `{}` has {} bytes, which is not an exact number of `{}` elements",
            buffer.name(),
            bytes.len(),
            buffer.element()
        )));
    }
    u64::try_from(total_bits / element_bits).map_err(|_| {
        ProductionError::Compile(format!(
            "runtime-sized input `{}` element count exceeds u64",
            buffer.name()
        ))
    })
}

/// Retained failure reproduction capsule that replays through the shipped product path.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ReplayCapsule {
    /// Wire-encoded Program bytes.
    pub wire_bytes: Vec<u8>,
    /// Workload host input byte buffers.
    pub inputs: Vec<Vec<u8>>,
    /// Expected diagnostic code or error identity.
    pub expected_diagnostic_code: Option<String>,
    /// Optional expected output digest when verifying divergent values.
    pub expected_output_digest: Option<u64>,
}

impl ReplayCapsule {
    /// Construct a capsule from a Program and inputs.
    pub fn from_program(
        program: &Program,
        inputs: &[&[u8]],
        expected_diagnostic_code: Option<String>,
    ) -> Result<Self, String> {
        let wire_bytes = program.to_wire().map_err(|e| e.to_string())?;
        Ok(Self {
            wire_bytes,
            inputs: inputs.iter().map(|b| b.to_vec()).collect(),
            expected_diagnostic_code,
            expected_output_digest: None,
        })
    }

    /// Replay this capsule through wire decode, host rewrites, semantic compilation,
    /// artifact admission, and device submission.
    pub fn replay_shipped_path(
        &self,
        registration: &'static BackendRegistration,
    ) -> Result<ProductionExecution, ProductionError> {
        let decoded_program = Program::from_wire(&self.wire_bytes)
            .map_err(|err| ProductionError::Compile(format!("wire decode failed: {err}")))?;
        let mut optimized = decoded_program;
        for rewrite in vyre_foundation::transform::HOST_REWRITES {
            optimized = (rewrite.apply)(&optimized);
        }
        let input_slices = self.inputs.iter().map(Vec::as_slice).collect::<Vec<_>>();
        ProductionSession::from_registration(&optimized, registration)?.submit(&input_slices)
    }
}

/// Find a live semantic-execution-capable non-oracle backend registration.
#[cfg(feature = "device-tests")]
#[doc(hidden)]
pub fn live_test_backend() -> Result<&'static BackendRegistration, String> {
    let selected = std::env::var("VYRE_BACKEND")
        .ok()
        .filter(|value| !value.trim().is_empty());
    vyre_registry_link::backend::live_backend_registry()
        .map_err(|error| format!("backend registry startup failed: {error}"))?
        .iter()
        .find(|registration| {
            !registration.reference_oracle
                && registration.target_compiler.is_some()
                && registration.materializer.is_some()
                && selected
                    .as_deref()
                    .is_none_or(|backend| registration.id == backend)
        })
        .ok_or_else(|| {
            "Fix: a semantic-execution-capable backend must be registered. Link a concrete driver crate into the test binary.".to_string()
        })
}
