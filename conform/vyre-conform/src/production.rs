//! Production compiler, target payload, materialization, and submission route.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use thiserror::Error;
use vyre_driver::{BackendRegistration, BindingPlan, DispatchConfig, VyreBackend};
use vyre_foundation::ir::{BufferDecl, Program, ProgramGraph};
use vyre_megakernel::{CompileRequest, Digest, ExternalFacts, SearchBudget};
use vyre_runtime::artifact_admission::{ArtifactSession, ArtifactSessionError};

const MAX_ARTIFACT_BYTES: u64 = 64 * 1024 * 1024;
const CONFORMANCE_SEARCH_BUDGET: SearchBudget =
    SearchBudget::new(256, 100_000, 1, 1, 1_000_000_000);

/// Ceiling on one bounded step against a registered backend.
///
/// Compilation, materialization and submission all end in a device driver call,
/// and a driver call carries no timeout of its own: a kernel that never retires,
/// a driver object freed under a live handle, or a device that fell off the bus
/// blocks the calling thread with no error and no diagnostic. A conformance run
/// that inherits that block reports nothing at all, including nothing about the
/// operations it never reached, so every step below is measured against this
/// ceiling and reports the operation and the backend that exceeded it.
///
/// Two minutes is orders of magnitude above the slowest conformance dispatch in
/// this workspace, which compiles one small program and launches it once.
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
    /// Artifact materialization or submission failed.
    #[error(transparent)]
    Runtime(#[from] ArtifactSessionError),
    /// Backend acquisition or dispatch failed on the non-artifact route.
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
        /// Backend the step was running on.
        backend: &'static str,
    },
    /// A bounded step was abandoned on expiry and the session refuses more work.
    #[error(
        "`{op_id}` on backend `{backend}` abandoned a step that exceeded its deadline. \
         Fix: the abandoned step may still be inside a driver call against this \
         artifact; compile a fresh session instead of reusing this one."
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
/// than joined: the caller receives [`ProductionError::Deadline`] and continues,
/// and the abandoned thread holds only the resources of the step that failed.
///
/// # Errors
///
/// Returns [`ProductionError::Deadline`] when `deadline` elapses first,
/// [`ProductionError::Panicked`] when the step panics, and whatever `work`
/// returns otherwise.
pub fn run_bounded_step<T: Send + 'static>(
    step: &'static str,
    op_id: &str,
    backend: &'static str,
    deadline: Duration,
    work: impl FnOnce() -> Result<T, ProductionError> + Send + 'static,
) -> Result<T, ProductionError> {
    let (sender, receiver) = mpsc::channel();
    thread::Builder::new()
        .name(format!("vyre-conform {step}"))
        .spawn(move || drop(sender.send(work())))
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

/// Materialized production artifact used for repeated conformance submissions.
pub struct ProductionSession {
    neutral: Digest,
    payload: Digest,
    /// Shared so a bounded submission can own a reference while the caller keeps
    /// this handle.
    session: Arc<ArtifactSession>,
    /// Shared for the same reason: the bounded submission projects its outputs in
    /// this Program's buffer declaration order, not in canonical ABI slot order.
    program: Arc<Program>,
    op_id: String,
    backend: &'static str,
    /// Set once a bounded step was abandoned on expiry.
    ///
    /// The abandoned thread still holds this session and may still be inside a
    /// driver call against the same materialized artifact and the same bindings,
    /// so a second submission would run concurrently with work the caller can
    /// neither see nor stop. The session refuses instead.
    abandoned: AtomicBool,
}

impl ProductionSession {
    /// Compile a program with no host inputs.
    ///
    /// # Errors
    ///
    /// Returns [`ProductionError`] when the program needs representative host
    /// inputs for measured compilation, or compilation and materialization fail.
    pub fn compile(
        program: &Program,
        registration: &'static BackendRegistration,
    ) -> Result<Self, ProductionError> {
        Self::compile_with_representative_inputs(program, &[], registration)
    }

    /// Compile, target-compile, authenticate, and materialize one frontend program.
    ///
    /// `representative_inputs` uses Program host-input order and supplies the
    /// exact workload bytes used for on-device finalist measurement.
    ///
    /// # Errors
    ///
    /// Returns [`ProductionError`] when input planning, compilation,
    /// materialization, or the bounded compile step fails.
    pub fn compile_with_representative_inputs(
        program: &Program,
        representative_inputs: &[&[u8]],
        registration: &'static BackendRegistration,
    ) -> Result<Self, ProductionError> {
        let op_id = program.entry_op_id().unwrap_or(UNNAMED_OP_ID).to_string();
        let backend = registration.id;
        let owned_program = Arc::new(program.clone());
        let compiled_program = Arc::clone(&owned_program);
        let owned_inputs = representative_inputs
            .iter()
            .map(|bytes| bytes.to_vec())
            .collect::<Vec<_>>();
        let session = run_bounded_step(
            "compilation",
            &op_id,
            backend,
            PRODUCTION_STEP_DEADLINE,
            move || compile_artifact_session(&compiled_program, owned_inputs, registration),
        )?;
        let neutral = session.artifact()?;
        let payload = session.payload()?;
        Ok(Self {
            neutral,
            payload,
            session: Arc::new(session),
            program: owned_program,
            op_id,
            backend,
            abandoned: AtomicBool::new(false),
        })
    }

    /// Immutable neutral artifact identity used by live and packaged routes.
    #[must_use]
    pub const fn artifact_digest(&self) -> Digest {
        self.neutral
    }

    /// Authenticated target payload identity materialized by this session.
    #[must_use]
    pub const fn payload_digest(&self) -> Digest {
        self.payload
    }

    /// Submit caller inputs and return writable buffers in binding order.
    ///
    /// # Errors
    ///
    /// Returns [`ProductionError`] when the backend rejects the inputs or the
    /// bounded submission exceeds [`PRODUCTION_STEP_DEADLINE`].
    pub fn submit(&self, inputs: &[&[u8]]) -> Result<Vec<Vec<u8>>, ProductionError> {
        self.submit_bounded("submission", inputs, None)
    }

    /// Submit caller inputs with a typed invocation-grid override.
    ///
    /// # Errors
    ///
    /// Returns [`ProductionError`] when the grid is rejected, the backend rejects
    /// the inputs, or the bounded submission exceeds
    /// [`PRODUCTION_STEP_DEADLINE`].
    pub fn submit_with_invocation_grid(
        &self,
        inputs: &[&[u8]],
        grid: [u32; 3],
    ) -> Result<Vec<Vec<u8>>, ProductionError> {
        self.submit_bounded("grid-pinned submission", inputs, Some(grid))
    }

    fn submit_bounded(
        &self,
        step: &'static str,
        inputs: &[&[u8]],
        grid: Option<[u32; 3]>,
    ) -> Result<Vec<Vec<u8>>, ProductionError> {
        if self.abandoned.load(Ordering::Acquire) {
            return Err(ProductionError::Abandoned {
                op_id: self.op_id.clone(),
                backend: self.backend,
            });
        }
        // The bounded step outlives this call when it exceeds its deadline, so it
        // cannot borrow the caller's input slices.
        let owned = inputs
            .iter()
            .map(|bytes| bytes.to_vec())
            .collect::<Vec<_>>();
        let session = Arc::clone(&self.session);
        let program = Arc::clone(&self.program);
        let outcome = run_bounded_step(
            step,
            &self.op_id,
            self.backend,
            PRODUCTION_STEP_DEADLINE,
            move || {
                let borrowed = owned.iter().map(Vec::as_slice).collect::<Vec<_>>();
                let completion = match grid {
                    Some(grid) => {
                        let mut bindings = session.host_bindings(&borrowed)?;
                        bindings
                            .set_invocation_grid(grid)
                            .map_err(ArtifactSessionError::from)?;
                        session.submit_and_wait(bindings)?
                    }
                    None => session.submit_host_inputs(&borrowed)?,
                };
                Ok(session.program_outputs(&program, &completion)?)
            },
        );
        if matches!(outcome, Err(ProductionError::Deadline { .. })) {
            self.abandoned.store(true, Ordering::Release);
        }
        outcome
    }
}

/// Compile and materialize one program on `registration`, without a bound.
///
/// [`ProductionSession::compile_with_representative_inputs`] runs this under
/// [`PRODUCTION_STEP_DEADLINE`].
fn compile_artifact_session(
    program: &Program,
    representative_inputs: Vec<Vec<u8>>,
    registration: &'static BackendRegistration,
) -> Result<ArtifactSession, ProductionError> {
    let borrowed_inputs = representative_inputs
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    let binding_plan = BindingPlan::from_borrowed_inputs(program, &borrowed_inputs)
        .map_err(|error| ProductionError::Compile(error.to_string()))?;
    let mut runtime_counts = BTreeMap::new();
    for (&buffer_idx, bytes) in binding_plan
        .input_indices
        .iter()
        .zip(&representative_inputs)
    {
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
    let mut representative_map = BTreeMap::new();
    for (bytes, &buffer_idx) in representative_inputs
        .into_iter()
        .zip(&binding_plan.input_indices)
    {
        let buffer = &program.buffers()[buffer_idx];
        let graph_value = graph
            .values()
            .iter()
            .find(|value| value.name == buffer.name())
            .ok_or_else(|| {
                ProductionError::Compile(format!(
                    "graph value for input buffer `{}` not found",
                    buffer.name()
                ))
            })?;
        representative_map.insert(graph_value.id, bytes);
    }
    let device = registration
        .acquire()
        .map_err(|error| ProductionError::Dispatch(error.to_string()))?
        .device_profile()
        .compile_facts();
    let facts = ExternalFacts::new(Digest([0; 32]), BTreeMap::new());
    let request = CompileRequest::new(
        graph,
        facts,
        device,
        CONFORMANCE_SEARCH_BUDGET,
        MAX_ARTIFACT_BYTES,
    )
    .with_representative_inputs(representative_map)
    .validate()
    .map_err(|error| ProductionError::Compile(error.to_string()))?;
    Ok(ArtifactSession::compile(registration, &request)?)
}

fn runtime_element_count(buffer: &BufferDecl, bytes: &[u8]) -> Result<u64, ProductionError> {
    let element_bits = if let Some(bits) = buffer.element().bit_width() {
        bits
    } else if let Some(element_bytes) = buffer.element().size_bytes() {
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
    u64::try_from(total_bits / element_bits).map_err(|error| {
        ProductionError::Compile(format!(
            "runtime-sized input `{}` element count exceeds u64: {error}",
            buffer.name()
        ))
    })
}

/// How one program is executed on one backend.
///
/// A backend that registers a target compiler and a materializer is exercised
/// through the production artifact route, which is the path a caller's program
/// takes in a release. The reference oracle registers neither: it interprets a
/// neutral program and has no target payload to authenticate or materialize.
/// Demanding a facet a registration declares absent measures the registration
/// rather than the operation, so the route follows what the registration says it
/// has.
#[non_exhaustive]
pub enum ExecutionRoute {
    /// Compiled, authenticated and materialized target artifact.
    Artifact(ProductionSession),
    /// The backend's own dispatch entry point, for a backend with no artifact.
    #[non_exhaustive]
    Dispatch {
        /// Acquired backend, shared so a bounded dispatch can own a reference.
        backend: Arc<dyn VyreBackend>,
        /// Program dispatched on every submission.
        program: Program,
        /// Operation identity reported when a bounded dispatch exceeds its
        /// ceiling.
        op_id: String,
        /// Backend identity reported alongside it.
        backend_id: &'static str,
    },
}

impl ExecutionRoute {
    /// Open the route `registration` declares it supports for an input-free program.
    ///
    /// # Errors
    ///
    /// Returns [`ProductionError`] when an artifact route needs representative
    /// host inputs, or compilation, materialization, or backend acquisition fails.
    pub fn open(
        program: &Program,
        registration: &'static BackendRegistration,
    ) -> Result<Self, ProductionError> {
        Self::open_with_representative_inputs(program, &[], registration)
    }

    /// Open the route `registration` declares it supports for `program`.
    ///
    /// The artifact route needs both a target compiler and a materializer, so it
    /// is taken only when the registration declares both. A registration missing
    /// either has no artifact to submit, and the backend's own dispatch entry
    /// point is the route it does have. Representative inputs are used only by
    /// measured artifact compilation.
    ///
    /// # Errors
    ///
    /// Returns [`ProductionError`] when input planning, compilation,
    /// materialization, or backend acquisition fails.
    pub fn open_with_representative_inputs(
        program: &Program,
        representative_inputs: &[&[u8]],
        registration: &'static BackendRegistration,
    ) -> Result<Self, ProductionError> {
        if registration.target_compiler.is_some() && registration.materializer.is_some() {
            return ProductionSession::compile_with_representative_inputs(
                program,
                representative_inputs,
                registration,
            )
            .map(Self::Artifact);
        }
        let backend = registration
            .acquire()
            .map_err(|error| ProductionError::Dispatch(error.to_string()))?;
        Ok(Self::Dispatch {
            backend: Arc::from(backend),
            program: program.clone(),
            op_id: program.entry_op_id().unwrap_or(UNNAMED_OP_ID).to_string(),
            backend_id: registration.id,
        })
    }

    /// Submit caller inputs and return writable buffers in binding order.
    ///
    /// `config` carries the invocation grid the program needs. The artifact
    /// route ignores it: a materialized artifact was compiled with its grid
    /// already bound. The dispatch route needs it, because a neutral program
    /// dispatched under the default grid executes one invocation and leaves
    /// every other output element at zero.
    ///
    /// # Errors
    ///
    /// Returns [`ProductionError`] when the backend cannot execute the inputs, or
    /// when the bounded step exceeds [`PRODUCTION_STEP_DEADLINE`].
    pub fn submit(
        &self,
        inputs: &[&[u8]],
        config: &DispatchConfig,
    ) -> Result<Vec<Vec<u8>>, ProductionError> {
        match self {
            Self::Artifact(session) => session.submit(inputs),
            Self::Dispatch {
                backend,
                program,
                op_id,
                backend_id,
            } => {
                let backend = Arc::clone(backend);
                let program = program.clone();
                let config = config.clone();
                let owned = inputs
                    .iter()
                    .map(|bytes| bytes.to_vec())
                    .collect::<Vec<_>>();
                run_bounded_step(
                    "dispatch",
                    op_id,
                    backend_id,
                    PRODUCTION_STEP_DEADLINE,
                    move || {
                        let borrowed = owned.iter().map(Vec::as_slice).collect::<Vec<_>>();
                        backend
                            .dispatch_borrowed(&program, &borrowed, &config)
                            .map_err(|error| ProductionError::Dispatch(error.to_string()))
                    },
                )
            }
        }
    }

    /// What a passing case on this route proves, in the words a report records.
    #[must_use]
    pub const fn proof(&self) -> &'static str {
        match self {
            Self::Artifact(_) => "through canonical artifact submission",
            Self::Dispatch { .. } => "through backend dispatch of the neutral program",
        }
    }
}
