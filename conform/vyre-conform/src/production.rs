//! Production compiler, target payload, materialization, and submission route.

use std::collections::BTreeMap;

use thiserror::Error;
use vyre_driver::{BackendRegistration, DispatchConfig, VyreBackend};
use vyre_foundation::ir::{Program, ProgramGraph};
use vyre_megakernel::{CompileRequest, Digest, ExternalFacts, SearchBudget};
use vyre_runtime::artifact_admission::{ArtifactSession, ArtifactSessionError};

const MAX_ARTIFACT_BYTES: u64 = 64 * 1024 * 1024;
const CONFORMANCE_SEARCH_BUDGET: SearchBudget =
    SearchBudget::new(256, 100_000, 1, 0, 1_000_000_000);

/// Failure in the production conformance route.
#[derive(Debug, Error)]
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
}

/// Materialized production artifact used for repeated conformance submissions.
pub struct ProductionSession {
    neutral: Digest,
    payload: Digest,
    session: ArtifactSession,
}

impl ProductionSession {
    /// Compile, target-compile, authenticate, and materialize one frontend program.
    pub fn compile(
        program: &Program,
        registration: &'static BackendRegistration,
    ) -> Result<Self, ProductionError> {
        let graph = ProgramGraph::from_program("main", program.clone())
            .map_err(|error| ProductionError::Compile(error.to_string()))?;
        let request = CompileRequest::new(
            graph,
            ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
            CONFORMANCE_SEARCH_BUDGET,
            MAX_ARTIFACT_BYTES,
        )
        .validate()
        .map_err(|error| ProductionError::Compile(error.to_string()))?;
        let session = ArtifactSession::compile(registration, &request)?;
        let neutral_digest = session.artifact()?;
        let payload_digest = session.payload()?;
        Ok(Self {
            neutral: neutral_digest,
            payload: payload_digest,
            session,
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
    pub fn submit(&self, inputs: &[&[u8]]) -> Result<Vec<Vec<u8>>, ProductionError> {
        let completion = self.session.submit_host_inputs(inputs)?;
        Ok(self.session.ordered_outputs(&completion)?)
    }

    /// Submit caller inputs with a typed invocation-grid override.
    pub fn submit_with_invocation_grid(
        &self,
        inputs: &[&[u8]],
        grid: [u32; 3],
    ) -> Result<Vec<Vec<u8>>, ProductionError> {
        let mut bindings = self.session.host_bindings(inputs)?;
        bindings
            .set_invocation_grid(grid)
            .map_err(ArtifactSessionError::from)?;
        let completion = self.session.submit_and_wait(bindings)?;
        Ok(self.session.ordered_outputs(&completion)?)
    }
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
pub enum ExecutionRoute {
    /// Compiled, authenticated and materialized target artifact.
    Artifact(ProductionSession),
    /// The backend's own dispatch entry point, for a backend with no artifact.
    Dispatch {
        /// Acquired backend.
        backend: Box<dyn VyreBackend>,
        /// Program dispatched on every submission.
        program: Program,
    },
}

impl ExecutionRoute {
    /// Open the route `registration` declares it supports for `program`.
    ///
    /// The artifact route needs both a target compiler and a materializer, so it
    /// is taken only when the registration declares both. A registration missing
    /// either has no artifact to submit, and the backend's own dispatch entry
    /// point is the route it does have.
    ///
    /// # Errors
    ///
    /// Returns [`ProductionError`] when compilation, materialization or backend
    /// acquisition fails.
    pub fn open(
        program: &Program,
        registration: &'static BackendRegistration,
    ) -> Result<Self, ProductionError> {
        if registration.target_compiler.is_some() && registration.materializer.is_some() {
            return ProductionSession::compile(program, registration).map(Self::Artifact);
        }
        let backend = registration
            .acquire()
            .map_err(|error| ProductionError::Dispatch(error.to_string()))?;
        Ok(Self::Dispatch {
            backend,
            program: program.clone(),
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
    /// Returns [`ProductionError`] when the backend cannot execute the inputs.
    pub fn submit(
        &self,
        inputs: &[&[u8]],
        config: &DispatchConfig,
    ) -> Result<Vec<Vec<u8>>, ProductionError> {
        match self {
            Self::Artifact(session) => session.submit(inputs),
            Self::Dispatch { backend, program } => backend
                .dispatch_borrowed(program, inputs, config)
                .map_err(|error| ProductionError::Dispatch(error.to_string())),
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
