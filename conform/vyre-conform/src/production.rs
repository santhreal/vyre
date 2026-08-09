//! Production compiler, target payload, materialization, and submission route.

use std::collections::BTreeMap;

use thiserror::Error;
use vyre_driver::BackendRegistration;
use vyre_foundation::ir::{Program, ProgramGraph};
use vyre_megakernel::{CompileRequest, Digest, ExternalFacts, SearchBudget};
use vyre_runtime::{ArtifactSession, ArtifactSessionError};

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
