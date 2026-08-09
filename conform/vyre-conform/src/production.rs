//! Production compiler, target payload, materialization, and submission route.

use std::collections::BTreeMap;

use thiserror::Error;
use vyre_driver::{BackendRegistration, BindingSet, BoundResource, Completion};
use vyre_foundation::ir::{BufferAccess, Program, ProgramGraph};
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
    /// Target compilation or payload association failed.
    #[error("production conformance target compilation failed: {0}")]
    Target(String),
    /// Artifact materialization or submission failed.
    #[error(transparent)]
    Runtime(#[from] ArtifactSessionError),
    /// Caller input count does not match the compiler-owned ABI.
    #[error("production conformance bindings rejected: {0}")]
    Bindings(String),
}

/// Materialized production artifact used for repeated conformance submissions.
pub struct ProductionSession {
    program: Program,
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
        let neutral = vyre_megakernel::compile(&request)
            .map_err(|error| ProductionError::Compile(error.to_string()))?;
        let neutral_digest = neutral.digest();
        let target_compiler = registration
            .target_compiler()
            .map_err(|error| ProductionError::Target(error.to_string()))?;
        let envelope = vyre_megakernel::attach_target(neutral, target_compiler.as_ref())
            .map_err(|error| ProductionError::Target(error.to_string()))?;
        let payload_digest = envelope.target_payloads()[0].digest();
        let bytes = envelope
            .to_bytes()
            .map_err(|error| ProductionError::Target(error.to_string()))?;
        let session = ArtifactSession::from_bytes(registration, &bytes)?;
        Ok(Self {
            program: program.clone(),
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
        let mut input_buffers = self
            .program
            .buffers()
            .iter()
            .filter(|buffer| {
                matches!(
                    buffer.access(),
                    BufferAccess::ReadOnly | BufferAccess::ReadWrite | BufferAccess::Uniform
                )
            })
            .collect::<Vec<_>>();
        input_buffers.sort_unstable_by_key(|buffer| buffer.binding());
        if input_buffers.len() != inputs.len() {
            return Err(ProductionError::Bindings(format!(
                "artifact ABI requires {} input buffers, caller supplied {}. Fix: build witness inputs from the Program buffer declarations.",
                input_buffers.len(),
                inputs.len()
            )));
        }

        let mut bindings: BindingSet = self.session.bindings()?;
        for (buffer, bytes) in input_buffers.into_iter().zip(inputs) {
            let value = self.session.resource(buffer.name())?;
            bindings.insert(value, BoundResource::Host(bytes.to_vec()));
        }
        let completion = self.session.submit_and_wait(bindings)?;
        self.ordered_outputs(completion)
    }

    fn ordered_outputs(&self, completion: Completion) -> Result<Vec<Vec<u8>>, ProductionError> {
        let mut writable = self
            .program
            .buffers()
            .iter()
            .filter(|buffer| {
                matches!(
                    buffer.access(),
                    BufferAccess::ReadWrite | BufferAccess::WriteOnly
                )
            })
            .collect::<Vec<_>>();
        writable.sort_unstable_by_key(|buffer| buffer.binding());
        writable
            .into_iter()
            .map(|buffer| {
                let value = self.session.resource(buffer.name())?;
                completion
                    .outputs
                    .get(&value)
                    .or_else(|| completion.retained.get(&value))
                    .cloned()
                    .ok_or_else(|| {
                        ProductionError::Bindings(format!(
                            "completion omitted writable artifact value `{}`. Fix: materializer output projection must follow the canonical artifact ABI.",
                            buffer.name()
                        ))
                    })
            })
            .collect()
    }
}
