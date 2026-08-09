//! Canonical compiler, target payload, materialization, and submission seam for scan products.

use std::collections::BTreeMap;

use thiserror::Error;
use vyre_driver::{
    BackendError, BackendRegistration, BindingSet, BoundResource, Completion, DeviceIdentity,
};
use vyre_foundation::ir::{Program, ProgramGraph};
use vyre_megakernel::{CompileRequest, Digest, ExternalFacts, SearchBudget};
use vyre_runtime::{ArtifactSession, ArtifactSessionError};

const MAX_SCAN_ARTIFACT_BYTES: u64 = 64 * 1024 * 1024;
const SCAN_SEARCH_BUDGET: SearchBudget = SearchBudget::new(256, 100_000, 1, 0, 1_000_000_000);

/// Failure while compiling or executing one canonical scan artifact.
#[derive(Debug, Error)]
pub enum ScanArtifactError {
    /// The scan program could not enter the canonical graph.
    #[error("scan artifact graph construction failed: {0}")]
    Graph(String),
    /// Whole-program compilation or envelope encoding failed.
    #[error("scan artifact compilation failed: {0}")]
    Compile(String),
    /// The registered target compiler facet rejected the selected artifact.
    #[error("scan target compilation failed: {0}")]
    Target(String),
    /// A required registered compiler or materializer facet is unavailable.
    #[error(transparent)]
    Backend(#[from] BackendError),
    /// Artifact admission, materialization, submission, or readback failed.
    #[error(transparent)]
    Runtime(#[from] ArtifactSessionError),
    /// Typed completion omitted a required scan value.
    #[error("scan artifact completion failed: {0}")]
    Completion(String),
}

/// Immutable compiled scan artifact materialized for one registered target.
pub struct ScanArtifactSession {
    artifact: Digest,
    payload: Digest,
    session: ArtifactSession,
}

impl ScanArtifactSession {
    /// Compile one scan program, attach the registered target payload, authenticate it,
    /// and materialize the exact target bytes.
    pub fn compile(
        program: &Program,
        registration: &'static BackendRegistration,
    ) -> Result<Self, ScanArtifactError> {
        let graph = ProgramGraph::from_program("scan", program.clone())
            .map_err(|error| ScanArtifactError::Graph(error.to_string()))?;
        let request = CompileRequest::new(
            graph,
            ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
            SCAN_SEARCH_BUDGET,
            MAX_SCAN_ARTIFACT_BYTES,
        )
        .validate()
        .map_err(|error| ScanArtifactError::Compile(error.to_string()))?;
        let artifact = vyre_megakernel::compile(&request)
            .map_err(|error| ScanArtifactError::Compile(error.to_string()))?;
        let artifact_digest = artifact.digest();
        let compiler = registration.target_compiler()?;
        let envelope = vyre_megakernel::attach_target(artifact, compiler.as_ref())
            .map_err(|error| ScanArtifactError::Target(error.to_string()))?;
        let payload_digest = envelope.target_payloads()[0].digest();
        let bytes = envelope
            .to_bytes()
            .map_err(|error| ScanArtifactError::Compile(error.to_string()))?;
        let session = ArtifactSession::from_bytes(registration, &bytes)?;
        Ok(Self {
            artifact: artifact_digest,
            payload: payload_digest,
            session,
        })
    }

    /// Neutral artifact identity shared by every materialization generation.
    #[must_use]
    pub const fn artifact_digest(&self) -> Digest {
        self.artifact
    }

    /// Exact authenticated target payload identity.
    #[must_use]
    pub const fn payload_digest(&self) -> Digest {
        self.payload
    }

    /// Current materialized device generation.
    pub fn device(&self) -> Result<DeviceIdentity, ScanArtifactError> {
        Ok(self.session.device()?)
    }

    /// Submit host bytes keyed by compiler-owned resource names.
    pub fn submit<'a>(
        &self,
        resources: impl IntoIterator<Item = (&'a str, &'a [u8])>,
    ) -> Result<Completion, ScanArtifactError> {
        let mut bindings: BindingSet = self.session.bindings()?;
        for (name, bytes) in resources {
            let value = self.session.resource(name)?;
            bindings.insert(value, BoundResource::Host(bytes.to_vec()));
        }
        Ok(self.session.submit_and_wait(bindings)?)
    }

    /// Return one writable value from typed output or retained completion state.
    pub fn completion_value<'a>(
        &self,
        completion: &'a Completion,
        name: &str,
    ) -> Result<&'a [u8], ScanArtifactError> {
        let value = self.session.resource(name)?;
        completion
            .outputs
            .get(&value)
            .or_else(|| completion.retained.get(&value))
            .map(Vec::as_slice)
            .ok_or_else(|| {
                ScanArtifactError::Completion(format!(
                    "materializer omitted writable resource `{name}`. Fix: project every canonical scan ABI output by artifact value identity."
                ))
            })
    }
}
