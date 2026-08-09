//! Canonical compiler, target payload, materialization, and submission seam for scan products.

use std::collections::BTreeMap;
use std::time::Instant;

use thiserror::Error;
use vyre_driver::{
    BackendError, BackendRegistration, BindingSet, BoundResource, Completion, DeviceIdentity,
    Submission, TimedDispatchResult,
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
    /// Product-side validation or staging failed before artifact submission.
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
        let session = ArtifactSession::compile(registration, &request)?;
        let artifact_digest = session.artifact()?;
        let payload_digest = session.payload()?;
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
    /// Submit host buffers in canonical ABI order and return writable values in ABI order.
    pub fn submit_ordered(&self, inputs: &[&[u8]]) -> Result<Vec<Vec<u8>>, ScanArtifactError> {
        let completion = self.session.submit_host_inputs(inputs)?;
        Ok(self.session.ordered_outputs(&completion)?)
    }
    /// Submit host buffers in canonical ABI order and return submission timing.
    pub fn submit_ordered_timed(
        &self,
        inputs: &[&[u8]],
    ) -> Result<TimedDispatchResult, ScanArtifactError> {
        let start = Instant::now();
        let completion = self.session.submit_host_inputs(inputs)?;
        let outputs = self.session.ordered_outputs(&completion)?;
        Ok(TimedDispatchResult {
            outputs,
            wall_ns: u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX),
            device_ns: completion.device_ns,
            enqueue_ns: None,
            wait_ns: None,
        })
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
pub(crate) fn dispatch_registered(
    program: &Program,
    backend_id: &str,
    inputs: &[&[u8]],
) -> Result<Vec<Vec<u8>>, BackendError> {
    let registration = vyre_driver::backend::backend_registration(backend_id)?;
    let session = ScanArtifactSession::compile(program, registration)
        .map_err(|error| BackendError::new(error.to_string()))?;
    session
        .submit_ordered(inputs)
        .map_err(|error| BackendError::new(error.to_string()))
}

pub(crate) fn dispatch_registered_timed(
    program: &Program,
    backend_id: &str,
    inputs: &[&[u8]],
) -> Result<TimedDispatchResult, BackendError> {
    let registration = vyre_driver::backend::backend_registration(backend_id)?;
    let session = ScanArtifactSession::compile(program, registration)
        .map_err(|error| BackendError::new(error.to_string()))?;
    session
        .submit_ordered_timed(inputs)
        .map_err(|error| BackendError::new(error.to_string()))
}

pub(crate) struct ArtifactPendingDispatch {
    session: ScanArtifactSession,
    submission: Box<dyn Submission>,
}

impl ArtifactPendingDispatch {
    #[must_use]
    pub(crate) fn is_ready(&self) -> bool {
        self.submission.is_ready()
    }

    pub(crate) fn await_result(self) -> Result<Vec<Vec<u8>>, BackendError> {
        let completion = self
            .submission
            .wait()
            .map_err(|error| BackendError::new(error.to_string()))?;
        self.session
            .session
            .ordered_outputs(&completion)
            .map_err(|error| BackendError::new(error.to_string()))
    }
}

pub(crate) fn dispatch_registered_async(
    program: &Program,
    backend_id: &str,
    inputs: &[Vec<u8>],
) -> Result<ArtifactPendingDispatch, BackendError> {
    let registration = vyre_driver::backend::backend_registration(backend_id)?;
    let session = ScanArtifactSession::compile(program, registration)
        .map_err(|error| BackendError::new(error.to_string()))?;
    let borrowed = inputs.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let bindings = session
        .session
        .host_bindings(&borrowed)
        .map_err(|error| BackendError::new(error.to_string()))?;
    let submission = session
        .session
        .submit(bindings)
        .map_err(|error| BackendError::new(error.to_string()))?;
    Ok(ArtifactPendingDispatch {
        session,
        submission,
    })
}
