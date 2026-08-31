//! Registered artifact-backed implementation of semantic execution.

use std::collections::BTreeMap;

use vyre_driver::{BackendRegistration, BindingSet, BoundResource};
use vyre_megakernel::{
    ArtifactValueId, SemanticExecutionError, SemanticExecutionOutput, SemanticExecutionRequest,
    SemanticExecutor,
};

use crate::artifact_admission::{ArtifactSession, ArtifactSessionError};

/// Compiler and artifact runtime bound to one registered backend.
pub struct RegisteredSemanticExecutor {
    registration: &'static BackendRegistration,
}

impl RegisteredSemanticExecutor {
    /// Bind semantic execution to one immutable backend registration.
    #[must_use]
    pub const fn new(registration: &'static BackendRegistration) -> Self {
        Self { registration }
    }

    /// Return the registered backend used for target compilation and admission.
    #[must_use]
    pub const fn registration(&self) -> &'static BackendRegistration {
        self.registration
    }
}

impl SemanticExecutor for RegisteredSemanticExecutor {
    fn execute(
        &self,
        request: &SemanticExecutionRequest<'_>,
    ) -> Result<SemanticExecutionOutput, SemanticExecutionError> {
        let compile_request = request
            .compile_request()
            .validate()
            .map_err(SemanticExecutionError::Compile)?;
        let session =
            ArtifactSession::compile(self.registration, &compile_request).map_err(|error| {
                match error {
                    ArtifactSessionError::Compile(error) => SemanticExecutionError::Compile(error),
                    ArtifactSessionError::Target(error) => SemanticExecutionError::Target(error),
                    error => SemanticExecutionError::Backend(error.to_string()),
                }
            })?;
        let artifact = session
            .artifact()
            .map_err(|error| SemanticExecutionError::Backend(error.to_string()))?;
        let payload = session
            .payload()
            .map_err(|error| SemanticExecutionError::Backend(error.to_string()))?;
        let mut bindings = BindingSet::new(artifact);
        for (value, bytes) in request.inputs() {
            bindings.insert(
                ArtifactValueId(value.0),
                BoundResource::Host(bytes.to_vec()),
            );
        }
        let completion = session
            .submit_and_wait(bindings)
            .map_err(|error| SemanticExecutionError::Backend(error.to_string()))?;
        let expected_outputs = vyre_megakernel::returned_graph_values(request.logical().graph());
        let mut completion_outputs = completion.outputs;
        let mut completion_retained = completion.retained;
        let mut outputs = BTreeMap::new();
        for value in expected_outputs {
            let artifact_value = ArtifactValueId(value.0);
            let output = completion_outputs.remove(&artifact_value);
            let retained = completion_retained.remove(&artifact_value);
            let bytes = match (output, retained) {
                (Some(_), Some(_)) => {
                    return Err(SemanticExecutionError::Backend(format!(
                        "artifact completion returned graph value {} as both output and retained state. Fix: emit each canonical graph value once",
                        value.0
                    )));
                }
                (Some(bytes), None) | (None, Some(bytes)) => bytes,
                (None, None) => {
                    return Err(SemanticExecutionError::Backend(format!(
                        "artifact completion omitted canonical graph output {}. Fix: return every terminal graph value exactly once",
                        value.0
                    )));
                }
            };
            outputs.insert(value, bytes);
        }
        if !completion_outputs.is_empty() || !completion_retained.is_empty() {
            return Err(SemanticExecutionError::Backend(format!(
                "artifact completion returned {} undeclared output value(s) and {} undeclared retained value(s). Fix: return only terminal graph values",
                completion_outputs.len(),
                completion_retained.len()
            )));
        }
        Ok(SemanticExecutionOutput {
            artifact,
            payload,
            outputs,
        })
    }
}
