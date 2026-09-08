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
        let outputs =
            reconcile_completion(expected_outputs, completion.outputs, completion.retained)?;
        Ok(SemanticExecutionOutput {
            artifact,
            payload,
            outputs,
        })
    }
}

/// Match an artifact completion against the graph values a request declares.
///
/// Every declared value arrives exactly once, through either channel: a
/// terminal buffer is delivered as an output, and a value the artifact carries
/// across dispatches is delivered as retained state.
///
/// A leftover output is a contract failure, because the caller receives a
/// buffer no graph value names. A leftover retained entry is not: retained
/// state is the pipeline's own carry, a fused subprogram and a fence-split
/// segment chain each produce some, and a one-shot semantic execution consumes
/// none of it. Reading the two as one fault reported a wrong answer for eight
/// operations whose only defect was retaining state, and suppressing the
/// retained half at the projection instead dropped the middle of every
/// retained chain.
fn reconcile_completion(
    expected_outputs: impl IntoIterator<Item = vyre_foundation::ir::GraphValueId>,
    mut completion_outputs: BTreeMap<ArtifactValueId, Vec<u8>>,
    mut completion_retained: BTreeMap<ArtifactValueId, Vec<u8>>,
) -> Result<BTreeMap<vyre_foundation::ir::GraphValueId, Vec<u8>>, SemanticExecutionError> {
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
    if !completion_outputs.is_empty() {
        return Err(SemanticExecutionError::Backend(format!(
            "artifact completion returned {} undeclared output value(s): {:?}. Fix: return only terminal graph values",
            completion_outputs.len(),
            completion_outputs.keys().map(|value| value.0).collect::<Vec<_>>()
        )));
    }
    Ok(outputs)
}

// Inline: `reconcile_completion` is crate-private, and it decides which of the
// two completion channels is a fault. Reaching it through the executor needs a
// live backend, so the device parity suite is the integration proof and these
// cover the classification itself.
#[cfg(test)]
mod reconcile_tests {
    use super::{reconcile_completion, ArtifactValueId, BTreeMap};
    use vyre_foundation::ir::GraphValueId;

    fn bytes(byte: u8) -> Vec<u8> {
        vec![byte; 4]
    }

    fn map(entries: &[(u32, u8)]) -> BTreeMap<ArtifactValueId, Vec<u8>> {
        entries
            .iter()
            .map(|(id, byte)| (ArtifactValueId(*id), bytes(*byte)))
            .collect()
    }

    #[test]
    fn a_declared_value_delivered_as_an_output_is_returned() {
        let outputs = reconcile_completion([GraphValueId(7)], map(&[(7, 0xAA)]), BTreeMap::new())
            .expect("a declared output must reconcile");
        assert_eq!(outputs.get(&GraphValueId(7)), Some(&bytes(0xAA)));
    }

    #[test]
    fn a_declared_value_delivered_as_retained_state_is_returned() {
        let outputs = reconcile_completion([GraphValueId(7)], BTreeMap::new(), map(&[(7, 0xBB)]))
            .expect("a declared value carried as retained state must reconcile");
        assert_eq!(outputs.get(&GraphValueId(7)), Some(&bytes(0xBB)));
    }

    #[test]
    fn undeclared_retained_state_is_not_a_fault() {
        let outputs = reconcile_completion(
            [GraphValueId(1)],
            map(&[(1, 0x11)]),
            map(&[(50, 0x50), (51, 0x51), (52, 0x52)]),
        )
        .expect("retained pipeline state the graph does not name must not fail a completion");
        assert_eq!(outputs.len(), 1, "only declared values are returned");
        assert_eq!(outputs.get(&GraphValueId(1)), Some(&bytes(0x11)));
    }

    #[test]
    fn an_undeclared_output_names_the_values_it_refuses() {
        let error = reconcile_completion(
            [GraphValueId(1)],
            map(&[(1, 0x11), (98, 0x98), (99, 0x99)]),
            BTreeMap::new(),
        )
        .expect_err("a buffer no graph value names must be refused");
        let message = error.to_string();
        assert!(
            message.contains("2 undeclared output value(s)"),
            "the count must be the leftover output count: {message}"
        );
        assert!(
            message.contains("98") && message.contains("99"),
            "the refusal must name which values were undeclared: {message}"
        );
        assert!(
            !message.contains("retained"),
            "an output fault must not report a retained one: {message}"
        );
    }

    #[test]
    fn a_value_delivered_through_both_channels_is_refused() {
        let error = reconcile_completion([GraphValueId(3)], map(&[(3, 0x33)]), map(&[(3, 0x33)]))
            .expect_err("one value delivered twice must be refused");
        assert!(
            error.to_string().contains("both output and retained state"),
            "the refusal must state the duplication: {error}"
        );
    }

    #[test]
    fn an_omitted_declared_value_is_refused_by_name() {
        let error = reconcile_completion([GraphValueId(4)], BTreeMap::new(), BTreeMap::new())
            .expect_err("a declared value that arrived through neither channel must be refused");
        let message = error.to_string();
        assert!(
            message.contains("omitted canonical graph output 4"),
            "the refusal must name the missing value: {message}"
        );
    }

    #[test]
    fn an_omission_is_reported_before_an_undeclared_output() {
        // A completion that both omits a declared value and carries an
        // undeclared one is two faults. The omission is the one the caller can
        // act on, because it names the value that never arrived.
        let error = reconcile_completion([GraphValueId(4)], map(&[(77, 0x77)]), BTreeMap::new())
            .expect_err("a completion missing a declared value must be refused");
        assert!(
            error.to_string().contains("omitted canonical graph output 4"),
            "the omission is the reported fault: {error}"
        );
    }
}
