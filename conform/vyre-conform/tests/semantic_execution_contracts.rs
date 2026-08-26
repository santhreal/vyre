//! Observable contracts for conformance semantic requests and execution evidence.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use vyre::ir::{BufferDecl, DataType, Expr, Node, Program};
use vyre_conform::ProductionSession;
use vyre_megakernel::{
    CompileObjective, DeviceFacts, Digest, ExternalFacts, SearchBudget, SemanticExecutionError,
    SemanticExecutionOutput, SemanticExecutionPolicy, SemanticExecutionRequest, SemanticExecutor,
};

#[derive(Debug, PartialEq, Eq)]
struct ObservedRequest {
    inputs: Vec<Vec<u8>>,
    objective: CompileObjective,
    budget: SearchBudget,
    max_artifact_bytes: u64,
    target_facts: DeviceFacts,
}

struct RecordingExecutor {
    observed: Mutex<Option<ObservedRequest>>,
    artifact: Digest,
    payload: Digest,
    output: Vec<u8>,
}

impl SemanticExecutor for RecordingExecutor {
    fn execute(
        &self,
        request: &SemanticExecutionRequest<'_>,
    ) -> Result<SemanticExecutionOutput, SemanticExecutionError> {
        let inputs = request
            .inputs()
            .values()
            .map(|bytes| bytes.to_vec())
            .collect();
        *self.observed.lock().expect("recording executor lock") = Some(ObservedRequest {
            inputs,
            objective: request.objective(),
            budget: request.budget(),
            max_artifact_bytes: request.max_artifact_bytes(),
            target_facts: request.target_facts(),
        });
        let terminal_values = request
            .logical()
            .graph()
            .values()
            .iter()
            .filter(|value| value.producer.is_some() && value.consumers.is_empty())
            .map(|value| value.id)
            .collect::<BTreeSet<_>>();
        let outputs = terminal_values
            .into_iter()
            .map(|value| (value, self.output.clone()))
            .collect::<BTreeMap<_, _>>();
        Ok(SemanticExecutionOutput {
            artifact: self.artifact,
            payload: self.payload,
            outputs,
        })
    }
}

/// WHY: conformance must pass schedule-free semantics and explicit target policy
/// to the compiler boundary, then retain the admitted identities it returns.
#[test]
fn semantic_request_and_admitted_output_cross_the_production_boundary() {
    let program = Program::wrapped(
        vec![
            BufferDecl::read("input", 0, DataType::U32).with_count(1),
            BufferDecl::output("output", 1, DataType::U32).with_count(1),
        ],
        [8, 1, 1],
        vec![Node::store(
            "output",
            Expr::u32(0),
            Expr::load("input", Expr::u32(0)),
        )],
    );
    let artifact = Digest([17; 32]);
    let payload = Digest([29; 32]);
    let expected_output = 41_u32.to_le_bytes().to_vec();
    let executor = Arc::new(RecordingExecutor {
        observed: Mutex::new(None),
        artifact,
        payload,
        output: expected_output.clone(),
    });
    let target_facts = DeviceFacts::unknown();
    let budget = SearchBudget::new(19, 23, 1, 1, 31);
    let policy = SemanticExecutionPolicy::new(
        ExternalFacts::new(Digest([7; 32]), BTreeMap::new()),
        target_facts,
        CompileObjective::MinimizeLatency,
        budget,
        65_536,
    );
    let session = ProductionSession::with_executor(
        &program,
        executor.clone(),
        policy,
        "recording-semantic-backend",
    );
    let input = 37_u32.to_le_bytes();

    let execution = session
        .submit(&[&input])
        .expect("semantic executor must receive a valid canonical request");

    assert_eq!(execution.artifact, artifact);
    assert_eq!(execution.payload, payload);
    assert_eq!(execution.outputs, vec![expected_output]);
    assert_eq!(
        *executor.observed.lock().expect("recording executor lock"),
        Some(ObservedRequest {
            inputs: vec![input.to_vec()],
            objective: CompileObjective::MinimizeLatency,
            budget,
            max_artifact_bytes: 65_536,
            target_facts,
        })
    );
}
