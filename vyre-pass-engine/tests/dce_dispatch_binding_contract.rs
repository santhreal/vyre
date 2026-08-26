//! Semantic optimizer pipeline execution contract.

use std::collections::BTreeMap;
use std::sync::Mutex;

use vyre_driver_reference::ReferenceSemanticExecutor;
use vyre_foundation::ir::{BufferDecl, DataType, Expr, GraphValueId, Node, Program, ValueLifetime};
use vyre_foundation::validate::BackendCapabilities;
use vyre_megakernel::{
    CompileObjective, DeviceFacts, Digest, ExternalFacts, SearchBudget, SemanticExecutionError,
    SemanticExecutionOutput, SemanticExecutionPolicy, SemanticExecutionRequest, SemanticExecutor,
};
use vyre_pass_engine::optimizer::dce_via_encoded::DceError;
use vyre_pass_engine::optimizer::pipeline::{gpu_optimize, GpuOptimizeError};

#[derive(Clone, Debug, PartialEq, Eq)]
struct RequestSnapshot {
    stage: String,
    semantic_graph: Vec<u8>,
    inputs: Vec<(GraphValueId, usize)>,
    external_facts: ExternalFacts,
    target_facts: DeviceFacts,
    objective: CompileObjective,
    budget: SearchBudget,
    max_artifact_bytes: u64,
}

struct RecordingExecutor {
    hostile_route_preference: bool,
    fail_stage: Option<&'static str>,
    miskey_stage: Option<&'static str>,
    requests: Mutex<Vec<RequestSnapshot>>,
}

impl RecordingExecutor {
    fn new(hostile_route_preference: bool) -> Self {
        Self {
            hostile_route_preference,
            fail_stage: None,
            miskey_stage: None,
            requests: Mutex::new(Vec::new()),
        }
    }

    fn snapshots(&self) -> Vec<RequestSnapshot> {
        self.requests.lock().expect("request lock").clone()
    }
}

impl SemanticExecutor for RecordingExecutor {
    fn execute(
        &self,
        request: &SemanticExecutionRequest<'_>,
    ) -> Result<SemanticExecutionOutput, SemanticExecutionError> {
        let node = &request.logical().graph().nodes()[0];
        let stage = node.name.as_str();
        self.requests
            .lock()
            .expect("request lock")
            .push(RequestSnapshot {
                stage: stage.to_string(),
                semantic_graph: request.logical().semantic_wire().to_vec(),
                inputs: request
                    .inputs()
                    .iter()
                    .map(|(id, bytes)| (*id, bytes.len()))
                    .collect(),
                external_facts: request.external_facts().clone(),
                target_facts: request.target_facts(),
                objective: request.objective(),
                budget: request.budget(),
                max_artifact_bytes: request.max_artifact_bytes(),
            });
        if self.fail_stage == Some(stage) {
            return Err(SemanticExecutionError::Backend(format!(
                "{stage} hostile executor failure"
            )));
        }

        // This preference is intentionally hostile and intentionally unused. The
        // semantic request contains no route or launch control for it to alter.
        let _ = self.hostile_route_preference;
        let first_words = request
            .inputs()
            .values()
            .next()
            .map_or(0, |bytes| bytes.len() / 4);
        let encoded = |words: Vec<u32>| {
            words
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>()
        };
        let payloads = match stage {
            "canonicalize" | "pattern-match" => vec![encoded(vec![0; first_words])],
            "const-fold" => vec![encoded(vec![0; first_words]), encoded(vec![0; first_words])],
            "dce" => {
                let frontier_words = first_words.div_ceil(32).max(1);
                vec![
                    encoded(vec![u32::MAX; frontier_words]),
                    encoded(vec![0]),
                    encoded(vec![1]),
                ]
            }
            other => panic!("unexpected semantic optimizer stage {other}"),
        };
        let output_ids = if node.outputs.is_empty() {
            request
                .logical()
                .graph()
                .values()
                .iter()
                .filter_map(|value| {
                    (value.contract.lifetime == ValueLifetime::Retained).then_some(value.id)
                })
                .collect::<Vec<_>>()
        } else {
            node.outputs.clone()
        };
        assert_eq!(output_ids.len(), payloads.len());
        let mut outputs = output_ids
            .into_iter()
            .zip(payloads)
            .collect::<BTreeMap<_, _>>();
        if self.miskey_stage == Some(stage) {
            let bytes = outputs.pop_first().expect("stage has a canonical output").1;
            outputs.insert(GraphValueId(u32::MAX), bytes);
        }
        Ok(SemanticExecutionOutput {
            artifact: Digest([7; 32]),
            payload: Digest([8; 32]),
            outputs,
        })
    }
}

/// Facts for a device that reports the shared memory the pass programs use.
///
/// `DeviceFacts::unknown()` grants nothing, so a pass whose program declares
/// workgroup scratch is rejected against it. The passes here are compiled, not
/// hypothetically judged, so the policy names a device that can run them.
fn policy() -> SemanticExecutionPolicy {
    SemanticExecutionPolicy::new(
        ExternalFacts::new(Digest([3; 32]), BTreeMap::new()),
        DeviceFacts::new(
            BackendCapabilities {
                has_shared_memory: true,
                max_shared_memory_bytes: 32 * 1024,
                ..BackendCapabilities::default()
            },
            256,
        ),
        CompileObjective::MinimizeLatency,
        SearchBudget::new(8, 64, 0, 0, 1_000),
        1_000_000,
    )
}

fn input_program() -> Program {
    Program::wrapped(
        Vec::new(),
        [1, 1, 1],
        vec![Node::let_bind("sum", Expr::add(Expr::u32(2), Expr::u32(3)))],
    )
}

#[test]
fn full_pipeline_stage_order_and_graph_value_inputs_are_closed() {
    let executor = RecordingExecutor::new(false);
    let optimized = gpu_optimize(input_program(), &executor, &policy()).expect("pipeline succeeds");
    assert_eq!(optimized, input_program());

    let snapshots = executor.snapshots();
    assert_eq!(
        snapshots
            .iter()
            .map(|request| request.stage.as_str())
            .collect::<Vec<_>>(),
        ["canonicalize", "const-fold", "dce", "pattern-match"]
    );
    for request in snapshots {
        assert!(!request.semantic_graph.is_empty());
        assert!(!request.inputs.is_empty());
        assert!(request.inputs.windows(2).all(|pair| pair[0].0 < pair[1].0));
    }
}

#[test]
fn hostile_route_preferences_cannot_change_semantic_requests() {
    let ordinary = RecordingExecutor::new(false);
    let hostile = RecordingExecutor::new(true);
    gpu_optimize(input_program(), &ordinary, &policy()).expect("ordinary pipeline succeeds");
    gpu_optimize(input_program(), &hostile, &policy()).expect("hostile pipeline succeeds");
    assert_eq!(ordinary.snapshots(), hostile.snapshots());
}

#[test]
fn semantic_backend_failure_keeps_dce_stage_classification_and_context() {
    let executor = RecordingExecutor {
        hostile_route_preference: true,
        fail_stage: Some("dce"),
        requests: Mutex::new(Vec::new()),
        miskey_stage: None,
    };
    let error = gpu_optimize(input_program(), &executor, &policy()).expect_err("dce must fail");
    assert!(matches!(
        error,
        GpuOptimizeError::Dce(DceError::Semantic(SemanticExecutionError::Backend(_)))
    ));
    assert!(error.to_string().contains("gpu_optimize dce"));
    assert!(error
        .to_string()
        .contains("semantic artifact execution failed"));
}

#[test]
fn final_stage_decoding_requires_the_canonical_graph_output_identity() {
    let executor = RecordingExecutor {
        hostile_route_preference: false,
        fail_stage: None,
        miskey_stage: Some("pattern-match"),
        requests: Mutex::new(Vec::new()),
    };
    let error =
        gpu_optimize(input_program(), &executor, &policy()).expect_err("wrong output id must fail");
    assert!(matches!(error, GpuOptimizeError::AlgebraicIdentities(_)));
    assert!(error.to_string().contains("omitted canonical output value"));
}

#[test]
fn reference_semantic_executor_runs_the_complete_pipeline() {
    let input = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::add(Expr::u32(2), Expr::u32(3)),
        )],
    );
    let expected = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(5))],
    );
    let optimized = gpu_optimize(input, &ReferenceSemanticExecutor, &policy())
        .expect("reference semantic pipeline succeeds");
    assert_eq!(optimized, expected);
}
