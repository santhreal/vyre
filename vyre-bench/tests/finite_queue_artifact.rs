//! Finite resident-queue artifact compilation contracts.

use std::collections::BTreeMap;
use vyre::compiler::{self, CompileRequest, Digest, ExternalFacts, SearchBudget};
use vyre_foundation::ir::ProgramGraph;

/// WHY: finite host-submitted queue programs must reach the canonical CUDA target compiler;
/// a conditional early return was accepted by the legacy raw dispatch route but rejected by
/// verified PTX lowering because it could strand later synchronization.
#[test]
fn finite_queue_program_compiles_to_authenticated_cuda_payload() {
    let program =
        vyre_runtime::resident_work_queue::build_program_sharded_once_slots(256, 256, &[]);
    let graph = ProgramGraph::from_program("finite_queue", program)
        .expect("finite queue program must form a valid canonical graph");
    let request = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        SearchBudget::new(1, 1, 0, 0, 1),
        64 * 1024 * 1024,
    )
    .validate()
    .expect("finite queue compile request must validate");
    let artifact =
        compiler::compile(&request).expect("finite queue graph must compile to a neutral artifact");
    vyre_registry_link::backend::live_backend_registry()
        .expect("Fix: the backend registry must freeze cleanly");
    let registration =
        vyre_driver::backend::backend_registration(vyre_driver_cuda::CUDA_BACKEND_ID)
            .expect("CUDA target compiler registration must be linked");
    let compiler = registration
        .target_compiler()
        .expect("CUDA registration must provide a target compiler");
    let payload = compiler
        .compile(&artifact)
        .expect("finite queue artifact must lower to authenticated PTX");

    assert_eq!(payload.neutral_artifact(), artifact.digest());
    assert!(!payload.bytes().is_empty());
}
