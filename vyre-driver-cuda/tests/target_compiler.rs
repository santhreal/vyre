//! CUDA target-compiler registry and immutable module-bundle contracts.

use std::collections::BTreeMap;

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, Node, Program, ProgramGraph, ShapeDim, ValueContract,
    ValueLifetime,
};
use vyre_megakernel::{CompileRequest, Digest, ExternalFacts, SearchBudget, TargetModuleBundle};

fn artifact() -> vyre_megakernel::Artifact {
    let program = Program::wrapped(
        vec![BufferDecl::storage(
            "out",
            0,
            BufferAccess::ReadWrite,
            DataType::U32,
        )],
        [64, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
    );
    let mut graph = ProgramGraph::new();
    graph
        .add_external_value(
            "out",
            ValueContract {
                dtype: DataType::U32,
                shape: vec![ShapeDim::Known(1)],
                access: BufferAccess::ReadWrite,
                lifetime: ValueLifetime::Invocation,
            },
        )
        .unwrap();
    graph
        .add_node("main", program, Vec::new(), Vec::new())
        .unwrap();
    let request = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        SearchBudget::new(1, 1, 0, 0, 1),
        1_000_000,
    )
    .validate()
    .unwrap();
    vyre_megakernel::compile(&request).unwrap()
}

/// WHY: CUDA payload production must not acquire a GPU or compile a caller-owned Program.
#[test]
fn registered_target_compiler_emits_selected_ptx_bundle() {
    let registration = vyre_driver::backend::registered_backends()
        .iter()
        .find(|registration| registration.id == vyre_driver_cuda::CUDA_BACKEND_ID)
        .expect("CUDA target compiler registration must be linked");
    let compiler = registration.target_compiler().unwrap();
    let artifact = artifact();
    let payload = compiler.compile(&artifact).unwrap();
    let bundle = TargetModuleBundle::from_bytes(payload.bytes()).unwrap();
    assert_eq!(compiler.format().identity(), "ptx");
    assert_eq!(bundle.modules.len(), 1);
    let source = std::str::from_utf8(&bundle.modules[0].bytes).unwrap();
    assert!(source.contains(".visible .entry"));
    assert_eq!(payload.neutral_artifact(), artifact.digest());
}
