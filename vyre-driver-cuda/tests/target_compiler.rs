//! CUDA target-compiler registry and immutable module-bundle contracts.

use std::collections::BTreeMap;
use vyre_driver::{BindingSet, BoundResource};

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, GraphOutput, Node, Program, ProgramGraph, ShapeDim,
    ValueContract, ValueLifetime,
};
use vyre_megakernel::{CompileRequest, Digest, ExternalFacts, SearchBudget};

fn artifact() -> vyre_megakernel::Artifact {
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [64, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
    );
    let mut graph = ProgramGraph::new();
    graph
        .add_node(
            "main",
            program,
            Vec::new(),
            vec![GraphOutput {
                buffer: "out".into(),
                name: "out".into(),
                contract: ValueContract {
                    dtype: DataType::U32,
                    shape: vec![ShapeDim::Known(1)],
                    access: BufferAccess::ReadWrite,
                    lifetime: ValueLifetime::Output,
                },
                retained_successor_of: None,
            }],
        )
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
/// WHY: CUDA materialization must load authenticated PTX and execute without re-emitting it.
#[test]
fn registered_materializer_executes_authenticated_ptx() {
    let registration = vyre_driver::backend::registered_backends()
        .expect("valid backend registry")
        .iter()
        .find(|registration| registration.id == vyre_driver_cuda::CUDA_BACKEND_ID)
        .expect("CUDA materializer registration must be linked");
    let compiler = registration.target_compiler().unwrap();
    let materializer = registration
        .materializer()
        .expect("CUDA materializer must acquire on the GPU-required host");
    let artifact = artifact();
    let payload = compiler.compile(&artifact).unwrap();
    let instance = materializer.materialize(&artifact, &payload).unwrap();
    let completion = instance
        .submit(BindingSet::new(artifact.digest()))
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(completion.artifact, artifact.digest());
    assert_eq!(
        completion.outputs.get(&vyre_megakernel::ArtifactValueId(0)),
        Some(&1_u32.to_le_bytes().to_vec())
    );
}

/// WHY: resident resources must remain inside the authenticated artifact route.
#[test]
fn registered_materializer_executes_authenticated_ptx_with_resident_bindings() {
    let registration =
        vyre_driver::backend::backend_registration(vyre_driver_cuda::CUDA_BACKEND_ID)
            .expect("CUDA materializer registration must be linked");
    let compiler = registration.target_compiler().unwrap();
    let materializer = registration
        .materializer()
        .expect("CUDA materializer must acquire on the GPU-required host");
    let artifact = artifact();
    let payload = compiler.compile(&artifact).unwrap();
    let instance = materializer.materialize(&artifact, &payload).unwrap();
    let resource = materializer.allocate_resident(4).unwrap();
    materializer
        .upload_resident(&resource, &0_u32.to_le_bytes())
        .unwrap();
    let mut bindings = BindingSet::new(artifact.digest());
    bindings.insert(
        vyre_megakernel::ArtifactValueId(0),
        BoundResource::Resident(resource.clone()),
    );
    let completion = instance.submit(bindings).unwrap().wait().unwrap();
    assert_eq!(
        completion.outputs.get(&vyre_megakernel::ArtifactValueId(0)),
        Some(&1_u32.to_le_bytes().to_vec())
    );
    materializer.free_resident(resource).unwrap();
}
