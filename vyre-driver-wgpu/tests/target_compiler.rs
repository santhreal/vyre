//! WGPU target-compiler registry and immutable module-bundle contracts.

use std::collections::BTreeMap;
use vyre_driver::{BindingSet, BoundResource};

use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, GraphOutput, Node, Program, ProgramGraph, ShapeDim,
    ValueContract, ValueLifetime,
};
use vyre_megakernel::{CompileRequest, Digest, ExternalFacts, SearchBudget, TargetModuleBundle};

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
                    access: BufferAccess::WriteOnly,
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

/// WHY: WGPU payload production is a pure registered compiler operation with no device probe.
#[test]
fn registered_target_compiler_emits_selected_wgsl_bundle() {
    let registration = vyre_driver::backend::registered_backends()
        .expect("valid backend registry")
        .iter()
        .find(|registration| registration.id == vyre_driver_wgpu::WGPU_BACKEND_ID)
        .expect("WGPU target compiler registration must be linked");
    let compiler = registration.target_compiler().unwrap();
    let artifact = artifact();
    let payload = compiler.compile(&artifact).unwrap();
    let bundle = TargetModuleBundle::from_bytes(payload.bytes()).unwrap();
    assert_eq!(compiler.format().identity(), "wgsl");
    assert_eq!(compiler.format().version(), 2);
    assert_eq!(bundle.modules.len(), 1);
    let source = std::str::from_utf8(&bundle.modules[0].bytes).unwrap();
    assert!(source.contains("@compute"));
    assert_eq!(payload.neutral_artifact(), artifact.digest());
}

/// WHY: target support is a facet of the canonical semantic identity, not a
/// second backend-owned operation catalog.
#[test]
fn registered_target_facets_resolve_canonical_operations() {
    let facets = vyre_driver::backend::registered_target_operation_facets()
        .expect("valid target facet registry")
        .iter()
        .filter(|facet| facet.target_id == vyre_driver_wgpu::WGPU_BACKEND_ID)
        .collect::<Vec<_>>();
    assert!(
        !facets.is_empty(),
        "WGPU target compiler must expose at least one supported canonical operation"
    );
    for facet in facets {
        let operation = vyre_foundation::operation::OperationRegistry::global()
            .get(facet.operation_id)
            .expect("target facet must resolve one canonical semantic operation");
        assert!(
            operation.build.is_some(),
            "{} target facet must reference a neutral program",
            facet.operation_id
        );
    }
}

/// WHY: WGPU materialization must execute authenticated WGSL instead of re-emitting a Program.
#[test]
fn registered_materializer_executes_authenticated_wgsl() {
    let registration = vyre_driver::backend::registered_backends()
        .expect("valid backend registry")
        .iter()
        .find(|registration| registration.id == vyre_driver_wgpu::WGPU_BACKEND_ID)
        .expect("WGPU materializer registration must be linked");
    let compiler = registration.target_compiler().unwrap();
    let materializer = registration
        .materializer()
        .expect("WGPU materializer must acquire on the GPU-required host");
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

/// WHY: resident benchmark hot loops must submit authenticated artifact instances,
/// not bypass materialization through raw `Program` dispatch.
#[test]
fn registered_materializer_executes_resident_artifact_bindings() {
    let registration =
        vyre_driver::backend::backend_registration(vyre_driver_wgpu::WGPU_BACKEND_ID)
            .expect("WGPU materializer registration must be linked");
    let compiler = registration.target_compiler().unwrap();
    let materializer = registration
        .materializer()
        .expect("WGPU materializer must acquire on the GPU-required host");
    let artifact = artifact();
    let payload = compiler.compile(&artifact).unwrap();
    let instance = materializer.materialize(&artifact, &payload).unwrap();
    let resource = materializer.allocate_resident(4).unwrap();
    materializer.upload_resident(&resource, &[0; 4]).unwrap();
    let mut bindings = BindingSet::new(artifact.digest());
    bindings.insert(
        vyre_megakernel::ArtifactValueId(0),
        BoundResource::Resident(resource.clone()),
    );
    let completion = instance.submit(bindings).unwrap().wait().unwrap();
    materializer.free_resident(resource).unwrap();
    assert_eq!(
        completion.outputs.get(&vyre_megakernel::ArtifactValueId(0)),
        Some(&1_u32.to_le_bytes().to_vec())
    );
}
