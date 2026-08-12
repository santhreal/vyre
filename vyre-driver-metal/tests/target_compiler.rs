//! Metal target-compiler registry and immutable module-bundle contracts.

use std::collections::BTreeMap;
#[cfg(any(target_os = "macos", target_os = "ios"))]
use vyre_driver::BindingSet;

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

/// WHY: pure Metal target compilation must remain available without a live Apple device.
#[test]
fn registered_target_compiler_emits_selected_metal_bundle() {
    let registration = vyre_driver::backend::registered_backends()
        .expect("valid backend registry")
        .iter()
        .find(|registration| registration.id == vyre_driver_metal::METAL_BACKEND_ID)
        .expect("Metal target compiler registration must be linked on every host");
    let compiler = registration.target_compiler().unwrap();
    let artifact = artifact();
    let payload = compiler.compile(&artifact).unwrap();
    let bundle = TargetModuleBundle::from_bytes(payload.bytes()).unwrap();
    assert_eq!(compiler.format().identity(), "msl");
    assert_eq!(compiler.format().version(), 2);
    assert_eq!(bundle.modules.len(), 1);
    let target: vyre_emit_metal::MetalArtifact =
        serde_json::from_slice(&bundle.modules[0].bytes).unwrap();
    assert!(target.msl.contains("kernel"));
    assert_eq!(target.entry_point, bundle.modules[0].entry_point);
    assert_eq!(payload.entries()[0].name, target.entry_point);
    assert_eq!(payload.neutral_artifact(), artifact.digest());
}

/// WHY: the registered materializer must fail explicitly when Metal.framework is unavailable.
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
#[test]
fn registered_materializer_reports_platform_unavailability() {
    let registration = vyre_driver::backend::registered_backends()
        .expect("valid backend registry")
        .iter()
        .find(|registration| registration.id == vyre_driver_metal::METAL_BACKEND_ID)
        .expect("Metal materializer registration must be linked");
    let error = registration
        .materializer()
        .err()
        .expect("non-Apple Metal materialization must fail explicitly");
    assert!(matches!(
        error,
        vyre_driver::BackendError::UnsupportedFeature { .. }
    ));
}

/// WHY: native Metal materialization must execute the authenticated structured MSL artifact.
#[cfg(any(target_os = "macos", target_os = "ios"))]
#[test]
fn registered_materializer_executes_authenticated_msl() {
    let registration = vyre_driver::backend::registered_backends()
        .expect("valid backend registry")
        .iter()
        .find(|registration| registration.id == vyre_driver_metal::METAL_BACKEND_ID)
        .expect("Metal materializer registration must be linked");
    let compiler = registration.target_compiler().unwrap();
    let materializer = registration
        .materializer()
        .expect("Metal materializer must acquire on an Apple GPU host");
    let artifact = artifact();
    let payload = compiler.compile(&artifact).unwrap();
    let instance = materializer.materialize(&artifact, &payload).unwrap();
    let completion = instance
        .submit(BindingSet::new(artifact.digest()))
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(
        completion.outputs.get(&vyre_megakernel::ArtifactValueId(0)),
        Some(&1_u32.to_le_bytes().to_vec())
    );
}
