//! Metal target-compiler registry and immutable module-bundle contracts.

// Everything below the registry check builds and compiles an artifact, which
// only the Apple-gated tests do. The non-Apple test asserts the absence of a
// registration and reaches for none of it.
#[cfg(any(target_os = "macos", target_os = "ios"))]
use std::collections::BTreeMap;

#[cfg(any(target_os = "macos", target_os = "ios"))]
use vyre_driver::BindingSet;
#[cfg(any(target_os = "macos", target_os = "ios"))]
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, GraphOutput, Node, Program, ProgramGraph, ShapeDim,
    ValueContract, ValueLifetime,
};
#[cfg(any(target_os = "macos", target_os = "ios"))]
use vyre_megakernel::{CompileRequest, Digest, ExternalFacts, SearchBudget, TargetModuleBundle};

#[cfg(any(target_os = "macos", target_os = "ios"))]
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

#[cfg(any(target_os = "macos", target_os = "ios"))]
/// WHY: pure Metal target compilation remains available without acquiring a device on Apple hosts.
#[test]
fn registered_target_compiler_emits_selected_metal_bundle() {
    let registration = vyre_driver::registered_backends()
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

/// WHY: non-Apple hosts must not publish a fake linked Metal backend.
#[cfg(not(any(target_os = "macos", target_os = "ios")))]
#[test]
fn non_apple_hosts_publish_no_metal_registration() {
    let registrations =
        vyre_driver::registered_backends().expect("valid backend registry");
    assert!(registrations
        .iter()
        .all(|registration| registration.id != vyre_driver_metal::METAL_BACKEND_ID));
}

/// WHY: native Metal materialization must execute the authenticated structured MSL artifact.
#[cfg(any(target_os = "macos", target_os = "ios"))]
#[test]
fn registered_materializer_executes_authenticated_msl() {
    let registration = vyre_driver::registered_backends()
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
