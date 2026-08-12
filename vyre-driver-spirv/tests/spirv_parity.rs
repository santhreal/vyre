//! SPIR-V driver contracts through canonical verified lowering and emission.

use std::collections::BTreeMap;
use std::io::Write;
use std::process::Command;

use tempfile::NamedTempFile;
use vyre_driver::BindingSet;
use vyre_driver_spirv::SpirvBackend;
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, GraphOutput, Node, Program, ProgramGraph, ShapeDim,
    ValueContract, ValueLifetime,
};
use vyre_megakernel::{
    CompileRequest, Digest, ExternalFacts, SearchBudget, TargetModuleBundle, TargetPayload,
    TargetPayloadFormat,
};

fn assert_spirv_structural_invariants(label: &str, words: &[u32]) {
    assert!(
        !words.is_empty(),
        "Fix: {label} emitted an empty SPIR-V blob"
    );
    assert_eq!(
        words[0], 0x0723_0203,
        "Fix: {label} emitted a SPIR-V blob without the SPIR-V magic header"
    );

    if Command::new("spirv-val").arg("--version").output().is_ok() {
        let mut file = NamedTempFile::new()
            .unwrap_or_else(|error| panic!("Fix: create temp SPIR-V file for {label}: {error}"));
        for word in words {
            file.write_all(&word.to_le_bytes())
                .unwrap_or_else(|error| panic!("Fix: write SPIR-V bytes for {label}: {error}"));
        }
        let output = Command::new("spirv-val")
            .arg(file.path())
            .output()
            .unwrap_or_else(|error| panic!("Fix: launch spirv-val for {label}: {error}"));
        assert!(
            output.status.success(),
            "Fix: spirv-val rejected {label}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    } else {
        assert!(
            words.len() >= 5,
            "Fix: {label} emitted a truncated SPIR-V header fallback snapshot"
        );
        assert!(
            words[1] >= 0x0001_0000,
            "Fix: {label} emitted an invalid SPIR-V version word in fallback validation"
        );
    }
}

fn program() -> Program {
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [64, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
    )
}

fn artifact_with_configuration(configuration: u8) -> vyre_megakernel::Artifact {
    let mut graph = ProgramGraph::new();
    graph
        .add_node(
            "main",
            program(),
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
        ExternalFacts::new(Digest([configuration; 32]), BTreeMap::new()),
        SearchBudget::new(1, 1, 0, 0, 1),
        1_000_000,
    )
    .validate()
    .unwrap();
    vyre_megakernel::compile(&request).unwrap()
}

fn artifact() -> vyre_megakernel::Artifact {
    artifact_with_configuration(0)
}

#[test]
fn program_compilation_uses_one_deterministic_spirv_writer() {
    let first = SpirvBackend::program_to_spv(&program())
        .expect("Fix: canonical Program must compile to SPIR-V");
    let second = SpirvBackend::program_to_spv(&program())
        .expect("Fix: identical Program must compile to SPIR-V");
    assert_eq!(
        first, second,
        "Fix: identical Program must emit identical SPIR-V"
    );
    assert_spirv_structural_invariants("canonical_program", &first);
}

/// WHY: inventory discovery must compile immutable selected modules, never caller Programs.
#[test]
fn registered_target_compiler_emits_spirv_module_bundle() {
    let registration = vyre_driver::backend::registered_backends()
        .expect("valid backend registry")
        .iter()
        .find(|registration| registration.id == vyre_driver_spirv::SPIRV_BACKEND_ID)
        .expect("SPIR-V registration must be force-linked");
    let compiler = registration
        .target_compiler()
        .expect("SPIR-V target compiler must be registered");
    let artifact = artifact();
    let payload = compiler.compile(&artifact).expect("artifact must compile");
    let bundle = TargetModuleBundle::from_bytes(payload.bytes()).expect("bundle must decode");
    assert_eq!(bundle.modules.len(), 1);
    assert_eq!(bundle.modules[0].entry_point, "main");
    assert_eq!(payload.entries()[0].name, "main");
    assert_eq!(
        &bundle.modules[0].bytes[..4],
        &0x0723_0203_u32.to_le_bytes()
    );
    assert_eq!(payload.neutral_artifact(), artifact.digest());
}

/// WHY: materialization and submission must execute authenticated payload bytes without recompiling.
#[test]
fn registered_materializer_executes_artifact_instance() {
    let registration = vyre_driver::backend::registered_backends()
        .expect("valid backend registry")
        .iter()
        .find(|registration| registration.id == vyre_driver_spirv::SPIRV_BACKEND_ID)
        .expect("SPIR-V registration must be force-linked");
    let compiler = registration.target_compiler().unwrap();
    let materializer = registration
        .materializer()
        .expect("Vulkan device materializer must acquire on the GPU-required host");
    let artifact = artifact();
    let payload = compiler.compile(&artifact).unwrap();
    let instance = materializer.materialize(&artifact, &payload).unwrap();
    let wrong_artifact = artifact_with_configuration(1);
    assert!(
        materializer.materialize(&wrong_artifact, &payload).is_err(),
        "payload association mismatch must fail before native materialization"
    );
    let wrong_format = TargetPayload::new(
        &artifact,
        TargetPayloadFormat::new("not-spv", 1).unwrap(),
        payload.profile().clone(),
        payload.entries().to_vec(),
        payload.bytes().to_vec(),
    )
    .unwrap();
    assert!(
        materializer.materialize(&artifact, &wrong_format).is_err(),
        "payload format mismatch must fail before native materialization"
    );
    let malformed = TargetPayload::new(
        &artifact,
        payload.format().clone(),
        payload.profile().clone(),
        payload.entries().to_vec(),
        vec![1],
    )
    .unwrap();
    assert!(
        materializer.materialize(&artifact, &malformed).is_err(),
        "malformed module bytes must fail before native materialization"
    );
    let mut invalid_spirv_bundle = TargetModuleBundle::from_bytes(payload.bytes()).unwrap();
    invalid_spirv_bundle.modules[0].bytes[..4].copy_from_slice(&0_u32.to_le_bytes());
    let invalid_spirv = TargetPayload::new(
        &artifact,
        payload.format().clone(),
        payload.profile().clone(),
        payload.entries().to_vec(),
        invalid_spirv_bundle.to_bytes().unwrap(),
    )
    .unwrap();
    assert!(
        materializer.materialize(&artifact, &invalid_spirv).is_err(),
        "invalid SPIR-V magic must fail before native materialization"
    );
    let empty_bundle = TargetModuleBundle::new(Vec::new());
    let missing_module = TargetPayload::new(
        &artifact,
        payload.format().clone(),
        payload.profile().clone(),
        payload.entries().to_vec(),
        empty_bundle.to_bytes().unwrap(),
    )
    .unwrap();
    assert!(
        materializer
            .materialize(&artifact, &missing_module)
            .is_err(),
        "module count mismatch must fail before native materialization"
    );
    let mut wrong_bundle = TargetModuleBundle::from_bytes(payload.bytes()).unwrap();
    wrong_bundle.modules[0].group.0 += 1;
    let wrong_group = TargetPayload::new(
        &artifact,
        payload.format().clone(),
        payload.profile().clone(),
        payload.entries().to_vec(),
        wrong_bundle.to_bytes().unwrap(),
    )
    .unwrap();
    assert!(
        materializer.materialize(&artifact, &wrong_group).is_err(),
        "selected group mismatch must fail before native materialization"
    );
    assert!(
        instance
            .submit(BindingSet::new(wrong_artifact.digest()))
            .is_err(),
        "binding association mismatch must fail before submission"
    );
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

#[test]
fn invalid_program_fails_before_spirv_emission() {
    let invalid = Program::wrapped(Vec::new(), [0, 1, 1], Vec::new());
    let error = SpirvBackend::program_to_spv(&invalid)
        .expect_err("Fix: invalid workgroup geometry must fail verified lowering");
    assert!(error.contains("verified lowering failed"));
    assert!(error.contains("Fix:"));
}
