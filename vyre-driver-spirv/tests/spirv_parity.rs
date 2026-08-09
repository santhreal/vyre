//! SPIR-V driver contracts through canonical verified lowering and emission.

use std::collections::BTreeMap;
use std::io::Write;
use std::process::Command;

use tempfile::NamedTempFile;
use vyre_driver_spirv::SpirvBackend;
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, Node, Program, ProgramGraph, ShapeDim, ValueContract,
    ValueLifetime,
};
use vyre_megakernel::{CompileRequest, Digest, ExternalFacts, SearchBudget, TargetModuleBundle};

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
        vec![BufferDecl::storage(
            "out",
            0,
            BufferAccess::ReadWrite,
            DataType::U32,
        )],
        [64, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
    )
}

fn artifact() -> vyre_megakernel::Artifact {
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
        .add_node("main", program(), Vec::new(), Vec::new())
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
    assert_eq!(
        &bundle.modules[0].bytes[..4],
        &0x0723_0203_u32.to_le_bytes()
    );
    assert_eq!(payload.neutral_artifact(), artifact.digest());
}

#[test]
fn invalid_program_fails_before_spirv_emission() {
    let invalid = Program::wrapped(Vec::new(), [0, 1, 1], Vec::new());
    let error = SpirvBackend::program_to_spv(&invalid)
        .expect_err("Fix: invalid workgroup geometry must fail verified lowering");
    assert!(error.contains("verified lowering failed"));
    assert!(error.contains("Fix:"));
}
