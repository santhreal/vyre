//! SPIR-V driver contracts through canonical verified lowering and emission.
//!
//! The registry, payload-format and authenticated-execution statements are the
//! shared contract in `tests/support/target_compiler_contract.rs`. What stays
//! here is SPIR-V specific: structural validation of emitted words through
//! `spirv-val`, determinism of the SPIR-V writer, and the perturbation cases that
//! prove this materializer refuses a payload before it touches Vulkan.

use std::io::Write;
use std::process::Command;

use tempfile::NamedTempFile;
use vyre_driver::BindingSet;
use vyre_driver_spirv::SpirvBackend;
use vyre_foundation::ir::Program;
use vyre_megakernel::{TargetModuleBundle, TargetPayload, TargetPayloadFormat};

mod support;
use support::target_compiler_contract::{
    assert_materializer_executes_payload, assert_target_compiler_emits_bundle, registration,
    store_one_program,
};
use support::{artifact, foreign_artifact, spirv};

/// First word of every well-formed SPIR-V module.
const SPIRV_MAGIC: u32 = 0x0723_0203;

fn assert_spirv_structural_invariants(label: &str, words: &[u32]) {
    assert!(
        !words.is_empty(),
        "Fix: {label} emitted an empty SPIR-V blob"
    );
    assert_eq!(
        words[0], SPIRV_MAGIC,
        "Fix: {label} emitted a SPIR-V blob without the SPIR-V magic header"
    );

    // spirv-val is the only thing here that can tell valid SPIR-V from a
    // plausible header. When it was absent this function asserted the blob held
    // at least five words and carried a version word, then returned, so every
    // emission passed on a machine without the validator and the gate built on
    // this function proved nothing there. A missing validator is a
    // configuration failure, not a clean tree.
    let probe = Command::new("spirv-val")
        .arg("--version")
        .output()
        .unwrap_or_else(|error| {
            panic!(
                "Fix: install spirv-tools so spirv-val is on PATH. SPIR-V emission cannot be \
                 validated without it, and passing {label} without validating it is the defect \
                 this check exists to prevent ({error})"
            )
        });
    assert!(
        probe.status.success(),
        "Fix: spirv-val is on PATH but did not report a version, so it cannot validate {label}: {}",
        String::from_utf8_lossy(&probe.stderr)
    );

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
}

#[test]
fn program_compilation_uses_one_deterministic_spirv_writer() {
    let first = SpirvBackend::program_to_spv(&store_one_program())
        .expect("Fix: canonical Program must compile to SPIR-V");
    let second = SpirvBackend::program_to_spv(&store_one_program())
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
    assert_target_compiler_emits_bundle(&spirv(), |bundle| {
        assert_eq!(
            &bundle.modules[0].bytes[..4],
            &SPIRV_MAGIC.to_le_bytes(),
            "Fix: SPIR-V target module must begin with the SPIR-V magic word"
        );
    });
}

/// WHY: materialization and submission must execute authenticated payload bytes
/// without recompiling.
#[test]
fn registered_materializer_executes_artifact_instance() {
    assert_materializer_executes_payload(&spirv());
}

/// WHY: every perturbation of an authentic payload must be refused before the
/// materializer touches Vulkan. This backend reaches the shared admission choke
/// point, so a case that passes here would pass for a payload no artifact digest
/// covers.
#[test]
fn perturbed_payloads_fail_before_native_materialization() {
    let registration = registration(vyre_driver_spirv::SPIRV_BACKEND_ID);
    let compiler = registration.target_compiler().unwrap();
    let materializer = registration
        .materializer()
        .expect("Vulkan device materializer must acquire on the GPU-required host");
    let artifact = artifact();
    let payload = compiler.compile(&artifact).unwrap();
    let instance = materializer.materialize(&artifact, &payload).unwrap();
    let wrong_artifact = foreign_artifact();
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
}

#[test]
fn invalid_program_fails_before_spirv_emission() {
    let invalid = Program::wrapped(Vec::new(), [0, 1, 1], Vec::new());
    let error = SpirvBackend::program_to_spv(&invalid)
        .expect_err("Fix: invalid workgroup geometry must fail verified lowering");
    assert!(error.contains("verified lowering failed"));
    assert!(error.contains("Fix:"));
}
