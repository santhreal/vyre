//! Choke-point contract for shared target-payload admission.
//!
//! WHY: every concrete driver used to open its payload with its own copy of
//! these checks, and the copies drifted. Two backends demanded an entry point
//! named `main` and two did not; one spelled shared rejections in different
//! text. A payload one backend refused, another ran. `vyre_driver::materialize`
//! is now the single admission decision, so this pins that decision against
//! real compiled payloads, one case per rejection branch.
//!
//! This exercises the neutral choke point through a backend that can produce a
//! genuine payload; it does not test anything SPIR-V specific. The companion
//! structural contract proves no backend has grown a private copy again.
//!
//! Not covered here: native module loading, which needs a device.

use std::collections::BTreeMap;

use vyre_driver::materialize::{self, MaterializerTarget};
use vyre_driver::BackendError;
use vyre_foundation::ir::{
    BufferAccess, BufferDecl, DataType, Expr, GraphOutput, Node, Program, ProgramGraph, ShapeDim,
    ValueContract, ValueLifetime,
};
use vyre_megakernel::{
    Artifact, CompileRequest, Digest, ExternalFacts, SearchBudget, TargetEntryPoint,
    TargetModuleBundle, TargetPayload, TargetPayloadFormat, TargetProfile,
};

fn program() -> Program {
    Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [64, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(1))],
    )
}

fn artifact_with_configuration(configuration: u8) -> Artifact {
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
        .expect("graph node must register");
    let request = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([configuration; 32]), BTreeMap::new()),
        SearchBudget::new(1, 1, 0, 0, 1),
        1_000_000,
    )
    .validate()
    .expect("compile request must validate");
    vyre_megakernel::compile(&request).expect("artifact must compile")
}

/// A real artifact and the real payload a target compiler produced for it.
fn compiled() -> (Artifact, TargetPayload) {
    let registration = vyre_driver::backend::registered_backends()
        .expect("valid backend registry")
        .iter()
        .find(|registration| registration.id == vyre_driver_spirv::SPIRV_BACKEND_ID)
        .expect("SPIR-V registration must be force-linked")
        .clone();
    let compiler = registration
        .target_compiler()
        .expect("SPIR-V target compiler must be registered");
    let artifact = artifact_with_configuration(0);
    let payload = compiler
        .compile(&artifact)
        .expect("artifact must compile to a target payload");
    (artifact, payload)
}

fn target<'a>(payload: &'a TargetPayload) -> MaterializerTarget<'a> {
    MaterializerTarget {
        backend_id: vyre_driver_spirv::SPIRV_BACKEND_ID,
        format: payload.format(),
        profile: payload.profile(),
    }
}

/// Rebuild a payload around a perturbed bundle and entry list.
fn repack(
    artifact: &Artifact,
    payload: &TargetPayload,
    bundle: &TargetModuleBundle,
    entries: Vec<TargetEntryPoint>,
) -> TargetPayload {
    TargetPayload::new(
        artifact,
        payload.format().clone(),
        payload.profile().clone(),
        entries,
        bundle.to_bytes().expect("bundle must encode"),
    )
    .expect("perturbed payload must still seal")
}

fn bundle_of(payload: &TargetPayload) -> TargetModuleBundle {
    TargetModuleBundle::from_bytes(payload.bytes()).expect("bundle must decode")
}

fn expect_invalid_program(error: BackendError, expected: &str) {
    match error {
        BackendError::InvalidProgram { fix } => assert!(
            fix.contains(expected),
            "rejection must name the cause; wanted `{expected}`, got `{fix}`"
        ),
        other => panic!("expected InvalidProgram, got {other:?}"),
    }
}

/// WHY: the accepting path must hand back one admitted module per selected
/// fusion group, already carrying the decoded Program and the payload's grid.
#[test]
fn admission_pairs_every_selected_group_with_its_program_and_grid() {
    let (artifact, payload) = compiled();
    let admitted =
        materialize::admit(&artifact, &payload, target(&payload)).expect("payload must admit");

    assert_eq!(
        admitted.len(),
        artifact.fusion().len(),
        "one admitted module per compiler-selected fusion group"
    );
    for (module, entry) in admitted.iter().zip(payload.entries()) {
        assert_eq!(module.image.entry_point, entry.name);
        assert_eq!(
            module.config.dispatch_grid,
            Some(entry.grid_size),
            "admission must carry the payload entry grid, not a default"
        );
        assert_eq!(module.config.grid_override, Some(entry.grid_size));
        assert!(
            !module.image.bytes.is_empty(),
            "admitted module must retain its target-native bytes"
        );
    }
}

/// WHY: a payload sealed against a different artifact must never open, or a
/// backend would execute a kernel compiled for another program.
#[test]
fn admission_rejects_a_payload_sealed_for_another_artifact() {
    let (_, payload) = compiled();
    let other = artifact_with_configuration(1);
    let error = materialize::admit(&other, &payload, target(&payload))
        .expect_err("foreign artifact must be rejected");
    expect_invalid_program(error, "not authenticated");
}

/// WHY: format is the one mismatch that is a capability answer, not corruption,
/// so it must stay UnsupportedFeature and name the acquiring backend.
#[test]
fn admission_reports_a_foreign_payload_format_as_unsupported() {
    let (artifact, payload) = compiled();
    let foreign = TargetPayloadFormat::new("not-spv", 1).expect("format must build");
    let mismatched = MaterializerTarget {
        format: &foreign,
        ..target(&payload)
    };
    match materialize::admit(&artifact, &payload, mismatched) {
        Err(BackendError::UnsupportedFeature { name, backend }) => {
            assert!(
                name.contains(payload.format().identity()),
                "rejection must name the offered format, got `{name}`"
            );
            assert_eq!(backend, vyre_driver_spirv::SPIRV_BACKEND_ID);
        }
        other => panic!("expected UnsupportedFeature, got {other:?}"),
    }
}

/// WHY: a payload built for another device profile is not runnable here even
/// though it is authentic for the artifact.
#[test]
fn admission_rejects_a_payload_built_for_another_profile() {
    let (artifact, payload) = compiled();
    let foreign =
        TargetProfile::new("foreign-profile", 1, [64, 1, 1], 64, 0, 32).expect("profile must build");
    let mismatched = MaterializerTarget {
        profile: &foreign,
        ..target(&payload)
    };
    let error = materialize::admit(&artifact, &payload, mismatched)
        .expect_err("foreign profile must be rejected");
    expect_invalid_program(error, "profile does not match");
}

/// WHY: this is the check two of four backends were missing. A module whose
/// entry point is not `main` was executable on half the fleet.
#[test]
fn admission_rejects_a_module_whose_entry_point_is_not_main() {
    let (artifact, payload) = compiled();
    let mut bundle = bundle_of(&payload);
    bundle.modules[0].entry_point = "not_main".to_string();
    let mut entries = payload.entries().to_vec();
    entries[0].name = "not_main".to_string();
    let perturbed = repack(&artifact, &payload, &bundle, entries);

    let error = materialize::admit(&artifact, &perturbed, target(&perturbed))
        .expect_err("non-main entry point must be rejected");
    expect_invalid_program(error, "entry point must be `main`");
}

/// WHY: entry metadata and the emitted module must name the same entry, or the
/// dispatch grid is read from an entry that describes a different kernel.
#[test]
fn admission_rejects_entry_metadata_that_names_another_entry() {
    let (artifact, payload) = compiled();
    let bundle = bundle_of(&payload);
    let mut entries = payload.entries().to_vec();
    entries[0].name = "main_shadow".to_string();
    let perturbed = repack(&artifact, &payload, &bundle, entries);

    let error = materialize::admit(&artifact, &perturbed, target(&perturbed))
        .expect_err("entry name disagreement must be rejected");
    expect_invalid_program(error, "must name the emitted target entry point");
}

/// WHY: the module count is how admission knows the payload implements the whole
/// selected plan and not a subset of it.
#[test]
fn admission_rejects_a_module_count_that_disagrees_with_the_plan() {
    let (artifact, payload) = compiled();
    let bundle = TargetModuleBundle::new(Vec::new());
    let perturbed = repack(&artifact, &payload, &bundle, payload.entries().to_vec());

    let error = materialize::admit(&artifact, &perturbed, target(&perturbed))
        .expect_err("a bundle that implements no selected group must be rejected");
    expect_invalid_program(error, "module count must equal");
}

/// WHY: a corrupt Program wire must fail admission, never reach a backend that
/// would decode garbage into a dispatch. The bundle schema catches this during
/// decode, so it surfaces as an attributed decode failure rather than as the
/// defensive per-module wire check further down `admit`.
#[test]
fn admission_rejects_a_malformed_program_wire() {
    let (artifact, payload) = compiled();
    let mut bundle = bundle_of(&payload);
    bundle.modules[0].program = vec![0xff, 0xff, 0xff, 0xff];
    let perturbed = repack(&artifact, &payload, &bundle, payload.entries().to_vec());

    match materialize::admit(&artifact, &perturbed, target(&perturbed)) {
        Err(BackendError::KernelCompileFailed {
            backend,
            compiler_message,
        }) => {
            assert_eq!(backend, vyre_driver_spirv::SPIRV_BACKEND_ID);
            assert!(
                compiler_message.contains("selected Program is malformed"),
                "rejection must name the malformed wire, got `{compiler_message}`"
            );
        }
        other => panic!("expected KernelCompileFailed, got {other:?}"),
    }
}

/// WHY: bundle corruption is a decode failure, and it must be attributed to the
/// backend that offered the payload rather than reported as a bad program.
#[test]
fn admission_attributes_bundle_corruption_to_the_acquiring_backend() {
    let (artifact, payload) = compiled();
    let mut bytes = payload.bytes().to_vec();
    let last = bytes.len() - 1;
    bytes[last] ^= 0xff;
    let corrupt = TargetPayload::new(
        &artifact,
        payload.format().clone(),
        payload.profile().clone(),
        payload.entries().to_vec(),
        bytes,
    )
    .expect("corrupt bundle must still seal into a payload");

    match materialize::admit(&artifact, &corrupt, target(&corrupt)) {
        Err(BackendError::KernelCompileFailed {
            backend,
            compiler_message,
        }) => {
            assert_eq!(backend, vyre_driver_spirv::SPIRV_BACKEND_ID);
            assert!(
                compiler_message.contains("rebuild the target payload"),
                "decode failure must state the corrective action, got `{compiler_message}`"
            );
        }
        other => panic!("expected KernelCompileFailed, got {other:?}"),
    }
}
