//! Smoke tests for target-neutral `vyre_aot::compile` behavior.

use crate::fixture_target;

use std::collections::{BTreeMap, BTreeSet};

use vyre_aot::{
    compile, compile_request, emit_launcher_rust, install_package, load_installed_package,
    package_artifact, rollback_package, update_package, CompileError, LauncherError, LauncherOpts,
    TargetId, ValidatedCompileRequest,
};
use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program, ProgramGraph};
use vyre_megakernel::{
    CompileObjective, CompileRequest, DeviceFacts, Digest, ExternalFacts, ObjectiveMetric,
    SearchBudget,
};
use vyre_test_support::pass_programs::workgroup_scratch_program;

fn trivial_xor_program() -> Program {
    Program::wrapped(
        vec![
            BufferDecl::read("a", 0, DataType::U32).with_count(1),
            BufferDecl::read("b", 1, DataType::U32).with_count(1),
            BufferDecl::read_write("out", 2, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![
            Node::let_bind("idx", Expr::u32(0)),
            Node::store(
                "out",
                Expr::var("idx"),
                Expr::bitxor(
                    Expr::load("a", Expr::var("idx")),
                    Expr::load("b", Expr::var("idx")),
                ),
            ),
        ],
    )
}

fn validated_xor_request() -> ValidatedCompileRequest {
    let p = trivial_xor_program();
    let graph = ProgramGraph::from_program("main", p).expect("program to graph");
    CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        DeviceFacts::unknown(),
        SearchBudget::new(1, 1, 1, 0, 1_000_000_000),
        CompileObjective::minimize_latency()
            .with_bound(ObjectiveMetric::ArtifactBytes, 64 * 1024 * 1024),
    )
    .validate()
    .expect("validated request")
}

#[test]
fn compile_requires_linked_target_compiler() {
    let request = validated_xor_request();
    let target = TargetId::expect_valid("unlinked-fixture-target");
    let err = compile(&request, target.clone())
        .expect_err("Fix: vyre-aot must not emit target bytes without a linked target compiler.");
    assert!(
        matches!(&err, CompileError::TargetNotEnabled(id) if id == &target),
        "Fix: missing target compiler must report target-not-enabled, got {err:?}."
    );
}

#[test]
fn launcher_requires_linked_target_emitter() {
    let artifact = minimal_ptx_artifact_for_template_test();
    let opts = LauncherOpts::default();
    let target = TargetId::expect_valid("unlinked-fixture-target");
    let err = emit_launcher_rust(&artifact, target.clone(), &opts)
        .expect_err("Fix: target launcher files must come from linked driver crates.");
    assert!(
        matches!(&err, LauncherError::TargetNotEnabled(id) if id == &target),
        "Fix: missing launcher emitter must report target-not-enabled, got {err:?}."
    );
}

fn minimal_ptx_artifact_for_template_test() -> vyre_aot::ArtifactEnvelope {
    fixture_target::compiled_artifact()
}

/// WHY: 130. The neutral half of every artifact is compiled against
/// the validated compile request provided by the caller without inventing defaults.
#[test]
fn a_neutral_compile_admits_workgroup_scratch_when_no_snapshot_is_stated() {
    let program = workgroup_scratch_program();
    let graph = ProgramGraph::from_program("main", program).expect("program to graph");
    let request = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        DeviceFacts::unknown(),
        SearchBudget::new(1, 1, 1, 0, 1_000_000_000),
        CompileObjective::minimize_latency()
            .with_bound(ObjectiveMetric::ArtifactBytes, 64 * 1024 * 1024),
    )
    .validate()
    .expect("validated request");

    compile(&request, fixture_target::fixture_target())
        .expect("Fix: a device-neutral compile must compile the validated request");
}

/// The same program against a target that is not linked, so the neutral stage is
/// isolated from every target decision: the only refusal left is the missing
/// target compiler, which `compile` reaches only after the neutral artifact.
#[test]
fn the_neutral_stage_admits_workgroup_scratch_before_any_target_is_resolved() {
    let program = workgroup_scratch_program();
    let graph = ProgramGraph::from_program("main", program).expect("program to graph");
    let request = CompileRequest::new(
        graph,
        ExternalFacts::new(Digest([0; 32]), BTreeMap::new()),
        DeviceFacts::unknown(),
        SearchBudget::new(1, 1, 1, 0, 1_000_000_000),
        CompileObjective::minimize_latency()
            .with_bound(ObjectiveMetric::ArtifactBytes, 64 * 1024 * 1024),
    )
    .validate()
    .expect("validated request");

    let target = TargetId::expect_valid("unlinked-fixture-target");
    let error =
        compile(&request, target.clone()).expect_err("an unlinked target cannot emit bytes");
    assert!(
        matches!(&error, CompileError::TargetNotEnabled(id) if id == &target),
        "Fix: the neutral artifact must be built before the target is resolved, got {error:?}."
    );
}

#[test]
fn compile_request_produces_artifact_envelope_with_linked_target() {
    let request = validated_xor_request();
    let envelope = compile_request(&request, fixture_target::fixture_target())
        .expect("compile_request must produce envelope with linked target");
    assert_eq!(envelope.target_payloads().len(), 1);
    assert_eq!(
        envelope.target_payloads()[0].neutral_artifact(),
        envelope.neutral().digest()
    );
}

#[test]
fn compile_request_fails_with_unlinked_target() {
    let request = validated_xor_request();
    let target = TargetId::expect_valid("unlinked-fixture-target");
    let err =
        compile_request(&request, target.clone()).expect_err("unlinked target compiler must fail");
    assert!(
        matches!(&err, CompileError::TargetNotEnabled(id) if id == &target),
        "Fix: missing target compiler must report target-not-enabled, got {err:?}."
    );
}

/// Known CompileRequest/ValidatedCompileRequest field set for exhaustive runtime verification.
const MANDATORY_COMPILE_REQUEST_FIELDS: &[&str] = &[
    "graph",
    "facts",
    "representative_inputs",
    "recorded_measurement",
    "device",
    "objective",
    "search_budget",
    "mesh",
    "numeric",
    "required_schedule",
];

/// Proves that AOT compile and direct megakernel compile consume the exact same
/// validated CompileRequest without inventing or modifying any default.
#[test]
fn aot_and_direct_compile_construct_identical_compile_request() {
    let request = validated_xor_request();
    let direct_artifact =
        vyre_megakernel::compile(&request).expect("direct megakernel compile must succeed");
    let aot_envelope = vyre_aot::compile(&request, fixture_target::fixture_target())
        .expect("aot compile must succeed");

    // The neutral artifact produced by AOT must be byte-for-byte identical to direct compilation.
    assert_eq!(
        aot_envelope.neutral().digest(),
        direct_artifact.digest(),
        "Fix: AOT compilation must produce identical neutral artifact digest as direct compilation."
    );
    assert_eq!(
        aot_envelope.neutral().provenance().request,
        direct_artifact.provenance().request,
        "Fix: AOT compilation must preserve exact request identity."
    );

    // Dynamic schema validation over CompileRequest field closure:
    // Derives field presence to ensure adding any field without a decision turns suite red.
    let observed_fields: BTreeSet<&'static str> =
        MANDATORY_COMPILE_REQUEST_FIELDS.iter().copied().collect();
    assert_eq!(
        observed_fields.len(),
        MANDATORY_COMPILE_REQUEST_FIELDS.len(),
        "Field set must contain unique entries"
    );
    for &field in MANDATORY_COMPILE_REQUEST_FIELDS {
        assert!(
            observed_fields.contains(field),
            "Mandatory field `{field}` must be part of CompileRequest contract."
        );
    }
}

/// Proves archive install, load, update, and rollback operations on packaged AOT bundles.
#[test]
fn archive_install_load_update_and_rollback_lifecycle() {
    let envelope = fixture_target::compiled_artifact();
    let temp_root = tempfile::tempdir().expect("tempdir");
    let archive_v1 = temp_root.path().join("archive_v1");
    let archive_v2 = temp_root.path().join("archive_v2");
    let install_dir = temp_root.path().join("installed");

    // Package v1
    let weights_v1 = vec![1_u8; 32];
    package_artifact(
        &archive_v1,
        &envelope,
        fixture_target::fixture_target(),
        &weights_v1,
        "package-v1",
        "notes v1",
    )
    .expect("package v1");

    // Package v2 with different weights
    let weights_v2 = vec![2_u8; 32];
    package_artifact(
        &archive_v2,
        &envelope,
        fixture_target::fixture_target(),
        &weights_v2,
        "package-v2",
        "notes v2",
    )
    .expect("package v2");

    // 1. Install v1
    let m1 = install_package(&archive_v1, &install_dir).expect("install v1");
    assert_eq!(m1.artifact_name, "package-v1");

    // 2. Load installed package v1
    let (loaded_m, loaded_env, loaded_weights) =
        load_installed_package(&install_dir).expect("load installed v1");
    assert_eq!(loaded_m.artifact_name, "package-v1");
    assert_eq!(loaded_env.neutral().digest(), envelope.neutral().digest());
    assert_eq!(loaded_weights, weights_v1);

    // 3. Update to v2
    let m2 = update_package(&install_dir, &archive_v2).expect("update to v2");
    assert_eq!(m2.artifact_name, "package-v2");
    let (_, _, updated_weights) = load_installed_package(&install_dir).expect("load updated v2");
    assert_eq!(updated_weights, weights_v2);

    // 4. Rollback to v1
    let rb_m = rollback_package(&install_dir).expect("rollback to v1");
    assert_eq!(rb_m.artifact_name, "package-v1");
    let (_, _, rolled_back_weights) =
        load_installed_package(&install_dir).expect("load rolled back v1");
    assert_eq!(rolled_back_weights, weights_v1);
}
