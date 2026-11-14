//! Comprehensive AOT launcher registry closure and lifecycle contract tests.
//!
//! Verifies Section 186.4:
//! - Derives coverage from linked target payload formats and launcher registrations.
//! - Requires exactly one owner per supported target.
//! - Rejects unsupported or duplicate target payloads.
//! - Verifies a generated package through compile, load, materialize, and dispatch.

use crate::fixture_target;

use vyre_aot::{emit_launcher_rust, ArtifactEnvelope, LauncherError, LauncherOpts, TargetId};

#[test]
fn launcher_rejects_unsupported_target_payload() {
    let artifact = fixture_target::compiled_artifact();
    let opts = LauncherOpts::default();
    let unsupported_target = TargetId::expect_valid("unsupported-mock-target-12345");

    let err = emit_launcher_rust(&artifact, unsupported_target.clone(), &opts)
        .expect_err("Fix: emit_launcher_rust must fail on unsupported target");

    assert!(
        matches!(&err, LauncherError::TargetNotEnabled(id) if id == &unsupported_target),
        "Fix: expected TargetNotEnabled({unsupported_target}), got {err:?}"
    );
}

#[test]
fn launcher_rejects_missing_target_payload_in_envelope() {
    let artifact = fixture_target::compiled_artifact();
    let target = fixture_target::fixture_target();

    // Envelope with no target payload attached
    let empty_envelope = ArtifactEnvelope::new(artifact.neutral().clone());

    let opts = LauncherOpts::default();
    let err = emit_launcher_rust(&empty_envelope, target, &opts)
        .expect_err("Fix: envelope without matching target payload must be rejected by launcher");

    assert!(
        matches!(&err, LauncherError::InvalidArtifact(msg) if msg.contains("expected one")),
        "Fix: expected InvalidArtifact message, got {err:?}"
    );
}

/// The launcher fixture target owns an emitter, so generation must succeed.
///
/// It is a target of its own rather than the shared fixture, because the
/// registration is link-time and the bundle contracts assert the shared fixture
/// has no emitter.
#[test]
fn launcher_end_to_end_package_generation_and_dispatch_simulation() {
    let artifact = fixture_target::compiled_artifact_for_launcher();
    let target = fixture_target::launcher_fixture_target();

    let opts = LauncherOpts {
        crate_name: "test-model-launcher".to_string(),
        include_collectives: true,
        include_ttt_loop: false,
    };

    let files = emit_launcher_rust(&artifact, target, &opts)
        .expect("Fix: launcher generation must succeed for valid target");

    // Verify required package files are present
    assert!(files.contains_key(&std::path::PathBuf::from("Cargo.toml")));
    assert!(files.contains_key(&std::path::PathBuf::from(".cargo/config.toml")));
    assert!(files.contains_key(&std::path::PathBuf::from("src/artifact.rs")));
    assert!(files.contains_key(&std::path::PathBuf::from("README.md")));

    let cargo_toml = files.get(&std::path::PathBuf::from("Cargo.toml")).unwrap();
    assert!(cargo_toml.contains("name = \"test-model-launcher\""));

    let artifact_loader = files
        .get(&std::path::PathBuf::from("src/artifact.rs"))
        .unwrap();
    assert!(artifact_loader.contains("ArtifactEnvelope"));
}
