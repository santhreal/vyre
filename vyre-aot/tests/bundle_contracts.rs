//! Bundle packaging contract tests for canonical artifact envelopes.

mod common;

use vyre_aot::{bundle, package_artifact, BundleError, LauncherError, LauncherOpts};

/// Existing launcher behavior remains target-owned after the artifact schema cutover.
#[test]
fn bundle_requires_linked_launcher_emitter() {
    let directory = tempfile::tempdir().expect("temporary directory must exist");
    let artifact = common::compiled_artifact();
    let error = bundle(
        directory.path(),
        &artifact,
        &[42; 128],
        "test-bundle",
        &LauncherOpts::default(),
        "test notes",
    )
    .expect_err("bundle must not synthesize an unregistered target launcher");

    assert!(matches!(
        error,
        BundleError::Launcher(LauncherError::TargetNotEnabled("secondary_text"))
    ));
}

/// Launcher admission must still happen before any package files are written.
#[test]
fn bundle_does_not_write_partial_artifacts_without_launcher() {
    let directory = tempfile::tempdir().expect("temporary directory must exist");
    let artifact = common::compiled_artifact();
    let error = bundle(
        directory.path(),
        &artifact,
        &[0; 16],
        "launcher-test",
        &LauncherOpts::default(),
        "",
    )
    .expect_err("missing launcher must fail before package writes");

    assert!(matches!(error, BundleError::Launcher(_)));
    assert!(!directory.path().join("manifest.json").exists());
    assert!(!directory.path().join("artifact.vmk.lzma").exists());
    assert!(!directory.path().join("weights.brotli").exists());
}

/// Canonical resource byte counts remain the authority for package weight admission.
#[test]
fn package_rejects_weight_payload_larger_than_first_canonical_resource() {
    let parent = tempfile::tempdir().expect("temporary directory must exist");
    let output = parent.path().join("oversized");
    let artifact = common::compiled_artifact();
    let error = package_artifact(&output, &artifact, &[0; 2048], "oversized", "")
        .expect_err("oversized weights must fail before package creation");

    assert!(matches!(
        &error,
        BundleError::InvalidArtifact(message) if message.contains("weights payload has")
    ));
    assert!(!output.exists());
}

/// A valid canonical package writes only envelope, weights, and the packaging manifest.
#[test]
fn package_writes_the_canonical_artifact_files() {
    let directory = tempfile::tempdir().expect("temporary directory must exist");
    let artifact = common::compiled_artifact();
    let package = package_artifact(
        directory.path(),
        &artifact,
        &[0; 128],
        "canonical",
        "",
    )
    .expect("valid canonical package must write");

    assert_eq!(package.files.len(), 3);
    assert!(directory.path().join("manifest.json").exists());
    assert!(directory.path().join("artifact.vmk.lzma").exists());
    assert!(directory.path().join("weights.brotli").exists());
}
