//! Launcher source emission contract tests.

mod common;

use vyre_aot::{emit_launcher_rust, ArtifactEnvelope, LauncherError, LauncherOpts, Target};

fn minimal_ptx_artifact() -> ArtifactEnvelope {
    common::compiled_artifact()
}

#[test]
fn launcher_requires_linked_target_emitter() {
    let artifact = minimal_ptx_artifact();
    let opts = LauncherOpts::default();
    let err = emit_launcher_rust(&artifact, Target::Ptx, &opts).expect_err(
        "Fix: vyre-aot must not synthesize target-owned launcher files without a linked driver.",
    );
    assert!(
        matches!(err, LauncherError::TargetNotEnabled("secondary_text")),
        "Fix: missing launcher emitter must report target-not-enabled, got {err:?}."
    );
}

#[test]
fn launcher_options_are_target_neutral() {
    let opts = LauncherOpts {
        crate_name: "custom-launcher".to_string(),
        include_collectives: false,
        include_ttt_loop: true,
    };
    assert_eq!(opts.crate_name, "custom-launcher");
    assert!(!opts.include_collectives);
    assert!(opts.include_ttt_loop);
}
