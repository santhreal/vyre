//! Public feature-boundary regressions for self-substrate solver families.

#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// Directory of the scratch crate this test compiles against the substrate.
///
/// It lives in the cargo build directory, not in `std::env::temp_dir()`: the
/// fixture compiles the substrate and the foundation, a temp filesystem is
/// capped, and one such fixture filling it fails every other build on the host.
fn fixture_root() -> PathBuf {
    vyre_test_support::monorepo::cargo_target_directory().join("feature-boundary-fixture")
}

/// Write the fixture manifest and the consumer source, keeping earlier builds.
///
/// The directory is not cleared between the two consumers this test compiles.
/// Clearing it also deleted the fixture's own build directory, so each consumer
/// rebuilt the whole substrate graph to answer a question about one import. The
/// manifest and the single source file are the only files written, so
/// overwriting them leaves nothing stale behind.
fn write_fixture(root: &Path, source: &str) {
    fs::create_dir_all(root.join("src")).expect("Fix: feature fixture source must be creatable");
    let workspace = vyre_test_support::monorepo::vyre_workspace_root();
    let substrate = workspace.join("vyre-pass-engine");
    let foundation = workspace.join("vyre-foundation");
    fs::write(
        root.join("Cargo.toml"),
        format!(
            "[workspace]\n\n[package]\nname = \"self-substrate-feature-fixture\"\nversion = \"0.0.0\"\nedition = \"2021\"\n\n[dependencies]\nvyre-pass-engine = {{ path = {:?}, default-features = false, features = [\"optimizer\"] }}\nvyre-foundation = {{ path = {:?} }}\n",
            substrate, foundation
        ),
    )
    .expect("Fix: feature fixture manifest must be writable");
    fs::write(root.join("src/main.rs"), source)
        .expect("Fix: feature fixture source must be writable");
}

/// Run cargo in the fixture, under the fixture's own build configuration.
///
/// No build-affecting variable is set here. Job count and build directory are
/// declared once per checkout, in cargo configuration, and a fixture that
/// exported its own made every gate that ran it build a different build.
fn cargo(root: &Path, arguments: &[&str]) -> Output {
    Command::new("cargo")
        .args(arguments)
        .current_dir(root)
        .output()
        .expect("Fix: Cargo must execute for the feature boundary fixture")
}

/// The optimizer-only contract must compile without exposing solver families.
///
/// This guards both sides of the boundary: the canonical optimizer namespace is
/// usable, while an accidental import from the solver family that now lives in
/// `vyre-libs` is rejected: this crate does not re-export it at any feature.
/// The resolved primitive graph must also omit unrelated heavyweight domains.
#[test]
fn optimizer_feature_is_standalone_and_excludes_unrequested_solver_domains() {
    let root = fixture_root();
    write_fixture(
        &root,
        "fn accepts(_: &dyn vyre_foundation::program_dispatch::ProgramDispatcher) {}\nfn main() { let _ = vyre_pass_engine::optimizer::dce_program::OP_ID; }\n",
    );
    let success = cargo(&root, &["check", "--quiet"]);
    assert!(
        success.status.success(),
        "optimizer-only consumer must compile:\n{}",
        String::from_utf8_lossy(&success.stderr)
    );

    let tree = cargo(
        &root,
        &["tree", "--edges", "features", "--invert", "vyre-primitives"],
    );
    assert!(tree.status.success(), "feature tree must resolve");
    let tree = String::from_utf8(tree.stdout).expect("Fix: Cargo feature tree must be UTF-8");
    for excluded in [
        "feature \"nn\"",
        "feature \"parsing\"",
        "feature \"topology\"",
    ] {
        assert!(
            !tree.contains(excluded),
            "optimizer-only dependency graph must exclude {excluded}:\n{tree}"
        );
    }

    write_fixture(
        &root,
        "use vyre_pass_engine::solvers::tensor_train_compression;\nfn main() { let _ = tensor_train_compression::TT_MAX_RANK; }\n",
    );
    let rejected = cargo(&root, &["check", "--quiet"]);
    assert!(
        !rejected.status.success(),
        "optimizer-only consumer must not see the solver family"
    );
    let stderr = String::from_utf8_lossy(&rejected.stderr);
    assert!(
        stderr.contains("could not find `solvers` in `vyre_pass_engine`"),
        "rejection must identify the unavailable public family:\n{stderr}"
    );

    // The fixture stays: it is a build tree inside the build directory, and the
    // next run of this test reuses the substrate it already compiled. `cargo
    // clean` removes it with everything else cargo wrote.
}
