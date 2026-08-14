//! Public feature-boundary regressions for self-substrate solver families.

#![forbid(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn fixture_root() -> PathBuf {
    std::env::temp_dir().join(format!(
        "vyre-self-substrate-feature-boundaries-{}",
        std::process::id()
    ))
}

fn write_fixture(root: &Path, source: &str) {
    let _ = fs::remove_dir_all(root);
    fs::create_dir_all(root.join("src")).expect("Fix: feature fixture source must be creatable");
    let substrate = Path::new(env!("CARGO_MANIFEST_DIR"));
    fs::write(
        root.join("Cargo.toml"),
        format!(
            "[package]\nname = \"self-substrate-feature-fixture\"\nversion = \"0.0.0\"\nedition = \"2021\"\n\n[dependencies]\nvyre-self-substrate = {{ path = {:?}, default-features = false, features = [\"optimizer\"] }}\n",
            substrate
        ),
    )
    .expect("Fix: feature fixture manifest must be writable");
    fs::write(root.join("src/main.rs"), source)
        .expect("Fix: feature fixture source must be writable");
}

fn cargo(root: &Path, arguments: &[&str]) -> Output {
    Command::new("cargo")
        .args(arguments)
        .env("CARGO_BUILD_JOBS", "1")
        .env("CARGO_TARGET_DIR", root.join("target"))
        .current_dir(root)
        .output()
        .expect("Fix: Cargo must execute for the feature boundary fixture")
}

/// The optimizer-only contract must compile without exposing solver families.
///
/// This guards both sides of the boundary: the canonical optimizer namespace is
/// usable, while an accidental import from the math solver family is rejected.
/// The resolved primitive graph must also omit unrelated heavyweight domains.
#[test]
fn optimizer_feature_is_standalone_and_excludes_unrequested_solver_domains() {
    let root = fixture_root();
    write_fixture(
        &root,
        "fn accepts(_: &dyn vyre_foundation::program_dispatch::ProgramDispatcher) {}\nfn main() {}\n",
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
        "use vyre_self_substrate::math::tensor_train_compression;\nfn main() { let _ = tensor_train_compression::TT_MAX_RANK; }\n",
    );
    let rejected = cargo(&root, &["check", "--quiet"]);
    assert!(
        !rejected.status.success(),
        "optimizer-only consumer must not see the math solver family"
    );
    let stderr = String::from_utf8_lossy(&rejected.stderr);
    assert!(
        stderr.contains("could not find `math` in `vyre_self_substrate`"),
        "rejection must identify the unavailable public family:\n{stderr}"
    );

    fs::remove_dir_all(&root).expect("Fix: feature fixture must be removable");
}
