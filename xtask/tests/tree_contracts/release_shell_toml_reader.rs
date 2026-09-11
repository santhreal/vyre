//! `scripts/lib/toml_reader.sh` read against a manifest with known values.
//!
//! The release launchers resolve the version, both product tags and the two
//! repository names through this reader, and every one of them ends up in a tag
//! that gets pushed or a repository that gets published to. The reader was
//! python and shelled into `tomllib`, which arrived in 3.11: on a 3.9 host the
//! import failed, the loader returned nothing, and the caller carried on with an
//! unset tag. The reader is bash now, so what it accepts is a subset of TOML
//! rather than a parser, and the subset is what these cases pin.
//!
//! Each case runs the real script through `bash`, because a reimplementation of
//! its parse in Rust would pass while the script fails. What is not covered: a
//! basic string carrying an escape, a multi-line string, and a dotted key
//! written inside an inline table are refused rather than read, and the
//! refusals below are the proof of that rather than of any value.

use std::fs;
use std::path::Path;
use std::process::Command;

use crate::workspace_sources::workspace_root;

/// A manifest exercising every shape the release manifests use, plus the shapes
/// the reader must refuse.
const MANIFEST: &str = r#"
# A comment above a top-level key.
bare = "top"
number = 7
flag = true
listed = ["a", "b"]

[versions]
vyre = "0.8.0"   # trailing comment
quoted = 'single'

[release_groups.vyre]
repository = "santhreal/vyre"

[[external_actions]]
id = "publish-crates"
"#;

/// Run the reader for one key and return its exit code and stdout.
fn read(root: &Path, manifest: &Path, key: &str) -> (i32, String) {
    let script = format!(
        "source scripts/lib/toml_reader.sh; vyre_read_toml_values '{}' fixture 1 '{key}'; \
         status=$?; printf '%s\\n' \"${{VYRE_TOML_VALUES[@]}}\"; exit $status",
        manifest.display()
    );
    let output = Command::new("bash")
        .arg("-c")
        .arg(script)
        .current_dir(root)
        .output()
        .expect("Fix: bash must be available to run the release TOML reader.");
    let code = output.status.code().unwrap_or(-1);
    let stdout = String::from_utf8(output.stdout).expect("Fix: reader output must be UTF-8.");
    (code, stdout.trim_end().to_string())
}

/// The manifest written into a temporary directory beside the checkout.
fn fixture() -> (tempfile::TempDir, std::path::PathBuf) {
    let directory = tempfile::tempdir().expect("Fix: a temporary directory must be creatable.");
    let path = directory.path().join("manifest.toml");
    fs::write(&path, MANIFEST).expect("Fix: the fixture manifest must be writable.");
    (directory, path)
}

#[test]
fn every_scalar_shape_the_release_manifests_use_is_read_whole() {
    let root = workspace_root();
    let (_directory, manifest) = fixture();
    for (key, expected) in [
        ("bare", "top"),
        ("number", "7"),
        ("flag", "true"),
        ("versions.vyre", "0.8.0"),
        ("versions.quoted", "single"),
        ("release_groups.vyre.repository", "santhreal/vyre"),
    ] {
        let (code, value) = read(&root, &manifest, key);
        assert_eq!(code, 0, "reading {key} failed: {value}");
        assert_eq!(value, expected, "{key} read the wrong value");
    }
}

#[test]
fn a_key_the_reader_cannot_resolve_to_one_scalar_is_refused() {
    let root = workspace_root();
    let (_directory, manifest) = fixture();
    for key in [
        // Absent entirely.
        "versions.absent",
        // Present as an array, which is not one value.
        "listed",
        // A table header is not a value.
        "versions",
        // An array-of-tables entry is not addressable by a dotted key, and its
        // keys must not leak into the top level either.
        "external_actions.id",
        "id",
        // A key that exists under one table must not be found under another.
        "release_groups.vyre.vyre",
    ] {
        let (code, value) = read(&root, &manifest, key);
        assert_eq!(
            code, 2,
            "reading {key} was expected to be refused, got {value}"
        );
    }
}

#[test]
fn the_caller_is_held_to_the_key_count_it_declared() {
    let root = workspace_root();
    let (_directory, manifest) = fixture();
    let script = format!(
        "source scripts/lib/toml_reader.sh; vyre_read_toml_values '{}' fixture 2 bare",
        manifest.display()
    );
    let output = Command::new("bash")
        .arg("-c")
        .arg(script)
        .current_dir(&root)
        .output()
        .expect("Fix: bash must be available to run the release TOML reader.");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("expected 2"), "got {stderr}");
}

#[test]
fn a_manifest_that_is_not_there_is_named_rather_than_read_as_empty() {
    let root = workspace_root();
    let (directory, _manifest) = fixture();
    let absent = directory.path().join("absent.toml");
    let (code, _value) = read(&root, &absent, "bare");
    assert_eq!(code, 2);
}
