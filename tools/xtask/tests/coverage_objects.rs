//! The coverage report reads only executables the coverage run built.
//!
//! Closes the class where an instrumented executable left in the shared
//! profile directory by an older commit, another feature set, or another
//! checkout is read by `cargo llvm-cov report`. Its regions have no profile in
//! the run, so every one of its lines counts as missed and the measured
//! coverage falls with no source change.
//!
//! Not covered here: whether cargo lists every executable a test spawns. A
//! test that builds a binary through its own cargo invocation outside
//! `examples` is deleted as stale, and this suite cannot see such a test.
#![cfg(unix)]

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use serde_json::json;
use xtask::coverage_objects::{prune, select};

struct Layout {
    _dir: tempfile::TempDir,
    profile: PathBuf,
}

impl Layout {
    fn new() -> Self {
        let dir = tempfile::tempdir().expect("Fix: fixture directory must exist");
        let profile = dir.path().join("llvm-cov-target/debug");
        fs::create_dir_all(profile.join("deps")).expect("Fix: deps directory must exist");
        fs::create_dir_all(profile.join("examples")).expect("Fix: examples directory must exist");
        Self { _dir: dir, profile }
    }

    fn file(&self, relative: &str, executable: bool) -> PathBuf {
        let path = self.profile.join(relative);
        fs::write(&path, relative).expect("Fix: fixture file must be writable");
        let mode = if executable { 0o755 } else { 0o644 };
        fs::set_permissions(&path, fs::Permissions::from_mode(mode))
            .expect("Fix: fixture permissions must be settable");
        path
    }
}

fn artifact(filenames: &[&Path], executable: Option<&Path>, test: bool) -> String {
    json!({
        "reason": "compiler-artifact",
        "filenames": filenames,
        "executable": executable,
        "profile": { "test": test },
    })
    .to_string()
}

/// A run's layout: listed test and binary executables, a hard link of a listed
/// binary, an unlisted example, a `.d` file, and three stale executables.
fn populated() -> (Layout, String, Vec<PathBuf>) {
    let layout = Layout::new();
    let test = layout.file("deps/all_tests-new", true);
    let rlib = layout.file("deps/libcrate-new.rlib", false);
    let bin = layout.file("deps/tool-new", true);
    fs::hard_link(&bin, layout.profile.join("tool")).expect("Fix: hard link must be creatable");
    layout.file("examples/demo", true);
    layout.file("deps/all_tests-new.d", true);
    let stale = vec![
        layout.file("deps/all_tests-old", true),
        layout.file("deps/libmacro-old.so", true),
        layout.file("tool-old", true),
    ];
    let messages = [
        "plain diagnostic text".to_string(),
        artifact(&[&rlib], None, false),
        artifact(&[&test], Some(&test), true),
        artifact(&[&bin], Some(&bin), false),
        json!({"reason": "build-finished", "success": true}).to_string(),
    ]
    .join("\n");
    let mut stale = stale;
    stale.sort();
    (layout, messages, stale)
}

#[test]
fn only_unlisted_executables_outside_examples_are_selected() {
    let (_layout, messages, stale) = populated();
    let selection = select(&messages).expect("Fix: a run with a test executable must select");
    assert_eq!(selection.stale, stale);
    // The listed test, rlib, and binary (one identity for both of its links),
    // plus the example.
    assert_eq!(selection.kept, 4);
}

#[test]
fn prune_deletes_the_selection_and_nothing_else() {
    let (layout, messages, stale) = populated();
    let selection = prune(&messages).expect("Fix: prune must succeed on a writable layout");
    assert_eq!(selection.stale, stale);
    for path in &stale {
        assert!(!path.exists(), "{} must be deleted", path.display());
    }
    for kept in [
        "deps/all_tests-new",
        "deps/libcrate-new.rlib",
        "deps/tool-new",
        "deps/all_tests-new.d",
        "tool",
        "examples/demo",
    ] {
        assert!(layout.profile.join(kept).exists(), "{kept} must be kept");
    }
    assert_eq!(
        select(&messages)
            .expect("Fix: a pruned layout still selects")
            .stale,
        Vec::<PathBuf>::new(),
        "a second prune must find nothing"
    );
}

#[test]
fn a_run_without_a_test_executable_is_rejected() {
    let layout = Layout::new();
    let bin = layout.file("deps/tool-new", true);
    layout.file("deps/tool-old", true);
    let error = select(&artifact(&[&bin], Some(&bin), false)).unwrap_err();
    assert!(
        error.contains("no compiler-artifact record lists a test executable"),
        "{error}"
    );
    assert!(layout.profile.join("deps/tool-old").exists());
}

#[test]
fn test_executables_in_two_directories_are_rejected() {
    let first = Layout::new();
    let second = Layout::new();
    let a = first.file("deps/a", true);
    let b = second.file("deps/b", true);
    let messages = [
        artifact(&[&a], Some(&a), true),
        artifact(&[&b], Some(&b), true),
    ]
    .join("\n");
    let error = prune(&messages).unwrap_err();
    assert!(error.contains("were linked into"), "{error}");
    assert!(
        a.exists() && b.exists(),
        "a rejected run must delete nothing"
    );
}

#[test]
fn a_malformed_artifact_record_is_rejected() {
    let error = select("{\"reason\":\"compiler-artifact\",\"filenames\":7}").unwrap_err();
    assert!(error.contains("does not parse"), "{error}");
}
