//! The git checkout a gate test runs a gate against.
//!
//! `Tree::open` reads a git checkout, so every gate test needs one, and five
//! test modules each wrote the same directory-create, file-write, `git init`
//! sequence with a different failure message. One owner means a gate test that
//! adds a fixture file states only the fixture.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

use tempfile::TempDir;

/// A git checkout holding the given files, each path relative to its root.
///
/// The returned directory owns the tree: dropping it deletes the checkout, so a
/// caller holds it for as long as the gate reads it.
///
/// # Panics
///
/// Panics when a fixture file cannot be written, or when `git` is missing or
/// fails to initialize the checkout, because a gate test proves nothing against
/// a tree that was not built.
pub fn checkout(files: &[(&str, &str)]) -> (TempDir, PathBuf) {
    let temporary = TempDir::new().expect("a temporary directory for the fixture checkout");
    let root = temporary.path().to_path_buf();
    for (path, text) in files {
        let target = root.join(path);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)
                .unwrap_or_else(|error| panic!("the fixture directory for {path}: {error}"));
        }
        fs::write(&target, text)
            .unwrap_or_else(|error| panic!("the fixture file {path}: {error}"));
    }
    let status = Command::new("git")
        .args(["init", "-q", "."])
        .current_dir(&root)
        .status()
        .expect("git is available to initialize a fixture checkout");
    assert!(
        status.success(),
        "git init failed in the fixture checkout, so no gate can read it"
    );
    (temporary, root)
}

/// A git checkout holding the given directories, each with one Rust file.
///
/// A rule scoped to roots reports a missing root, so a test of what the rule
/// finds inside them needs every root to exist.
///
/// # Panics
///
/// Panics for the same reasons as [`checkout`].
pub fn checkout_with_roots(roots: &[&str]) -> (TempDir, PathBuf) {
    let files: Vec<(String, &str)> = roots
        .iter()
        .map(|root| (format!("{root}/owner.rs"), "fn owner() {}\n"))
        .collect();
    let borrowed: Vec<(&str, &str)> = files
        .iter()
        .map(|(path, text)| (path.as_str(), *text))
        .collect();
    checkout(&borrowed)
}
