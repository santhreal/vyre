//! What the structural scan counts as a crate and as a source file.
//!
//! WHY: every layout rule judges the rosters `scan` builds, so a roster that
//! includes build output convicts generated code of the rules written for
//! checked-in source, and a roster that walks build output pays a directory
//! round trip per artifact to find nothing. Both were true here: the crate-root
//! walk and the per-tree source walk descended into `target/` and into hidden
//! directories, and the `src/` module roster walked the same directories the
//! source roster had already walked.
//!
//! What these cover: a manifest under `target/` or under a hidden directory is
//! not a crate of the checkout, a `.rs` file under a nested `target/` is not a
//! source of it, and the module roster is exactly the `src/` part of the source
//! roster rather than an independently walked second answer.

#![forbid(unsafe_code)]

use std::fs;
use std::path::Path;

/// A fixture checkout: a root workspace manifest plus the given files.
fn checkout(files: &[(&str, &str)]) -> tempfile::TempDir {
    let root = tempfile::tempdir().expect("Fix: fixture checkout must be creatable");
    write(
        root.path(),
        "Cargo.toml",
        "[workspace]\nmembers = [\"kept\"]\n",
    );
    for (path, text) in files {
        write(root.path(), path, text);
    }
    root
}

fn write(root: &Path, relative: &str, text: &str) {
    let path = root.join(relative);
    fs::create_dir_all(path.parent().expect("Fix: a fixture file has a parent"))
        .expect("Fix: fixture directory must be creatable");
    fs::write(&path, text).expect("Fix: fixture file must be writable");
}

const MANIFEST: &str = "[package]\nname = \"kept\"\nversion = \"0.0.0\"\n";
const SOURCE: &str = "pub fn kept() {}\n";

#[test]
fn build_output_and_hidden_directories_are_not_crate_roots() {
    let root = checkout(&[
        ("kept/Cargo.toml", MANIFEST),
        ("kept/src/lib.rs", SOURCE),
        ("target/unpacked/Cargo.toml", MANIFEST),
        ("target/unpacked/src/lib.rs", SOURCE),
        (".cache/vendored/Cargo.toml", MANIFEST),
        (".cache/vendored/src/lib.rs", SOURCE),
    ]);

    let directories: Vec<String> = structure_gate::scan(root.path())
        .crate_roots
        .into_iter()
        .map(|crate_root| crate_root.directory)
        .collect();

    assert_eq!(
        directories,
        vec!["kept".to_string()],
        "Fix: the crate-root roster must hold only crates of the checkout. A manifest under `target/` is a dependency cargo unpacked and a manifest under a hidden directory is a cache, and judging either applies the layout rules to code nobody here wrote."
    );
}

#[test]
fn build_output_under_a_crate_is_not_a_source_file() {
    let root = checkout(&[
        ("kept/Cargo.toml", MANIFEST),
        ("kept/src/lib.rs", SOURCE),
        ("kept/src/target/generated.rs", SOURCE),
        ("kept/tests/all_tests.rs", SOURCE),
        ("kept/tests/.hidden/scratch.rs", SOURCE),
    ]);

    let source_files = structure_gate::scan(root.path()).source_files;

    assert_eq!(
        source_files,
        vec![
            "kept/src/lib.rs".to_string(),
            "kept/tests/all_tests.rs".to_string(),
        ],
        "Fix: the source roster must skip build output and hidden directories under a crate. A fuzz target keeps its own `target/`, and every generated file under it would otherwise be judged as checked-in source."
    );
}

#[test]
fn the_module_roster_is_the_src_part_of_the_source_roster() {
    let root = checkout(&[
        ("kept/Cargo.toml", MANIFEST),
        ("kept/src/lib.rs", SOURCE),
        ("kept/src/nested/deep.rs", SOURCE),
        ("kept/tests/all_tests.rs", SOURCE),
        ("kept/benches/throughput.rs", SOURCE),
        (
            "second/Cargo.toml",
            "[package]\nname = \"second\"\nversion = \"0.0.0\"\n",
        ),
        ("second/src/lib.rs", SOURCE),
    ]);

    let workspace = structure_gate::scan(root.path());
    let expected: Vec<String> = workspace
        .source_files
        .iter()
        .filter(|file| file.contains("/src/"))
        .cloned()
        .collect();

    assert_eq!(
        workspace.module_files, expected,
        "Fix: the module roster must be the `src/` part of the source roster. `src` is one of the source trees, so a second walk of it can only report the same paths, and two independent walks disagree the first time one of them is pruned differently."
    );
    assert!(
        workspace
            .module_files
            .contains(&"kept/src/nested/deep.rs".to_string()),
        "Fix: the module roster must reach every depth under `src/`, not only its top level."
    );
}
