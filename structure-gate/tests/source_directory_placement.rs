//! Where a domain's code lives is read from the tree, not from its name.
//!
//! WHY: the operation matrix names an owner directory for every registered op,
//! and operation ids are frozen, so the id supplies a crate and a domain that
//! may both have moved. Two answers went wrong in this tree. Asking whether the
//! id-derived directory exists named an empty shell left behind by a deletion,
//! because git tracks files and not directories. Asking only for a top-level
//! module named the domain missed `vyre-libs/src/nn/optim`, whose ops carry an
//! `optim` domain that has no top-level directory, and a table of moved names
//! goes stale on the next move.
//!
//! What these cover: a directory that holds no Rust source is not an owner, a
//! nested domain is found, the shallowest of two same-named directories wins,
//! and an unknown domain yields nothing so the caller can report the row rather
//! than mint a plausible path.

#![forbid(unsafe_code)]

use std::fs;
use std::path::Path;

use structure_gate::source_scan::{carries_rust_source, source_directory_named};

fn tree(directories: &[&str], sources: &[&str]) -> tempfile::TempDir {
    let root = tempfile::tempdir().expect("Fix: fixture tree must be creatable");
    for directory in directories {
        fs::create_dir_all(root.path().join(directory))
            .expect("Fix: fixture directory must be creatable");
    }
    for source in sources {
        let path = root.path().join(source);
        fs::create_dir_all(path.parent().expect("Fix: a fixture source has a parent"))
            .expect("Fix: fixture directory must be creatable");
        fs::write(&path, "pub fn placed() {}\n").expect("Fix: fixture source must be writable");
    }
    root
}

fn found(root: &Path, name: &str) -> Option<String> {
    source_directory_named(root, name).map(|path| {
        path.strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/")
    })
}

#[test]
fn a_directory_holding_no_rust_source_is_not_an_owner() {
    let root = tree(&["src/matching"], &["src/scan/kernel.rs"]);

    assert!(
        !carries_rust_source(&root.path().join("src/matching")),
        "an emptied directory answered as code"
    );
    assert_eq!(found(root.path(), "matching"), None);
}

#[test]
fn a_domain_nested_under_another_one_is_found() {
    let root = tree(&[], &["src/nn/optim/adam.rs", "src/nn/attention/mask.rs"]);

    assert_eq!(found(root.path(), "optim").as_deref(), Some("src/nn/optim"));
}

#[test]
fn the_shallowest_of_two_directories_with_one_name_wins() {
    let root = tree(
        &[],
        &["src/quant/scale.rs", "src/nn/quant/per_channel/scale.rs"],
    );

    assert_eq!(found(root.path(), "quant").as_deref(), Some("src/quant"));
}

#[test]
fn a_shell_at_the_shallow_position_loses_to_the_directory_with_code() {
    let root = tree(&["src/quant"], &["src/nn/quant/per_channel/scale.rs"]);

    assert_eq!(
        found(root.path(), "quant").as_deref(),
        Some("src/nn/quant"),
        "an emptied top-level directory outranked the code"
    );
}

#[test]
fn a_domain_no_directory_carries_yields_nothing() {
    let root = tree(&[], &["src/graph/toposort.rs"]);

    assert_eq!(found(root.path(), "matching"), None);
}

#[test]
fn a_build_output_directory_is_never_an_owner() {
    let root = tree(&[], &["target/debug/build/optim/generated.rs"]);

    assert_eq!(found(root.path(), "optim"), None);
}
