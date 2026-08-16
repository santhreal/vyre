//! What the root manifest must say about the example crates.
//!
//! An example directory that becomes a workspace member drags an out-of-tree
//! demonstrator into every workspace build and lets it inherit the workspace
//! lints, patches and features it exists to prove a consumer does not need.
//! `exclude` is what keeps it out, so the manifest is the contract and this is
//! where it is read.
//!
//! Whether each example carries a manifest, declares its own workspace, builds
//! outside this tree and passes what it asserts belongs to the
//! `example-capability` gate, which builds them. Nothing here restates it.

use std::collections::BTreeSet;
use std::path::Path;

use vyre_test_support::monorepo::{vyre_workspace_root, vyre_workspace_rosters};

/// Directory names under `examples/` that carry at least one file.
fn example_directories(examples: &Path) -> BTreeSet<String> {
    let Ok(entries) = std::fs::read_dir(examples) else {
        return BTreeSet::new();
    };
    entries
        .flatten()
        .filter(|entry| entry.path().is_dir())
        .filter(|entry| {
            entry.path().read_dir().is_ok_and(|files| {
                files
                    .flatten()
                    .any(|file| file.file_type().is_ok_and(|kind| kind.is_file()))
            })
        })
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect()
}

#[test]
fn every_example_directory_is_excluded_from_the_workspace() {
    let root = vyre_workspace_root();
    let directories = example_directories(&root.join("examples"));
    assert!(
        !directories.is_empty(),
        "the checkout tracks no example, so this contract has no subject"
    );
    let rosters = vyre_workspace_rosters();

    let mut violations = Vec::new();
    for name in &directories {
        let path = format!("examples/{name}");
        if !rosters.excluded.contains(&path) {
            violations.push(format!("{path} is missing from workspace exclude"));
        }
        if rosters.members.contains(&path) {
            violations.push(format!("{path} is a workspace member"));
        }
    }

    assert!(
        violations.is_empty(),
        "every example directory is excluded and none is a member. Violations:\n{}",
        violations.join("\n")
    );
}

#[test]
fn every_example_path_the_root_manifest_names_exists() {
    let root = vyre_workspace_root();
    let text = std::fs::read_to_string(root.join("Cargo.toml"))
        .expect("the workspace root carries a manifest");

    let mut violations = Vec::new();
    for (number, line) in text.lines().enumerate() {
        for prefix in ["examples/", "demos/"] {
            let Some(at) = line.find(prefix) else {
                continue;
            };
            let named: String = line[at..]
                .chars()
                .take_while(|character| {
                    character.is_ascii_alphanumeric() || "/_-.".contains(*character)
                })
                .collect();
            if !root.join(&named).exists() {
                violations.push(format!(
                    "Cargo.toml:{} names `{named}`, which does not exist",
                    number + 1
                ));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "the root manifest names no path that was deleted. Violations:\n{}",
        violations.join("\n")
    );
}
