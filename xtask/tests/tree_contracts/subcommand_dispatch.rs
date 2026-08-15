//! Only `delegate` resolves the binary that runs a subcommand.
//!
//! WHY: `xtask` owns the subcommand table and routes by name. A build-task
//! binary that spawns `std::env::current_exe` instead is right only while it
//! happens to be `xtask`: `release-evidence` did exactly that, moved to
//! `xtask-evidence`, and re-entered itself for thirteen subcommands, reporting
//! twelve as unimplemented while labelling each failure `xtask <name>`. Nothing
//! went red, because every table still agreed with every other table; the wrong
//! process was being asked.
//!
//! So the contract is about who may answer the question at all. `delegate` owns
//! it, every other file asks `delegate`, and the packages scanned are derived
//! from the workspace roster at run time rather than listed here.

use std::path::{Path, PathBuf};

use super::common::workspace_root;

/// The one file allowed to resolve the running binary, relative to the checkout.
const OWNER: &str = "xtask/src/delegate.rs";

/// The call no other build-task source may make.
const SELF_RESOLUTION: &str = "current_exe";

/// Build-task packages: the dispatcher and its delegates, from the roster.
///
/// A delegate is a workspace member named after the dispatcher, so a third one
/// added tomorrow is scanned tomorrow without editing this file.
fn build_task_packages(root: &Path) -> Vec<String> {
    let mut packages: Vec<String> = structure_gate::workspace_members(root)
        .into_iter()
        .filter(|package| package == "xtask" || package.starts_with("xtask-"))
        .collect();
    packages.sort();
    packages
}

/// Every `.rs` file under `directory`, in sorted order.
fn rust_sources(directory: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let Ok(entries) = std::fs::read_dir(directory) else {
        return found;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            found.extend(rust_sources(&path));
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            found.push(path);
        }
    }
    found.sort();
    found
}

/// Files under every build-task package's `src` that name `SELF_RESOLUTION`,
/// reported relative to the checkout root.
fn self_resolving_sources(root: &Path) -> Vec<String> {
    let mut found = Vec::new();
    for package in build_task_packages(root) {
        let directory = structure_gate::member_directory(root, &package);
        for source in rust_sources(&directory.join("src")) {
            let text = std::fs::read_to_string(&source)
                .unwrap_or_else(|error| panic!("Fix: cannot read {}: {error}", source.display()));
            if text.contains(SELF_RESOLUTION) {
                let relative = source
                    .strip_prefix(root)
                    .unwrap_or(&source)
                    .to_string_lossy()
                    .replace('\\', "/");
                found.push(relative);
            }
        }
    }
    found.sort();
    found
}

/// WHY: the scan is worthless if it cannot see the text it forbids, and a roster
/// that returned one package would let a delegate carry the defect unseen. Both
/// are asserted before the prohibition, so a broken roster or a moved owner
/// fails as itself instead of as a clean tree.
#[test]
fn the_scan_covers_the_dispatcher_and_its_delegates() {
    let root = workspace_root();
    let packages = build_task_packages(&root);
    assert!(
        packages.len() >= 2 && packages.contains(&"xtask".to_string()),
        "Fix: the build-task roster is {packages:?}; it must name the dispatcher \
         and at least one delegate, or this contract judges nothing"
    );
    let owner = root.join(OWNER);
    let text = std::fs::read_to_string(&owner)
        .unwrap_or_else(|error| panic!("Fix: cannot read {}: {error}", owner.display()));
    assert!(
        text.contains(SELF_RESOLUTION),
        "Fix: {OWNER} no longer resolves the running binary. Move this contract \
         to whichever file owns that answer now."
    );
}

/// WHY: one owner for "which binary runs this subcommand". A second answer is
/// not a duplicate spelling of the same thing: it is a different answer in every
/// binary that is not the dispatcher, and it fails as the subcommand's own gate.
#[test]
fn no_other_build_task_source_resolves_its_own_binary() {
    let root = workspace_root();
    let offenders: Vec<String> = self_resolving_sources(&root)
        .into_iter()
        .filter(|path| path != OWNER)
        .collect();
    assert!(
        offenders.is_empty(),
        "Fix: {offenders:?} resolve the running binary directly. Call \
         `xtask::delegate::dispatcher()`, which returns the binary that owns the \
         subcommand table; `{SELF_RESOLUTION}` names whichever binary is running \
         and is the dispatcher only inside `xtask`."
    );
}
