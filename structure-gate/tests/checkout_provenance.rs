//! A gate must report the checkout it was invoked in.
//!
//! WHY: every checkout of this repository shares one cargo target directory, and
//! cargo hashes a workspace member by its path *relative* to the workspace root,
//! then checks freshness by mtime. Two checkouts therefore compute the same unit
//! hash, address the same artifacts, and hand each other compiled binaries. A
//! gate that resolves the tree from anything fixed at compile time then answers
//! for whichever tree built last: measured twice, once as 208 violations from a
//! worktree whose own source said 209, and once as `dup-scan` reporting this
//! tree's `xtask` at 473 duplicated lines against a pin of 465 while this tree's
//! `xtask/dup-baseline.toml` said 411, because the shared binary carried a
//! worktree's path and read that worktree's baseline file.
//!
//! The first attempt at closing this was a `VYRE_CHECKOUT_ROOT` in
//! `.cargo/config.toml` read with `env!`, on the theory that recording the value
//! in each crate's dep-info would force a per-checkout rebuild. It did not hold,
//! and it failed silently, which is worse than the bug: cargo does not export a
//! `relative = true` config variable to the process it runs, so the run-time
//! lookup always fell through to the compiled-in value. These tests assert the
//! property instead of that mechanism.
//!
//! What these cover: every `.rs` file under every workspace member's `src/`,
//! `tests/` and `benches/`, plus the root `tests/` directory, with nothing
//! stripped first. Test binaries and test fixtures used to be out of scope, and
//! that exclusion was the hole. A fixture or golden-file locator built from a
//! compiled-in crate directory made a byte-stability test read the *other*
//! checkout's golden file and assert byte stability against it, and the same
//! mechanism can make a gate pass while the working tree is wrong, which is the
//! dangerous direction.
//!
//! What these still do NOT catch: a stale test binary for a crate that has no
//! gate at all. Nothing in such a crate resolves a repository path, so there is
//! no spelling to reject; the reused binary simply carries another tree's code.
//! `cargo clean -p <crate>` is the cure.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Canonical form, so a symlinked or trailing-slash path compares by identity.
fn canonical(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

/// The root the gate resolves contains the directory the gate was invoked in.
///
/// This is the property the mechanism was supposed to deliver, stated without
/// reference to the mechanism. A binary compiled by another checkout returns
/// that checkout's root, which cannot be an ancestor of this run's working
/// directory, so the reuse this file exists for turns it red.
#[test]
fn the_resolved_root_is_the_tree_this_run_is_inside() {
    let resolved = canonical(&structure_gate::workspace_root());
    let invoked_in =
        canonical(&std::env::current_dir().expect("Fix: the working directory must be readable"));

    assert!(
        invoked_in.starts_with(&resolved),
        "the gate resolved `{}` as the checkout root, which does not contain `{}`, the directory \
         this run was invoked in. A target directory shared by several checkouts handed this run \
         a binary built from another tree, so every number it reports describes that tree.",
        resolved.display(),
        invoked_in.display()
    );
    assert!(
        declares_workspace(&resolved.join("Cargo.toml")),
        "the gate resolved `{}`, whose Cargo.toml does not declare a [workspace]",
        resolved.display()
    );
}

/// The root is resolved by walking, so a member directory yields the workspace.
///
/// Cargo sets the working directory to a member crate for `cargo test`, so the
/// walk is what makes the property above hold in the common case.
#[test]
fn a_member_directory_resolves_to_the_workspace_above_it() {
    let root = canonical(&structure_gate::workspace_root());
    let member = root.join("structure-gate");
    assert!(
        member.join("Cargo.toml").is_file(),
        "fixture manifest moved: {}",
        member.display()
    );

    let resolved = structure_gate::workspace_root_from(&member).map(|path| canonical(&path));

    assert_eq!(
        resolved.as_deref(),
        Some(root.as_path()),
        "a member directory must resolve to the workspace root above it"
    );
}

/// No source file in the workspace resolves the checkout from a compiled-in path.
///
/// The member list is read from the root manifest and every directory is walked,
/// both at run time, so a new crate and a new file are covered without an edit
/// here: a new offender turns this red by default rather than inheriting the
/// defect in silence.
///
/// Nothing is stripped before the scan. A path resolved inside a `#[cfg(test)]`
/// module, an integration test, a bench, or a fixture locator is exactly the
/// case this test was extended to cover, because a golden file read out of the
/// other checkout makes a byte-stability assertion describe that tree.
///
/// Both spellings are rejected: `CARGO_MANIFEST_DIR` is the one that caused
/// this, and `VYRE_CHECKOUT_ROOT` is the fix that did not work and must not come
/// back. [`banned_spellings`] assembles both at run time so neither literal
/// appears in this file, which is why the gate does not report itself and needs
/// no exemption list.
#[test]
fn no_source_file_resolves_the_checkout_from_a_compiled_in_path() {
    let root = structure_gate::workspace_root();
    let needles = banned_spellings();
    let mut offenders = Vec::new();
    let mut scanned = 0_usize;

    for (label, directory) in scanned_directories(&root) {
        for source in rust_files(&directory) {
            let Ok(text) = std::fs::read_to_string(&source) else {
                continue;
            };
            scanned += 1;
            if needles.iter().any(|needle| text.contains(needle.as_str())) {
                let shown = source.strip_prefix(&root).unwrap_or(source.as_path());
                offenders.push(format!("{label}: {}", shown.display()));
            }
        }
    }
    offenders.sort();

    assert!(
        scanned > 0,
        "the scan read no source files, so it is guarding nothing. Fix: check that \
         `{}` is a checkout of this workspace.",
        root.display()
    );
    assert!(
        offenders.is_empty(),
        "{offenders:#?} resolve a repository path from a directory fixed at compile time. \
         Cargo reuses a compiled unit across checkouts that share a target directory, so that \
         directory names whichever tree built last: the file read is that tree's document, pin, \
         fixture or golden file, and the assertion describes that tree while claiming to \
         describe this one. Fix: resolve the root from the working directory at run time, with \
         `structure_gate::workspace_root` or the `vyre_test_support::monorepo` delegation to it, \
         and join the crate directory name onto it for an own-crate fixture path."
    );
}

/// The two rejected spellings, assembled from parts at run time.
///
/// Written this way so neither literal appears in this file. A gate that reports
/// itself gets an exemption, and an exemption list is a hole of exactly the kind
/// this test exists to close.
fn banned_spellings() -> Vec<String> {
    ["CARGO_MANIFEST_DIR", "VYRE_CHECKOUT_ROOT"]
        .iter()
        .map(|name| format!("env{}({name:?})", "!"))
        .collect()
}

/// Every directory the scan reads, paired with the member that owns it.
///
/// Members come from the root manifest, so a new crate is covered by adding it
/// to the workspace rather than by editing this file. The root `tests/`
/// directory holds cross-crate contracts compiled into `vyre-foundation` and is
/// no member's `tests/`, so it is named separately.
fn scanned_directories(root: &Path) -> Vec<(String, PathBuf)> {
    let mut owners: BTreeSet<String> = structure_gate::workspace_members(root)
        .into_iter()
        .collect();
    owners.extend(conform_members(root));

    let mut directories = Vec::new();
    for member in owners {
        let crate_dir = root.join(&member);
        for area in ["src", "tests", "benches"] {
            let candidate = crate_dir.join(area);
            if candidate.is_dir() {
                directories.push((member.clone(), candidate));
            }
        }
    }

    let root_tests = root.join("tests");
    if root_tests.is_dir() {
        directories.push(("tests".to_string(), root_tests));
    }

    assert!(
        directories.len() > 1,
        "the workspace manifest at `{}` yielded no member source directories",
        root.display()
    );
    directories
}

/// Conformance crates, discovered on disk rather than read from the manifest.
///
/// They are ordinary workspace members and normally arrive through [`members`].
/// Discovering them as well covers the one case that list cannot: a conformance
/// crate dropped from the manifest stops being a member and stops being scanned
/// in the same edit, and its harness is where a stale golden file does the most
/// damage.
fn conform_members(root: &Path) -> Vec<String> {
    let mut found = Vec::new();
    let Ok(entries) = std::fs::read_dir(root.join("conform")) else {
        return found;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.join("Cargo.toml").is_file() {
            continue;
        }
        if let Some(name) = path.file_name().and_then(|name| name.to_str()) {
            found.push(format!("conform/{name}"));
        }
    }
    found
}

fn declares_workspace(manifest: &Path) -> bool {
    std::fs::read_to_string(manifest).is_ok_and(|text| {
        text.lines()
            .any(|line| line.trim_start().starts_with("[workspace]"))
    })
}

/// Every `.rs` file under `directory`, at any depth.
///
/// Walked at run time, so a file added to a scanned directory is covered without
/// an edit here. That is the fail-by-default half of the closure: a listed set of
/// files goes stale in silence, which is the same failure as having no gate.
fn rust_files(directory: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut pending = vec![directory.to_path_buf()];
    while let Some(current) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&current) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                files.push(path);
            }
        }
    }
    files
}
