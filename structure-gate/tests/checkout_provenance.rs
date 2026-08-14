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
//! What these do NOT catch: a stale *test* binary for a crate outside the gate
//! set, or a test fixture resolved from `CARGO_MANIFEST_DIR`. Cross-checkout
//! reuse is what makes a shared target directory worth having, and only code
//! whose output describes the tree opts out of it. For those, the exposure is a
//! stale assertion rather than a published number, and `cargo clean -p <crate>`
//! is the cure.

#![forbid(unsafe_code)]

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
    let invoked_in = canonical(
        &std::env::current_dir().expect("Fix: the working directory must be readable"),
    );

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

/// No shipped gate resolves the checkout from a value fixed at compile time.
///
/// The member list is read at run time, so a new gate binary that bakes a path
/// in turns this red instead of inheriting the defect in silence. Both spellings
/// are named: `CARGO_MANIFEST_DIR` is the one that caused this, and
/// `VYRE_CHECKOUT_ROOT` is the fix that did not work and must not come back.
#[test]
fn no_shipped_gate_resolves_the_checkout_from_a_compiled_in_path() {
    let root = structure_gate::workspace_root();
    let mut offenders = Vec::new();

    for member in members(&root) {
        let crate_dir = root.join(&member);
        if !ships_a_binary(&crate_dir) {
            continue;
        }
        for source in member_sources(&crate_dir.join("src")) {
            // Production code only: a fixture path in a `#[cfg(test)]` module is
            // a stale assertion at worst, which this file's header puts out of
            // scope, and the stripper has one owner in the gate itself.
            let production = structure_gate::strip_cfg_test_items(&source);
            if production.contains("env!(\"CARGO_MANIFEST_DIR\")")
                || production.contains("env!(\"VYRE_CHECKOUT_ROOT\"")
            {
                offenders.push(member.clone());
                break;
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "{offenders:?} ship a binary that resolves a repository path from a compiled-in value. \
         Cargo reuses a binary across checkouts that share a target directory, so that value \
         names whichever tree built last. Fix: resolve the root from the working directory at \
         run time, as xtask::checkout and structure_gate::workspace_root do."
    );
}

fn declares_workspace(manifest: &Path) -> bool {
    std::fs::read_to_string(manifest).is_ok_and(|text| {
        text.lines()
            .any(|line| line.trim_start().starts_with("[workspace]"))
    })
}

/// Workspace members, as declared by the root manifest.
fn members(root: &Path) -> Vec<String> {
    let text = std::fs::read_to_string(root.join("Cargo.toml"))
        .expect("Fix: the workspace manifest must be readable");
    let value = toml::from_str::<toml::Value>(&text)
        .expect("Fix: the workspace manifest must parse as TOML");
    value
        .get("workspace")
        .and_then(|workspace| workspace.get("members"))
        .and_then(toml::Value::as_array)
        .map(|members| {
            members
                .iter()
                .filter_map(toml::Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// Whether a member produces an executable.
fn ships_a_binary(crate_dir: &Path) -> bool {
    if crate_dir.join("src/main.rs").is_file() || crate_dir.join("src/bin").is_dir() {
        return true;
    }
    std::fs::read_to_string(crate_dir.join("Cargo.toml"))
        .is_ok_and(|text| text.contains("[[bin]]"))
}

/// Every `.rs` file under a crate's `src`, as text.
fn member_sources(src: &Path) -> Vec<String> {
    let mut sources = Vec::new();
    let mut pending = vec![src.to_path_buf()];
    while let Some(directory) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&directory) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                if let Ok(text) = std::fs::read_to_string(&path) {
                    sources.push(text);
                }
            }
        }
    }
    sources
}
