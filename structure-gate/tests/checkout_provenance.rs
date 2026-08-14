//! The gate must report the checkout it was invoked in.
//!
//! WHY: every checkout of this repository shares one cargo target directory,
//! and cargo hashes a workspace member by its path *relative* to the workspace
//! root, then checks freshness by mtime. Two checkouts therefore compute the
//! same unit hash, address the same artifacts, and hand each other compiled
//! logic whenever the reader's files are older than the last build. The gate
//! read the live tree with another checkout's rules and reported that tree as
//! judged: 208 violations from one worktree while its own source said 209, and
//! `dup-scan` read 11811 duplicate lines for `vyre-libs` against a true 13406.
//! Nothing failed, which is why it survived.
//!
//! What these do NOT catch: a stale binary for a crate outside the gate set.
//! Cross-checkout reuse is still how a library crate builds in one second, and
//! only the crates whose output describes the tree opt out of it.

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};

/// Canonical form, so a symlinked or trailing-slash path compares by identity.
fn canonical(path: &Path) -> PathBuf {
    path.canonicalize()
        .unwrap_or_else(|error| panic!("Fix: {} must exist: {error}", path.display()))
}

/// The compiled gate belongs to the checkout it is judging.
///
/// `CARGO_MANIFEST_DIR` is set by cargo for this run and names the checkout the
/// gate was invoked in. `compiled_checkout_root` is baked in by whichever
/// checkout compiled the crate. They disagree exactly when the shared target
/// directory has handed this run a foreign binary.
#[test]
fn the_gate_binary_was_compiled_by_the_checkout_it_runs_in() {
    let invoked_crate = PathBuf::from(
        std::env::var_os("CARGO_MANIFEST_DIR")
            .expect("Fix: cargo sets CARGO_MANIFEST_DIR for every test run."),
    );
    let invoked = invoked_crate
        .parent()
        .expect("Fix: structure-gate must live under the vyre workspace root.");
    let compiled = structure_gate::compiled_checkout_root();

    assert_eq!(
        canonical(&compiled),
        canonical(invoked),
        "this gate was compiled by {} and run in {}. A target directory shared by \
         several checkouts handed this run a binary built from another tree, so every \
         number it reports describes that tree. Fix: keep VYRE_CHECKOUT_ROOT in \
         .cargo/config.toml and keep this crate reading it, or run \
         `cargo clean -p structure-gate` before trusting the run.",
        compiled.display(),
        invoked.display()
    );
}

/// Every gate binary that resolves a path inside the checkout declares the
/// input that tells two checkouts apart.
///
/// A gate binary is a member that ships an executable and locates files in the
/// tree: its whole output is a claim about that tree, so being handed another
/// checkout's binary makes the claim describe the wrong tree. The set is
/// derived from the member list at run time, so a new gate turns this red
/// instead of silently inheriting a binary from whichever tree compiled last.
///
/// Library members are deliberately out of scope. Cross-checkout reuse is what
/// makes a shared target directory worth having, and opting `vyre-libs` or
/// `vyre-driver` out of it would rebuild most of the workspace every time work
/// moved between two checkouts. Their exposure is a stale test binary, not a
/// wrong published number, and `cargo clean -p <crate>` is the cure.
#[test]
fn every_gate_binary_that_resolves_checkout_paths_declares_the_fingerprint_input() {
    let root = structure_gate::workspace_root();
    let members = structure_gate::scan(&root).members;

    let mut missing = Vec::new();
    for member in &members {
        let crate_dir = root.join(member);
        if !ships_a_binary(&crate_dir) {
            continue;
        }
        let source = member_sources(&crate_dir.join("src"));
        let resolves_paths = source.iter().any(|text| text.contains("CARGO_MANIFEST_DIR"));
        let declares_input = source
            .iter()
            .any(|text| text.contains("VYRE_CHECKOUT_ROOT") || text.contains("checkout_root()"));
        if resolves_paths && !declares_input {
            missing.push(member.clone());
        }
    }

    assert!(
        missing.is_empty(),
        "{missing:?} ship a binary that resolves paths inside the checkout from a \
         compiled-in manifest directory but read no VYRE_CHECKOUT_ROOT, so cargo may \
         hand them a binary compiled by another checkout that shares the target \
         directory. Fix: resolve the root through a `checkout_root()` that reads \
         `env!(\"VYRE_CHECKOUT_ROOT\")`."
    );
}

/// Whether a member produces an executable.
fn ships_a_binary(crate_dir: &Path) -> bool {
    if crate_dir.join("src/main.rs").is_file() || crate_dir.join("src/bin").is_dir() {
        return true;
    }
    std::fs::read_to_string(crate_dir.join("Cargo.toml"))
        .is_ok_and(|manifest| manifest.contains("[[bin]]"))
}

/// Every `.rs` file under a crate's `src`, as text.
fn member_sources(src: &Path) -> Vec<String> {
    walkdir::WalkDir::new(src)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "rs"))
        .filter_map(|entry| std::fs::read_to_string(entry.path()).ok())
        .collect()
}
