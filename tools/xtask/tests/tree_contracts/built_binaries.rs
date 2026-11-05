//! One lock over the executables cargo built for this harness.
//!
//! `cli-docs` rebuilds every workspace binary into the profile directory these
//! tests execute from, and the rebuild relinks any binary whose feature set
//! differs from the test build's. Windows fails to open an executable for
//! writing while any process runs it, so a contract executing `xtask` or a
//! helper binary at that moment fails the link with `LNK1104`. Contracts that
//! execute a built binary hold [`shared`]; the contract that runs `cli-docs`
//! holds [`exclusive`].

use std::sync::{PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::workspace_sources::{sources_under, workspace_root};

static BUILT_BINARIES: RwLock<()> = RwLock::new(());

/// Hold while a contract executes a cargo-built binary.
pub(crate) fn shared() -> RwLockReadGuard<'static, ()> {
    BUILT_BINARIES
        .read()
        .unwrap_or_else(PoisonError::into_inner)
}

/// Hold while a contract runs a command that rebuilds the built binaries.
pub(crate) fn exclusive() -> RwLockWriteGuard<'static, ()> {
    BUILT_BINARIES
        .write()
        .unwrap_or_else(PoisonError::into_inner)
}

/// WHY: the lock excludes only the contracts that take it. A contract added
/// with `CARGO_BIN_EXE_` and no guard races the `cli-docs` rebuild again, and
/// on Windows that fails as an unrelated link error in another test. The roster
/// is every source file of this harness, read at run time.
#[test]
fn every_contract_that_executes_a_built_binary_holds_the_lock() {
    let directory = workspace_root().join("tools/xtask/tests/tree_contracts");
    let sources = sources_under(&directory, &["rs"]);
    assert!(
        sources.len() > 1,
        "Fix: the scan found {} source file(s) under {}; it judges nothing",
        sources.len(),
        directory.display()
    );
    let unguarded: Vec<String> = sources
        .iter()
        .filter(|path| !path.ends_with("built_binaries.rs"))
        .filter_map(|path| {
            let text = std::fs::read_to_string(path)
                .unwrap_or_else(|error| panic!("Fix: cannot read {}: {error}", path.display()));
            (text.contains("CARGO_BIN_EXE_") && !text.contains("built_binaries::"))
                .then(|| path.display().to_string())
        })
        .collect();
    assert!(
        unguarded.is_empty(),
        "Fix: {unguarded:?} execute a cargo-built binary without the built_binaries lock. \
         Hold `built_binaries::shared()` while the binary runs."
    );
}
