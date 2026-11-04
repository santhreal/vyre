//! Path resolution the `vyre-libs-parsing` contract tests share.

use std::path::PathBuf;

/// This crate's directory, resolved from the working directory at run time.
pub(crate) fn crate_dir() -> PathBuf {
    vyre_test_support::monorepo::vyre_crate_directory(env!("CARGO_PKG_NAME"))
}
