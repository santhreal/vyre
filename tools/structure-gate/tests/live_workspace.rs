//! The checked-in tree, scanned once per test binary.
//!
//! A scan parses every member source for registrations, materializers, module
//! routes and published modules. Every live-tree contract in this binary reads
//! the one scan [`workspace`] holds.

use std::sync::OnceLock;

use structure_gate::{scan, workspace_root, Workspace};

/// The scan of the checkout that contains the working directory.
pub fn workspace() -> &'static Workspace {
    static WORKSPACE: OnceLock<Workspace> = OnceLock::new();
    WORKSPACE.get_or_init(|| scan(&workspace_root()))
}
