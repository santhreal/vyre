//! Workspace-level contract tests (VYRE-TASK-000003).
//!
//! These modules are wired into `vyre-foundation` via
//! `tests/contract_workspace.rs` so `cargo test -p vyre-foundation contract`
//! executes cross-crate invariants without a dedicated workspace test crate.

mod claims_inventory_smoke;
mod foundation_validate_contract;
mod public_api_surface;
mod xtask_help_smoke;

/// Workspace root (`vyre/` directory).
pub(crate) fn workspace_root() -> std::path::PathBuf {
    vyre_test_support::monorepo::vyre_workspace_root()
}
