//! The gates this crate implements, and the sweep that runs the registry.
//!
//! A gate here reads the repository and reports findings. There is no other
//! category: what used to be a composite that re-ran other subcommands is now a
//! named subset of the registry, and the cargo invocations it drove are gates of
//! their own, so the sweep holds them to a baseline like everything else.
//!
//! `gates::sweep` is the runner and the wiring meta-check that keeps every
//! registered gate connected to a pinned baseline and a workflow.

pub mod check_tier_deps;
pub mod dedup_report;
pub mod dep_drift;
pub mod dup_scan;
pub mod feature_isolation;
pub mod hot_path_scan;
pub mod hygiene_matrix;
pub mod implementation_family;
pub mod lockfile;
pub mod op_names;
pub mod ownership;
pub mod parity_testing;
pub mod platform_boundary;
pub mod sweep;
pub mod use_paths;
pub mod workspace_build;

use crate::gate::Gate;

/// Every gate this module owns.
///
/// The registry is assembled from one slice per area at run time, so adding a
/// gate is adding it here and nowhere else. The sweep enumerates what this
/// yields, which is why a gate cannot be registered and left unswept.
pub static GATES: &[&dyn Gate] = &[
    &check_tier_deps::CheckTierDeps,
    &dep_drift::DepDrift,
    &dup_scan::DupScan,
    &feature_isolation::FeatureIsolation,
    &hot_path_scan::HotPathScan,
    &hygiene_matrix::HygieneMatrix,
    &lockfile::LockfileClean,
    &op_names::OpNames,
    &parity_testing::ParityTestingIsolated,
    &platform_boundary::PlatformBoundary,
    &workspace_build::WorkspaceCheck,
    &workspace_build::WorkspaceClippy,
    &workspace_build::WorkspaceDocs,
    &workspace_build::WorkspaceTests,
];
