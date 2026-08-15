//! The gates this crate implements, and the sweep that runs the registry.
//!
//! A gate here reads the repository and reports findings. There is no other
//! category: what used to be a composite that re-ran other subcommands is now a
//! named subset of the registry, and the cargo invocations it drove are gates of
//! their own, so the sweep holds them to a baseline like everything else.
//!
//! `gates::sweep` is the runner and the wiring meta-check that keeps every
//! registered gate connected to a pinned baseline and a workflow.

pub mod audit_status;
pub mod check_tier_deps;
pub mod dedup_report;
pub mod dep_drift;
pub mod doc_contract;
pub mod dup_scan;
pub mod evidence_paths;
pub mod feature_isolation;
pub mod file_size;
pub mod frozen_contract;
pub mod gpu_loudness;
pub mod hot_path;
pub mod hot_path_scan;
pub mod hygiene_matrix;
pub mod implementation_family;
pub mod lint_hygiene;
pub mod lockfile;
pub mod op_names;
pub mod ownership;
pub mod parity_testing;
pub mod platform_boundary;
pub mod platform_docs;
pub mod proptest_coverage;
pub mod repo_hygiene;
pub mod scan;
pub mod shader_source;
pub mod sweep;
pub mod unification;
pub mod use_paths;
pub mod workspace_build;

use crate::gate::Gate;

/// Every gate this module owns.
///
/// The registry is assembled from one slice per area at run time, so adding a
/// gate is adding it here and nowhere else. The sweep enumerates what this
/// yields, which is why a gate cannot be registered and left unswept.
pub static GATES: &[&dyn Gate] = &[
    &audit_status::AuditStatus,
    &check_tier_deps::CheckTierDeps,
    &dep_drift::DepDrift,
    &doc_contract::DocClaims,
    &dup_scan::DupScan,
    &evidence_paths::EvidencePaths,
    &evidence_paths::InvariantPaths,
    &feature_isolation::FeatureIsolation,
    &file_size::FileSize,
    &frozen_contract::BackendExtension,
    &frozen_contract::FrozenContracts,
    &frozen_contract::ProgramWireFields,
    &frozen_contract::ReadbackRing,
    &gpu_loudness::GpuLoudness,
    &hot_path::BlockingWait,
    &hot_path::InventoryWalk,
    &hot_path::NestedRows,
    &hot_path::OwnedDispatch,
    &hot_path::ReserveArgument,
    &hot_path::UnboundedCache,
    &hot_path::UnboundedRead,
    &hot_path_scan::HotPathScan,
    &hygiene_matrix::HygieneMatrix,
    &lint_hygiene::ExpectHasFix,
    &lint_hygiene::MissingDocsOverride,
    &lint_hygiene::UnsafeBudget,
    &lint_hygiene::UnsafeJustification,
    &lockfile::LockfileClean,
    &op_names::OpNames,
    &parity_testing::ParityTestingIsolated,
    &platform_boundary::PlatformBoundary,
    &platform_docs::PlatformConsumerDocs,
    &proptest_coverage::ProptestCoverage,
    &repo_hygiene::RepoHygiene,
    &repo_hygiene::SingleBacklog,
    &shader_source::ShaderSource,
    &unification::Unification,
    &workspace_build::WorkspaceCheck,
    &workspace_build::WorkspaceClippy,
    &workspace_build::WorkspaceDocs,
    &workspace_build::WorkspaceTests,
];
