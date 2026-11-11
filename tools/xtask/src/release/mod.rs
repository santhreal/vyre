//! The subcommands that produce release evidence and close the release.
//!
//! Everything here either writes an artifact under `release/evidence/` or
//! reads one back and decides whether the release may proceed, together with
//! the release manifests (`release/release-train.toml`,
//! `release/repo-boundary.toml`) those decisions are made against.

pub mod conformance_evidence_semantics;
pub mod conformance_op_matrix;
pub mod conformance_workflows;
pub mod feature_matrix;
pub mod launch_contract;
pub mod launch_state;
pub mod metadata_matrix;
pub mod package_readiness;
pub mod release_conformance;
pub mod release_docs;
pub mod release_train;
pub mod repo_boundary;
pub mod version_matrix;

use crate::gate::GateBehavior;

/// Every release gate behavior implemented in this module.
pub static GATES: &[(&str, &dyn GateBehavior)] = &[
    ("feature-matrix", &feature_matrix::FeatureMatrixGate),
    ("launch-state", &launch_state::LaunchStateGate),
    ("metadata-matrix", &metadata_matrix::MetadataMatrixGate),
    (
        "package-readiness",
        &package_readiness::PackageReadinessGate,
    ),
    (
        "release-conformance",
        &release_conformance::ReleaseConformanceGate,
    ),
    ("release-docs", &release_docs::ReleaseDocs),
    ("version-matrix", &version_matrix::VersionMatrixGate),
];
