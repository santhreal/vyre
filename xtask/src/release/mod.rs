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
pub mod release_gate;
pub mod release_train;
pub mod release_workload_matrix;
pub mod repo_boundary;
pub mod version_matrix;
