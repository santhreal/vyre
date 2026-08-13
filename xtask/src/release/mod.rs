//! The subcommands that produce release evidence and close the release.
//!
//! Everything here either writes an artifact under `release/evidence/` or
//! reads one back and decides whether the release may proceed, together with
//! the release manifests (`release/release-train.toml`,
//! `release/repo-boundary.toml`) those decisions are made against.

pub(crate) mod backend_matrix;
pub(crate) mod conformance_evidence_semantics;
pub(crate) mod conformance_matrix;
pub(crate) mod feature_matrix;
pub(crate) mod launch_contract;
pub(crate) mod launch_state;
pub(crate) mod metadata_matrix;
pub(crate) mod optimization_corpus;
pub(crate) mod optimization_matrix;
pub(crate) mod package_readiness;
pub(crate) mod release_backend_rows;
pub(crate) mod release_conformance;
pub(crate) mod release_evidence;
pub(crate) mod release_gate;
pub(crate) mod release_train;
pub(crate) mod release_workload_matrix;
pub(crate) mod repo_boundary;
pub(crate) mod version_matrix;
pub(crate) mod vyre_release_gate;
