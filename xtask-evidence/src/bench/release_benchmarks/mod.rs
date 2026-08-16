//! Generate long-running release benchmark evidence artifacts.
//!
//! `release-evidence` intentionally avoids expensive benchmark runs.
//! This command is the explicit release path for producing the per
//! workload benchmark JSON artifacts listed by `release-matrix`.

mod args;
mod cpu_sota_proof;
mod evidence_schema;
mod frontier_leaderboard;
mod artifact_metrics;
mod metrics;
mod optimization;
mod release_thresholds;
mod run;
mod runner;
mod suite_inspect;

pub(crate) use frontier_leaderboard::{
    frontier_leaderboard_required_artifact_fields, validate_frontier_leaderboard_artifact_bytes,
    FRONTIER_LEADERBOARD_SCHEMA_VERSION, FRONTIER_LEADERBOARD_SEMANTIC_VALIDATOR,
};
pub(crate) use run::ReleaseBenchmarksGate;
