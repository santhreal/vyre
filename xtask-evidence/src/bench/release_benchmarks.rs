//! Generate long-running release benchmark evidence artifacts.
//!
//! `release-evidence` intentionally avoids expensive benchmark runs.
//! This command is the explicit release path for producing the per
//! workload benchmark JSON artifacts listed by `release-matrix`.

#[path = "release_benchmarks/args.rs"]
mod args;
#[path = "release_benchmarks/cpu_sota_proof.rs"]
mod cpu_sota_proof;
#[path = "release_benchmarks/evidence_schema.rs"]
mod evidence_schema;
#[path = "release_benchmarks/frontier_leaderboard.rs"]
mod frontier_leaderboard;
#[path = "release_benchmarks/inspect_core.rs"]
mod inspect_core;
#[path = "release_benchmarks/metrics.rs"]
mod metrics;
#[path = "release_benchmarks/optimization.rs"]
mod optimization;
#[path = "release_benchmarks/release_thresholds.rs"]
mod release_thresholds;
#[path = "release_benchmarks/run.rs"]
mod run;
#[path = "release_benchmarks/runner.rs"]
mod runner;
#[path = "release_benchmarks/suite_inspect.rs"]
mod suite_inspect;

pub(crate) use frontier_leaderboard::{
    frontier_leaderboard_required_artifact_fields, validate_frontier_leaderboard_artifact_bytes,
    FRONTIER_LEADERBOARD_SCHEMA_VERSION, FRONTIER_LEADERBOARD_SEMANTIC_VALIDATOR,
};
pub use run::ReleaseBenchmarksGate;
