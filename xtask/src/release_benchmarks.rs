//! Generate long-running release benchmark evidence artifacts.
//!
//! `release-evidence` intentionally avoids expensive benchmark runs.
//! This command is the explicit release path for producing the per
//! workload benchmark JSON artifacts listed by `release-matrix`.

#[path = "release_benchmarks/types.rs"]
mod types;
#[path = "release_benchmarks/args.rs"]
mod args;
#[path = "release_benchmarks/suite_inspect.rs"]
mod suite_inspect;
#[path = "release_benchmarks/optimization.rs"]
mod optimization;
#[path = "release_benchmarks/metrics.rs"]
mod metrics;
#[path = "release_benchmarks/runner.rs"]
mod runner;
#[path = "release_benchmarks/run.rs"]
mod run;

pub(crate) use run::run;
