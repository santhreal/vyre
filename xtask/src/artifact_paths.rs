//! Paths of the release evidence artifacts more than one subcommand names.
//!
//! A writer and its readers must agree on the path, so the literal lives here
//! once instead of being retyped at each end.

/// Leaderboard the frontier benchmark writes and the release gate reads back.
pub const FRONTIER_LEADERBOARD_ARTIFACT: &str =
    "release/evidence/benchmarks/frontier-leaderboard.json";

/// Duplicate registered operations the lego audit records.
pub const REGISTERED_OP_DUPLICATES_ARTIFACT: &str =
    "release/evidence/dedup/registered-op-duplicates.json";
/// Duplicate source families the lego audit records.
pub const LEGO_AUDIT_DUPLICATES_ARTIFACT: &str =
    "release/evidence/dedup/lego-audit-duplicates.json";

/// Exact benchmark evidence artifacts written and audited by `release-benchmarks`.
pub const RELEASE_BENCHMARKS_ARTIFACTS: &[&str] = &[
    "release/evidence/benchmarks/bench-release-axes.json",
    "release/evidence/benchmarks/cpu-only-100x-proof.json",
    "release/evidence/benchmarks/cuda-release-suite.json",
    "release/evidence/benchmarks/dataflow-analysis-release.json",
    FRONTIER_LEADERBOARD_ARTIFACT,
    "release/evidence/benchmarks/megakernel-condition-100x-proof.json",
    "release/evidence/benchmarks/megakernel-condition-cuda.json",
    "release/evidence/benchmarks/megakernel-latency-cuda.json",
    "release/evidence/benchmarks/workload-01-condition-eval.json",
    "release/evidence/benchmarks/workload-02-string-bitmap-scatter.json",
    "release/evidence/benchmarks/workload-03-offset-count-aggregation.json",
    "release/evidence/benchmarks/workload-04-metadata-conditions.json",
    "release/evidence/benchmarks/workload-05-entropy-window.json",
    "release/evidence/benchmarks/workload-06-quantified-condition-loops.json",
    "release/evidence/benchmarks/workload-07-alias-reaching-def.json",
    "release/evidence/benchmarks/workload-08-ifds-witness.json",
    "release/evidence/benchmarks/workload-09-ast-motif-traversal.json",
    "release/evidence/benchmarks/workload-10-megakernel-queued-batches.json",
    "release/evidence/benchmarks/workload-11-semantic-optimizer-impact.json",
    "release/evidence/benchmarks/workload-12-sparse-output-compaction.json",
    "release/evidence/benchmarks/workload-13-callgraph-reachability.json",
    "release/evidence/benchmarks/workload-14-compound-fused-filter.json",
    "release/evidence/benchmarks/workload-15-adaptive-routing.json",
    "release/evidence/benchmarks/workload-16-quantized-linear.json",
    "release/evidence/benchmarks/workload-17-egraph-saturation.json",
];
