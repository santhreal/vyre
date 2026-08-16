//! The numbers a release benchmark suite is held to.
//!
//! Each is a release requirement rather than a tuning knob, so they are stated
//! here once and read by every check that enforces one.

pub(super) const REQUIRED_CPU_SOTA_100X_CASES: &[&str] = &[
    "release.condition_eval.1m",
    "release.string_bitmap_scatter.1m",
    "release.offset_count_aggregation.1m",
    "release.entropy_window.1m",
    "release.quantified_condition_loops.1m",
    "release.alias_reaching_def.1m",
    "release.ifds_witness.1m",
    "release.ast_motif_traversal.1m",
    "release.megakernel_queue.1m",
    "sparse.compaction.count.1m",
];
pub(super) const MIN_CPU_SOTA_100X_RELEASE_CASES: usize = 10;
pub(super) const MAX_RELEASE_BENCHMARK_TEXT_BYTES: u64 = 256 * 1024 * 1024;
pub(super) const MIN_CUDA_RELEASE_MEMORY_MIB: u64 = 16 * 1024;
pub(super) const MIN_CUDA_RELEASE_COMPUTE_CAPABILITY_MAJOR: u64 = 8;
pub(super) const MIN_CUDA_RELEASE_COMPUTE_CAPABILITY_MINOR: u64 = 0;
