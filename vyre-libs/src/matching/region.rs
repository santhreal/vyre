//! Span-region dedup primitive.
//!
//! Every multimatch consumer (`vyre-libs::matching` engines, scanner consumer,
//! external analyzer) ends up doing the same operation after the GPU dispatch
//! returns: take the raw `Vec<ByteRange>`, collapse adjacent overlapping
//! or duplicate spans into a representative, return the deduped set.
//! Each consumer wrote it differently  -  some by `(detector_id,
//! credential)` HashMap, some by `(start, end)` pair sort, some by ad-
//! hoc loop. The lego-block fix is one primitive every consumer calls.
//!
//! # Algorithm
//!
//! Given a slice of `(pid, start, end)` triples sorted by `(pid, start, end)`,
//! emit one representative per maximal cluster of triples that
//! overlap or touch (`start[i] <= end[max_end_so_far]`) AND have the same
//! `pid`. This collapses both:
//!
//!   - `(pid=0, 5, 10)` and `(pid=0, 6, 11)` → `(pid=0, 5, 11)`
//!     (overlapping, same pattern  -  extend span).
//!   - `(pid=0, 5, 10)` and `(pid=0, 5, 10)` → one entry
//!     (exact dup).
//!
//! Distinct `pid`s never merge  -  two patterns matching the same
//! region produce two output spans (cross-pattern dedup is a
//! different operation; consumers that want it apply a second pass).
//!
//! # CPU + GPU
//!
//! - `dedup_regions_cpu` is the reference implementation: pure data,
//!   no IR, no backend. CPU-side consumers and parity tests use it.
//! - `region_sort_program` and `dedup_regions_cluster_program` emit
//!   GPU-resident sorted spans, survivor flags, and merged cluster ends
//!   so parser/scanner pipelines can compact deduped triples without a
//!   host readback between stages.
//!
//! Both share a single golden test fixture set so any divergence is
//! caught at conform time.

use std::cmp::Ordering;

pub use super::region_programs::{
    cap_regions_per_pattern_flag_program, compact_first_per_region_pattern_flag_program,
    dedup_regions_cluster_program, dedup_regions_flag_program, region_dedup_dispatch_grid,
    region_sort_program, CAP_REGIONS_PER_PATTERN_OP_ID, COMPACT_FIRST_PER_REGION_PATTERN_OP_ID,
    DEDUP_REGIONS_CLUSTER_OP_ID, DEDUP_REGIONS_FLAG_OP_ID, REGION_DEDUP_WORKGROUP_SIZE,
};

/// Operation-local region triple used by sort and dedup kernels.
///
/// Product boundaries use `vyre_foundation::match_result::ByteRange`; this type
/// retains `pid` because the region ABI operates directly on packed pattern ids.
///
/// `pid`: pattern id; `start` / `end`: byte offsets, half-open
/// `[start, end)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionTriple {
    /// Pattern id (which detector emitted this match).
    pub pid: u32,
    /// Inclusive start byte offset.
    pub start: u32,
    /// Exclusive end byte offset.
    pub end: u32,
}

impl RegionTriple {
    /// Construct a region triple. `end` must be `>= start`; equal
    /// values represent a zero-width match (legal for some regex
    /// constructs).
    #[must_use]
    pub const fn new(pid: u32, start: u32, end: u32) -> Self {
        Self { pid, start, end }
    }
}

impl Ord for RegionTriple {
    fn cmp(&self, other: &Self) -> Ordering {
        // Sort by (pid, start, end) so the dedup loop sees cluster
        // members consecutively without a secondary group-by pass.
        self.pid
            .cmp(&other.pid)
            .then(self.start.cmp(&other.start))
            .then(self.end.cmp(&other.end))
    }
}

impl PartialOrd for RegionTriple {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
