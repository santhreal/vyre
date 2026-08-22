//! Reduction primitives  -  `count`/`min`/`max`/`sum` over
//! bitsets and fixed-width u32 ValueSets.
//!
//! Scalar reductions use one grid-stride workgroup and global atomics
//! so the baseline primitive is parallel instead of serial lane-0
//! scaffolding. Higher-level workgroup-tree reductions still compose
//! these where a caller needs per-workgroup partials or f32 support.

/// `reduce_all` - emit `1` when every lane in a u32 ValueSet is non-zero.
pub mod all;
/// `reduce_any` - emit `1` when any lane in a u32 ValueSet is non-zero.
pub mod any;
/// The grid-stride atomic-scalar shape, shared with the bitset relations.
pub(crate) mod atomic_scalar;
pub mod count;
pub mod count_non_zero;
pub mod gather;
pub mod grid_stride_tree;
pub mod histogram;
mod indexed_move;
/// Unsigned maximum over a u32 ValueSet.
pub mod max;
/// Unsigned minimum over a u32 ValueSet.
pub mod min;
pub mod multi_block_prefix_scan;
pub mod radix_sort;
pub mod range_counts;
/// GPU reduction metrics for self-substrate scheduling and telemetry.
pub mod reduction_metrics;
pub mod scatter;
pub mod segment_reduce;
pub mod sum;
pub mod workgroup_any;
// Crate-private: the sweep and its pass are `pub(crate)` composition
// internals, so a `pub mod` here would add a module to the published surface
// with nothing in it.
pub(crate) mod workgroup_scan;
pub mod workgroup_tree;
