//! Graph traversal, dominance, and dispatch-pipeline compositions.
//!
//! These modules wire `crate::graph` programs into
//! self-substrate dispatch, scratch, evidence, and observability contracts.
//! Primitive graph logic stays in `crate::graph`; this module owns only
//! self-hosting integration.

pub mod adaptive_traverse;
pub mod csr_bidirectional;
pub mod csr_forward_or_changed;
pub mod csr_frontier_queue_batch_resident;
pub mod csr_frontier_queue_resident;
pub(crate) mod dispatch_bridge;
pub mod dominator_frontier;
pub mod exploded;
pub(crate) mod frontier;

pub mod motif;
pub mod path_reconstruct;
pub mod persistent_bfs;
pub(crate) mod plan_cache;
pub(crate) mod resident_handles;
pub(crate) mod structural_kernel_pipeline;
pub mod toposort;
pub(crate) mod traversal_dispatch_pipeline;
pub mod union_find_emit;
