//! Graph traversal, dominance, and dispatch-pipeline compositions.
//!
//! These modules wire `vyre-primitives::graph` programs into
//! self-substrate dispatch, scratch, evidence, and observability contracts.
//! Primitive graph logic stays in `vyre-primitives`; this module owns only
//! self-hosting integration.

pub mod adaptive_traverse;
pub mod alias_registry;
/// Parity-only host dispatcher, never part of the shipped dispatch surface.
///
/// Vyre executes on a device. The only host execution this workspace admits is
/// a reference oracle used as the comparison arm of a parity test, so this
/// module is absent from a default build: an ungated `pub mod` let any consumer
/// of this crate construct a host `ProgramDispatcher` and run a Program off the
/// device. Every caller is either an in-crate `#[cfg(test)]` module or an
/// integration test whose `[[test]]` row already declares `cpu-parity`, so the
/// gate costs nothing and `vyre-libs/tests/host_dispatch_is_parity_only.rs`
/// keeps it that way.
pub mod csr_bidirectional;
pub mod csr_forward_or_changed;
pub mod csr_frontier_queue_batch_memory;
pub mod csr_frontier_queue_batch_resident;
pub(crate) mod csr_frontier_queue_programs;
pub mod csr_frontier_queue_resident;
pub(crate) mod csr_frontier_queue_scratch;
pub(crate) mod dispatch_bridge;
pub mod dominator_frontier;
pub mod exploded;
pub mod level_wave_pass;
pub mod motif;
pub mod path_reconstruct;
pub mod persistent_bfs;
pub(crate) mod plan_cache;
pub(crate) mod resident_handles;
pub mod structural_kernel_pipeline;
pub mod toposort;
pub mod traversal_dispatch_pipeline;
pub mod union_find_emit;
pub mod vast_tree_walk;
