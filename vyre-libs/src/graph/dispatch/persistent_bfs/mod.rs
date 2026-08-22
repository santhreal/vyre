//! Multi-step BFS frontier expansion substrate consumer.
//!
//! Wires `crate::graph::persistent_bfs` so the optimizer can
//! compute multi-step reachability in a single primitive call instead
//! of looping `csr_forward_traverse` by hand. The primitive accumulates
//! into `frontier_out` via OR and reports a sticky changed-flag, so the
//! caller knows whether any new nodes were added across all steps.

mod dispatch;
mod resident;
mod resident_scratch;
#[cfg(test)]
#[path = "../../../../tests/internal/graph/dispatch/persistent_bfs/mod.rs"]
mod tests;

pub use dispatch::*;
pub use resident::*;
pub use resident_scratch::{
    PersistentBfsGpuScratch, PersistentBfsPlanCacheSnapshot, PersistentBfsResidentScratch,
    ResidentBfsGraph,
};

#[cfg(test)]
use resident::ensure_resident_query_handles;
