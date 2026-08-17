//! Resident adaptive sparse/dense graph traversal.
//!
//! This module wires `reduce_count` and
//! `graph::adaptive_traverse::adaptive_sparse_dense_step` into resident
//! device-ready sequences. Traversal semantics stay in
//! `graph::adaptive_traverse`, which is also the public path to the mode and
//! kernel selectors; this facade owns resident scratch and layout identity.

#[cfg(any(test, feature = "cpu-parity"))]
mod reference;
mod resident;
mod resident_scratch;
mod resident_steps;
#[cfg(test)]
#[path = "../../../../tests/internal/graph/dispatch/adaptive_traverse/mod.rs"]
mod tests;
mod upload;

use crate::graph::adaptive_traverse::AdaptiveTraversalMode;
#[cfg(test)]
use crate::graph::adaptive_traverse::{
    select_adaptive_traversal_mode, select_dense_traversal_kernel, DenseTraversalKernel,
};
#[cfg(any(test, feature = "cpu-parity"))]
pub use reference::*;
pub use resident::{
    ResidentAdaptiveFourRussiansDenseGraph, ResidentAdaptiveSparseQueueGraph,
    ResidentAdaptiveTraversalGraph,
};
pub use resident_scratch::{AdaptiveTraversalPlanCacheSnapshot, AdaptiveTraversalResidentScratch};
pub use resident_steps::*;
pub use upload::*;
