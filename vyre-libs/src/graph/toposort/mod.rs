//! Kahn-style topological sort with LIFO worklist  -  CPU reference +
//! single-invocation GPU `Program` builder.
//!
//! Consumed by optimizer reaching-defs, dominator-tree, and graph-IR analyses
//! that need a DAG walk.
//!
//! AUDIT_2026-04-24 F-TS-04: `toposort_program` emits a single-invocation
//! Program that runs Kahn's algorithm serially on lane 0. The serial
//! lane-0 builder is the current Tier-2.5 contract because topological
//! ordering has a loop-carried dependency; callers that need large-DAG
//! throughput compose this with graph partitioning or SCC batching.
//!
//! AUDIT_2026-04-24 F-TS-02: the classical Kahn presentation uses a
//! FIFO queue (BFS-ish). This module uses a stack (LIFO) via
//! `Vec::pop` because it is O(1), has better cache locality on the
//! worklist, and produces an equally valid topological order  -  both
//! orderings satisfy the Kahn invariant (a node is emitted only
//! after all its prerequisites). If a caller needs stable BFS order
//! for deterministic diffs, swap in a `VecDeque` worklist; the
//! correctness of the sort does not depend on the worklist policy.

mod csr;
mod edge_list;
mod error;
mod plan;
mod program;

pub use csr::{
    toposort_csr, toposort_csr_into, toposort_csr_into_with_scratch,
    validate_toposort_csr_inputs, validate_toposort_csr_order, ToposortCsrLayout,
    ToposortCsrScratch,
};
pub use edge_list::toposort;
pub use error::{ToposortCsrError, ToposortError};
pub use plan::{
    plan_toposort_csr_dispatch, toposort_csr_slice_fingerprint, ToposortCsrDispatchPlan,
    ToposortCsrStaticInputKey,
};
pub use program::toposort_program;

/// Canonical op id.
pub const OP_ID: &str = "vyre-primitives::graph::toposort";
/// Canonical dispatch input label for CSR offsets.
pub const TOPOSORT_OFFSETS_BUFFER: &str = "toposort offsets";
/// Canonical dispatch input label for CSR targets.
pub const TOPOSORT_TARGETS_BUFFER: &str = "toposort targets";
/// Canonical dispatch scratch label for indegrees.
pub const TOPOSORT_INDEGREE_SCRATCH_BUFFER: &str = "toposort indeg_scratch";
/// Canonical dispatch scratch label for the work queue.
pub const TOPOSORT_QUEUE_SCRATCH_BUFFER: &str = "toposort queue_scratch";
/// Canonical dispatch output label for the emitted order.
pub const TOPOSORT_ORDER_OUT_BUFFER: &str = "toposort order_out";
/// Single-lane Kahn dispatch grid.
pub const TOPOSORT_DISPATCH_GRID: [u32; 3] = [1, 1, 1];
