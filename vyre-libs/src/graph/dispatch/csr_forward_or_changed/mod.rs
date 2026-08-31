//! In-place expand-with-change-flag substrate consumer.
//!
//! Wires `crate::graph::csr_forward_or_changed` so iterative
//! dataflow loops can detect convergence in a single pass: the primitive returns the next
//! frontier AND a boolean changed-flag. Used by reachability /
//! liveness / reaching-defs fixpoint passes that previously had to
//! diff before/after states by hand.

mod dispatch;

pub use dispatch::{
    forward_closure_via_change_flag_gpu, forward_closure_via_change_flag_gpu_into,
    forward_closure_via_change_flag_gpu_with_scratch_into, ForwardChangedGpuScratch,
};

#[cfg(test)]
#[path = "../../../../tests/internal/graph/dispatch/csr_forward_or_changed/mod.rs"]
mod tests;
