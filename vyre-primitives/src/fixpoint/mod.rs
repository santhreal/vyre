//! Tier 2.5 fixpoint primitives  -  driver-free convergence loops for
//! bitset transfer functions.
//!
//! The vision's taint/flow semantics all reduce to "iterate a
//! bitset transfer function until the output bitset stops growing."
//! This module packages that pattern as a single primitive:
//!
//! - `bitset_fixpoint`  -  canonical ping-pong with a convergence
//!   flag. One Program that the backend dispatches repeatedly; the
//!   harness / runtime loops until the flag clears or
//!   `max_iterations` is hit.
//! - `persistent_fixpoint`  -  single-dispatch convergence on the GPU.
//!   Wraps a caller-supplied transfer-step body in a bounded in-kernel
//!   loop with the comparison + ping-pong + termination check inside
//!   the kernel. Replaces every "host iterates to fixpoint"
//!   docstring; convergence happens entirely on device. Pick this when
//!   the state fits ONE workgroup
//!   (`words <= PERSISTENT_FIXPOINT_WORKGROUP_SIZE[0]`): with a single
//!   group its workgroup-scope barriers are incidentally grid-wide, so
//!   its early exit is sound. It is UNSOUND above one workgroup
//!   because lane 0's plain clear of the single shared `changed` word
//!   races with other groups' `atomic_or` of it, and the group whose
//!   set is erased returns early with unconverged state.
//! - `persistent_fixpoint_grid`  -  the grid-correct sibling. Pick this
//!   when the state does NOT fit one workgroup. Emits top-level waves
//!   separated by `MemoryOrdering::GridSync` barriers instead of an
//!   in-kernel loop, and keeps a COLLECTIVE early exit by giving
//!   `changed` one never-cleared word per iteration. Costs a wider
//!   `changed` buffer (`max_iterations` words, caller-zeroed) and a
//!   cooperative-residency ceiling on the launch geometry.

pub mod bitset_fixpoint;
pub mod persistent_fixpoint;

/// The one assertion of the routing contract every routed convergence op obeys.
#[cfg(test)]
pub(crate) mod routing_contract;
