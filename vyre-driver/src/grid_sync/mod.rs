//! Grid-sync kernel splitting.
//!
//! Op id: `vyre-driver::grid_sync`. Soundness: `Exact` over the
//! cross-grid barrier contract.
//!
//! ## Why this lives in vyre-driver, not the backend
//!
//! Every backend that lacks a native cooperative whole-grid launch
//! needs the same kernel-split semantics for
//! `Node::Barrier { ordering: GridSync }`: split the program at the
//! barrier, dispatch each segment as its own kernel launch, and
//! re-feed the prior segment's outputs as inputs to the next. The
//! kernel-launch boundary itself is the grid-level fence  -  every
//! prior write becomes globally visible before the next launch reads.
//!
//! Backends route through [`crate::grid_sync::dispatch_with_grid_sync_split`] when
//! [`crate::backend::VyreBackend::supports_grid_sync`] is `false` and the program
//! contains any `Node::Barrier { ordering: GridSync }`. Backends that
//! return `true` emit one kernel and satisfy the barrier device-side.
//!
//! ## Algorithm
//!
//! 1. Walk the program's top-level entry sequence.
//! 2. Each prefix-suffix split at a `Node::Barrier { GridSync }`
//!    becomes one segment.
//! 3. For each segment, build a `Program` with a segment-local buffer
//!    table: buffers read or written by that segment plus passthrough
//!    read-write buffers that must preserve caller-visible storage.
//! 4. Dispatch segments in order, threading live buffers by buffer name
//!    rather than positional output slot. Segment read-only inputs are
//!    assembled from the caller's original bytes or prior segment
//!    outputs; final host-visible output slots are reassembled in the
//!    original program's output declaration order.
//!
//! ## Device-resident variant
//!
//! [`crate::grid_sync::dispatch_with_grid_sync_split_into`] round-trips every live buffer
//! host↔device between each segment and on every fixpoint pass. For a fused
//! multi-rule program whose shared output accumulator is hundreds of MiB and
//! which splits into hundreds of segments, that transfer, not launch
//! latency, dominates wall time. [`crate::grid_sync::dispatch_resident_grid_sync_fixpoint_into`]
//! is the device-resident counterpart: it uploads inputs into backend-resident
//! resources once, keeps them bound across every segment and fixpoint pass (so
//! the accumulator threads in place on-device, since resident dispatch never
//! clears a bound buffer between launches), and reads back only the final
//! outputs. It requires
//! [`crate::backend::VyreBackend::supports_resident_dispatch`]; callers route
//! to it on resident-capable backends and to the host split otherwise.
//! Both paths are recall- and proof-identical (proven by a host/resident
//! differential gate); the choice is purely a host↔device-traffic optimization.
//!
//! ## Soundness
//!
//! - Atomicity preserved: every `atomic_or` that fired in segment N
//!   has flushed to global memory by the time segment N+1 launches  -
//!   backend launch APIs issue an implicit grid-level fence at
//!   submission boundaries.
//! - Ordering preserved: the original program's host-visible output
//!   is byte-identical to the un-split version, modulo timing.
//! - No re-validation surprise: each split segment validates against
//!   the same backend supported-ops set as the original.

use std::collections::{HashMap, HashSet};

use crate::backend::BackendError;

mod barrier_split;
mod host_dispatch;
mod let_propagation;
mod live_buffers;
mod resident_dispatch;
mod segment_buffers;
#[cfg(test)]
mod test_programs;

pub use barrier_split::{contains_grid_sync, split_on_grid_sync, try_split_on_grid_sync};
pub use host_dispatch::{
    dispatch_with_grid_sync_split, dispatch_with_grid_sync_split_into,
    dispatch_with_grid_sync_split_timed, dispatch_with_grid_sync_split_via,
    dispatch_with_grid_sync_split_via_into,
};
pub use resident_dispatch::{
    dispatch_resident_grid_sync_fixpoint_into, dispatch_resident_with_grid_sync_split_timed,
};
pub use segment_buffers::plan_host_grid_sync_segment_programs;

// Split plumbing shared by more than one child module: fallible capacity
// reservation, segment error context, and the timed-dispatch wall clock.

fn reserve_grid_sync_vec<T>(
    vec: &mut Vec<T>,
    capacity: usize,
    field: &'static str,
) -> Result<(), BackendError> {
    crate::allocation::try_reserve_vec_to_capacity(vec, capacity).map_err(|error| {
        BackendError::InvalidProgram {
            fix: format!(
                "Fix: failed to reserve {field} for {capacity} entries during grid-sync dispatch splitting: {error}. Split the program into fewer grid-sync segments or run on a backend with native grid sync."
            ),
        }
    })
}

fn reserve_grid_sync_hash_map<K, V>(
    map: &mut HashMap<K, V>,
    capacity: usize,
    field: &'static str,
) -> Result<(), BackendError>
where
    K: Eq + std::hash::Hash,
{
    map.try_reserve(capacity)
        .map_err(|error| BackendError::InvalidProgram {
            fix: format!(
                "Fix: failed to reserve {field} for {capacity} entries during grid-sync dispatch splitting: {error}. Split the program into fewer grid-sync segments or run on a backend with native grid sync."
            ),
        })
}

fn reserve_grid_sync_hash_set<T>(
    set: &mut HashSet<T>,
    capacity: usize,
    field: &'static str,
) -> Result<(), BackendError>
where
    T: Eq + std::hash::Hash,
{
    set.try_reserve(capacity)
        .map_err(|error| BackendError::InvalidProgram {
            fix: format!(
                "Fix: failed to reserve {field} for {capacity} entries during grid-sync dispatch splitting: {error}. Split the program into fewer grid-sync segments or run on a backend with native grid sync."
            ),
        })
}

fn grid_sync_segment_error(
    error: BackendError,
    segment_idx: usize,
    segment_count: usize,
) -> BackendError {
    match error {
        BackendError::InvalidProgram { fix } => BackendError::InvalidProgram {
            fix: format!(
                "Fix: grid-sync split segment {segment_idx} of {segment_count} dispatch failed: {fix}"
            ),
        },
        other => other,
    }
}

fn elapsed_wall_ns(started: std::time::Instant) -> Result<u64, BackendError> {
    u64::try_from(started.elapsed().as_nanos()).map_err(|error| BackendError::InvalidProgram {
        fix: format!(
            "Fix: grid-sync segmented wall timing cannot fit u64 nanoseconds: {error}. Split telemetry windows or report per-segment timing."
        ),
    })
}
