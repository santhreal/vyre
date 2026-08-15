//! Program builders surfaced through the megakernel planner.

use vyre_primitives::math::bellman_shortest_path::{BellmanBuffers, BellmanExtents};
use vyre_primitives::math::sinkhorn_iterate::{SinkhornBuffers, SinkhornExtents};

/// Full Sinkhorn-balanced clustering Program builder. Wraps
/// [`vyre_libs::solvers::sinkhorn_full_clustering::sinkhorn_full_clustering_program`]
/// for callers that need the full iterative-balance variant rather than the
/// dispatch-clustering simplification.
///
/// The binding record and the extents are the primitive's own types, so a change
/// to either reaches this wrapper as a type error rather than as a silently
/// reordered argument list.
#[must_use]
pub fn build_sinkhorn_full_clustering_program(
    buffers: SinkhornBuffers<'_>,
    extents: SinkhornExtents,
) -> vyre_foundation::ir::Program {
    vyre_libs::solvers::sinkhorn_full_clustering::sinkhorn_full_clustering_program(buffers, extents)
}

/// Build a multi-word scallop-provenance Program. Wraps
/// [`vyre_libs::encoding::scallop_provenance_wide::scallop_provenance_wide_program`]
/// for >32-rule lineage tracking (W=8 → 256 rules max).
#[must_use]
pub fn build_scallop_provenance_wide_program(
    state: &str,
    next: &str,
    join_rules: &str,
    changed: &str,
    n: u32,
    w: u32,
    max_iterations: u32,
) -> vyre_foundation::ir::Program {
    vyre_libs::encoding::scallop_provenance_wide::scallop_provenance_wide_program(
        state,
        next,
        join_rules,
        changed,
        n,
        w,
        max_iterations,
    )
}
/// Bellman tensor-network ordering Program builder. Wraps
/// [`vyre_libs::solvers::bellman_tn_order::bellman_tn_order_program`].
///
/// `n_nodes` and `n_edges` are named by [`BellmanExtents`], and the buffer names
/// by [`BellmanBuffers`], which is what makes the pair unorderable by mistake.
#[must_use]
pub fn build_bellman_tn_order_program(
    buffers: BellmanBuffers<'_>,
    extents: BellmanExtents,
) -> vyre_foundation::ir::Program {
    vyre_libs::solvers::bellman_tn_order::bellman_tn_order_program(buffers, extents)
}

/// KFAC autotune-step Program builder. Wraps
/// [`vyre_libs::solvers::kfac_autotune_step::kfac_autotune_step_program`].
#[must_use]
pub fn build_kfac_autotune_step_program(
    blocks_out: &str,
    blocks_in: &str,
    scratch: &str,
    num_blocks: u32,
    n: u32,
) -> vyre_foundation::ir::Program {
    vyre_libs::solvers::kfac_autotune_step::kfac_autotune_step_program(
        blocks_out, blocks_in, scratch, num_blocks, n,
    )
}

/// Build a sinkhorn dispatch-clustering Program. Wraps
/// [`vyre_libs::solvers::sinkhorn_dispatch_clustering::sinkhorn_clustering_program`].
#[must_use]
pub fn build_sinkhorn_clustering_program(
    m: u32,
    n: u32,
    d: u32,
    iters: u32,
    eps: f32,
) -> vyre_foundation::ir::Program {
    vyre_libs::solvers::sinkhorn_dispatch_clustering::sinkhorn_clustering_program(
        m, n, d, iters, eps,
    )
}
/// Build a persistent-fixpoint Program around a caller-supplied
/// transfer body. Replaces a host-side `loop { dispatch(); check }`
/// pattern with a single GPU-side dispatch that ping-pongs
/// `current ↔ next` until `changed[0] == 0` or `max_iterations`.
///
/// P-DRIVER-9: every host fixpoint loop should migrate to this
/// substrate Program. Caller supplies `transfer_body` that reads
/// `current` and writes `next`; the wrapper handles the convergence
/// flag and ping-pong copy.
#[must_use]
pub fn build_persistent_fixpoint_program(
    transfer_body: Vec<vyre_foundation::ir::Node>,
    current: &str,
    next: &str,
    changed: &str,
    words: u32,
    max_iterations: u32,
) -> vyre_foundation::ir::Program {
    vyre_libs::analysis::persistent_fixpoint_program::persistent_fixpoint_program(
        transfer_body,
        current,
        next,
        changed,
        words,
        max_iterations,
    )
}
