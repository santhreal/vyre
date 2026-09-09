//! `persistent_bfs` - on-device multi-step BFS frontier expansion.
//!
//! The kernel copies `frontier_in` into `frontier_out`, then performs up to
//! `max_iters` forward traversal steps, accumulating reachable nodes into
//! `frontier_out` via atomic OR.  The first `min(max_iters, 4)` iterations
//! are unrolled and use a workgroup-local `wg_scratch` buffer to coalesce
//! per-workgroup change detection between steps.

mod dispatch_plan;
mod hash;
mod layout;
mod plan;
mod program;
#[cfg(test)]
#[path = "../../../tests/internal/graph/persistent_bfs/reference_adapter.rs"]
mod reference_adapter;
mod resident_plan;
mod validate;

mod registry;

#[cfg(test)]
#[path = "../../../tests/internal/graph/persistent_bfs/mod.rs"]
mod tests;

pub use hash::{persistent_bfs_layout_hash, persistent_bfs_program_layout_hash};
pub use layout::{
    PersistentBfsPlanCacheKey, PersistentBfsStaticInputKey, BATCH_OP_ID, BINDING_CHANGED,
    BINDING_CONVERGED, BINDING_DENSITY_ACTIVE, BINDING_FRONTIER_IN, BINDING_FRONTIER_OUT,
    DENSITY_ACTIVE_BUFFER, OP_ID, PERSISTENT_BFS_WORKGROUP_SIZE,
};
pub use plan::{
    copy_persistent_bfs_batch_seed_and_clear_changed_into, copy_persistent_bfs_seed_frontier_into,
    plan_persistent_bfs_dispatch, plan_persistent_bfs_resident_batch_dispatch,
    plan_persistent_bfs_resident_dispatch, validate_persistent_bfs_changed_flag,
    validate_persistent_bfs_converged_flag,
};
pub use program::{
    persistent_bfs, persistent_bfs_batch, persistent_bfs_batch_with_density,
    persistent_bfs_with_density, try_persistent_bfs_batch, try_persistent_bfs_batch_with_density,
};
pub use validate::{
    validate_persistent_bfs_batch_frontiers, validate_persistent_bfs_frontier,
    validate_persistent_bfs_graph_layout, validate_persistent_bfs_inputs,
};

#[cfg(test)]
pub(crate) use {
    layout::{
        PersistentBfsBatchLayout, PersistentBfsFrontierLayout, PersistentBfsLayout,
        PersistentBfsPlanCacheKind,
    },
    reference_adapter::{
        cpu_ref, cpu_ref_into, try_cpu_ref, try_cpu_ref_converged, try_cpu_ref_density,
        try_cpu_ref_into, try_cpu_ref_into_with_scratch, PersistentBfsConvergence,
        PersistentBfsCpuScratch,
    },
};
