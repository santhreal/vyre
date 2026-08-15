//! Resident work-queue sizing, launch geometry, and work-item contracts.

mod barriers;
mod caps;
mod config;
mod geometry;
mod grid;
#[cfg(feature = "libs-compositions")]
mod programs;
mod sizing;

pub use barriers::{elide_value_flow_barriers, BarrierElisionReport};
pub use caps::{
    ResidentQueueCapabilities, ResidentQueueReport, ResidentQueueTelemetry, ResidentWorkItem,
};
pub use config::{ResidentQueueConfig, ResidentWorkloadHints};
pub use geometry::{
    default_worker_groups_from_limits, dispatch_grid_for, padded_slot_count, worker_workgroup_size,
    ResidentLaunchGeometry,
};
pub use grid::{ResidentGridLimits, ResidentGridPlan, ResidentGridRequest};
#[cfg(feature = "libs-compositions")]
pub use programs::{
    build_bellman_tn_order_program, build_kfac_autotune_step_program,
    build_persistent_fixpoint_program, build_scallop_provenance_wide_program,
    build_sinkhorn_clustering_program, build_sinkhorn_full_clustering_program,
};
pub use sizing::ResidentSizingPolicy;

#[cfg(test)]
use super::task::TaskWorkItem;

#[cfg(test)]
mod tests;
