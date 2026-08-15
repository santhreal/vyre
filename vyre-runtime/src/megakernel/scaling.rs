//! Occupancy-aware resident work-queue scaling.

#[cfg(feature = "libs-compositions")]
pub use super::planner::{
    build_bellman_tn_order_program, build_kfac_autotune_step_program,
    build_persistent_fixpoint_program, build_sinkhorn_clustering_program,
};
pub use super::planner::{
    default_worker_groups_from_limits, dispatch_grid_for, padded_slot_count, worker_workgroup_size,
    ResidentGridLimits, ResidentGridPlan, ResidentGridRequest, ResidentLaunchGeometry,
    ResidentSizingPolicy,
};
#[cfg(test)]
pub use super::policy::{diffuse_priority_across_siblings, diffuse_priority_across_siblings_into};
pub use super::policy::{
    try_diffuse_priority_across_siblings, try_diffuse_priority_across_siblings_into,
    PriorityRequeueAccounting, ResidentExecutionMode, ResidentLaunchPolicy,
    ResidentLaunchRecommendation, ResidentLaunchRequest, ResidentQueuePressure,
};
