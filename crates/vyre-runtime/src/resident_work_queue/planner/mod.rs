//! Resident work-queue sizing, launch geometry, and work-item contracts.

mod barriers;
mod caps;
mod geometry;
mod grid;
mod sizing;

pub use barriers::elide_value_flow_barriers;
pub use caps::ResidentWorkItem;
pub use geometry::{
    dispatch_grid_for, padded_slot_count, worker_workgroup_size, ResidentLaunchGeometry,
};
pub use grid::{ResidentGridLimits, ResidentGridPlan, ResidentGridRequest};
pub use sizing::ResidentSizingPolicy;
