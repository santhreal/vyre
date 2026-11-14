//! Megakernel launch geometry helpers.

use super::grid::cached_geometry_from_slots;
use super::sizing::ResidentSizingPolicy;

/// Host-side launch geometry for a finite megakernel dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentLaunchGeometry {
    /// Lanes per worker workgroup used to compile the program.
    pub workgroup_size_x: u32,
    /// Ring slots allocated for the dispatch, padded to a full workgroup.
    pub slot_count: u32,
    /// Grid submitted to the backend.
    pub dispatch_grid: [u32; 3],
}

impl ResidentLaunchGeometry {
    /// Build geometry for an already-sized ring.
    #[must_use]
    pub fn from_slots(slot_count: u32, worker_count: u32, max_workgroup_size_x: u32) -> Self {
        cached_geometry_from_slots(slot_count, worker_count, max_workgroup_size_x)
    }

    /// Number of worker workgroups needed to cover every ring slot exactly once.
    #[must_use]
    pub const fn covering_worker_groups(&self) -> u32 {
        self.slot_count / self.workgroup_size_x
    }
}

/// Clamp the caller's worker setting into the legal x dimension used by the
/// current megakernel ABI.
#[must_use]
pub fn worker_workgroup_size(worker_count: u32, max_workgroup_size_x: u32) -> u32 {
    ResidentSizingPolicy::standard().worker_workgroup_size(worker_count, max_workgroup_size_x)
}

/// Round a logical slot count up to a whole workgroup.
#[must_use]
pub fn padded_slot_count(slot_count: u32, workgroup_size_x: u32) -> u32 {
    ResidentSizingPolicy::standard().padded_slot_count(slot_count, workgroup_size_x)
}

/// Compute the backend dispatch grid for a logical queue length.
#[must_use]
pub fn dispatch_grid_for(worker_count: u32, queue_len: u32, max_workgroup_size_x: u32) -> [u32; 3] {
    ResidentSizingPolicy::standard().dispatch_grid_for(
        worker_count,
        queue_len,
        max_workgroup_size_x,
    )
}
