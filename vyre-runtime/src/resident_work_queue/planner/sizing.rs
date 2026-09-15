use vyre_driver::BackendError;
use vyre_foundation::execution_plan::SchedulingPolicy;

use super::{ResidentGridLimits, ResidentGridPlan, ResidentGridRequest, ResidentLaunchGeometry};

/// Worker-grid realization for megakernel dispatch.
///
/// Every value here is arithmetic over a count the caller states and a limit the
/// adapter reports: workgroup width, slot padding, and the backend grid. Which
/// worker count to run is a schedule decision `vyre-megakernel` records in the
/// artifact, and this policy realizes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResidentSizingPolicy {
    scheduling: SchedulingPolicy,
}

impl Default for ResidentSizingPolicy {
    fn default() -> Self {
        Self::standard()
    }
}

impl ResidentSizingPolicy {
    /// Standard megakernel sizing policy used by built-in dispatch paths.
    #[must_use]
    pub const fn standard() -> Self {
        Self {
            scheduling: SchedulingPolicy::standard(),
        }
    }

    /// Build from a shared backend-neutral scheduling policy.
    #[must_use]
    pub const fn from_scheduling(scheduling: SchedulingPolicy) -> Self {
        Self { scheduling }
    }

    /// Default persistent worker workgroup count.
    #[must_use]
    pub const fn default_worker_count(&self) -> u32 {
        self.scheduling.default_worker_count()
    }

    /// Clamp a requested worker count into the legal workgroup x dimension.
    #[must_use]
    pub const fn worker_workgroup_size(&self, worker_count: u32, max_workgroup_size_x: u32) -> u32 {
        self.scheduling
            .worker_workgroup_size(worker_count, max_workgroup_size_x)
    }

    /// Round a logical slot count up to a whole worker workgroup.
    #[must_use]
    pub const fn padded_slot_count(&self, slot_count: u32, workgroup_size_x: u32) -> u32 {
        self.scheduling
            .padded_slot_count(slot_count, workgroup_size_x)
    }

    /// Compute the backend dispatch grid for a logical queue length.
    #[must_use]
    pub const fn dispatch_grid_for(
        &self,
        worker_count: u32,
        queue_len: u32,
        max_workgroup_size_x: u32,
    ) -> [u32; 3] {
        self.scheduling
            .dispatch_grid_for(worker_count, queue_len, max_workgroup_size_x)
    }

    /// Resolve worker groups, workgroup width, slot padding, and dispatch grid.
    ///
    /// `request.requested_worker_groups` is the count the selected schedule
    /// states, clamped to what the adapter admits. A zero count is rejected
    /// rather than replaced: deriving one from occupancy here would run a
    /// worker grid no artifact identity covers.
    ///
    /// # Errors
    ///
    /// Returns [`BackendError`] when adapter limits are malformed or the request
    /// states no worker count.
    pub fn resolve_grid(
        &self,
        request: ResidentGridRequest,
        limits: ResidentGridLimits,
    ) -> Result<ResidentGridPlan, BackendError> {
        limits.validate()?;

        if request.requested_worker_groups == 0 {
            return Err(BackendError::new(
                "resident worker-grid request states no worker count. Fix: pass the worker-group count the selected schedule records.",
            ));
        }
        let worker_groups = request
            .requested_worker_groups
            .min(limits.max_compute_workgroups_per_dimension)
            .max(1);

        let geometry = self.geometry_from_slots(
            request.queue_len.max(1),
            worker_groups,
            limits.max_workgroup_size_x,
        );

        Ok(ResidentGridPlan {
            geometry,
            worker_groups,
        })
    }

    /// Build geometry for an already-sized ring.
    #[must_use]
    pub fn geometry_from_slots(
        &self,
        slot_count: u32,
        worker_count: u32,
        max_workgroup_size_x: u32,
    ) -> ResidentLaunchGeometry {
        let workgroup_size_x = self.worker_workgroup_size(worker_count, max_workgroup_size_x);
        let slot_count = self.padded_slot_count(slot_count, workgroup_size_x);
        let dispatch_grid = self.dispatch_grid_for(worker_count, slot_count, workgroup_size_x);
        ResidentLaunchGeometry {
            workgroup_size_x,
            slot_count,
            dispatch_grid,
        }
    }
}
