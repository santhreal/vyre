//! Neutral launch legality and resident-ring sizing facts.
//!
//! Selecting a schedule belongs to `vyre-megakernel`. What is left here answers
//! legality and ring arithmetic: whether a fused launch stays inside the axis
//! budget, how many lanes a worker workgroup takes, and how a slot count pads
//! into whole workgroups. None of it ranks a candidate against a device.

/// Central contract for launch legality and resident-ring thresholds.
///
/// The values are private on purpose: callers ask policy questions instead of
/// copying numeric thresholds into each crate.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct SchedulingPolicy {
    fused_over_dispatch_multiplier: u64,
    default_worker_count: u32,
    max_dispatch_workgroups: u32,
}

impl Default for SchedulingPolicy {
    fn default() -> Self {
        Self::standard()
    }
}

impl SchedulingPolicy {
    /// Return the standard persistent-megakernel policy used by the built-in planners.
    #[must_use]
    pub const fn standard() -> Self {
        Self {
            fused_over_dispatch_multiplier: 4,
            default_worker_count: 64,
            max_dispatch_workgroups: 1024,
        }
    }

    /// Return true when an axis-wise fused launch stays within policy.
    #[must_use]
    pub const fn allow_fused_threads(&self, fused_threads: u64, max_arm_threads: u64) -> bool {
        fused_threads <= max_arm_threads.saturating_mul(self.fused_over_dispatch_multiplier)
    }

    /// Multiplier used to reject pathological axis-wise fused launch shapes.
    #[must_use]
    pub const fn fused_over_dispatch_multiplier(&self) -> u64 {
        self.fused_over_dispatch_multiplier
    }

    /// Default persistent worker workgroup count.
    #[must_use]
    pub const fn default_worker_count(&self) -> u32 {
        self.default_worker_count
    }

    /// Clamp a requested worker count into the legal workgroup x dimension.
    #[must_use]
    pub const fn worker_workgroup_size(&self, worker_count: u32, max_workgroup_size_x: u32) -> u32 {
        let max_workgroup_size_x = if max_workgroup_size_x > 1 {
            max_workgroup_size_x
        } else {
            1
        };
        if worker_count == 0 {
            1
        } else if worker_count > max_workgroup_size_x {
            max_workgroup_size_x
        } else {
            worker_count
        }
    }

    /// Round a logical slot count up to a whole worker workgroup.
    #[must_use]
    pub const fn padded_slot_count(&self, slot_count: u32, workgroup_size_x: u32) -> u32 {
        let workgroup_size_x = if workgroup_size_x > 1 {
            workgroup_size_x
        } else {
            1
        };
        let groups = slot_count
            .saturating_add(workgroup_size_x - 1)
            .saturating_div(workgroup_size_x);
        let groups = if groups > 1 { groups } else { 1 };
        groups.saturating_mul(workgroup_size_x)
    }

    /// Compute the backend dispatch grid for a logical queue length.
    #[must_use]
    pub const fn dispatch_grid_for(
        &self,
        worker_count: u32,
        queue_len: u32,
        max_workgroup_size_x: u32,
    ) -> [u32; 3] {
        let workgroup_width = if max_workgroup_size_x > 1 {
            max_workgroup_size_x
        } else {
            1
        };
        let requested_workers = if worker_count > 1 { worker_count } else { 1 };
        let workgroups = queue_len
            .saturating_add(workgroup_width - 1)
            .saturating_div(workgroup_width);
        let workgroups = if workgroups > 1 { workgroups } else { 1 };
        let final_workgroups = min3(workgroups, requested_workers, self.max_dispatch_workgroups);
        [final_workgroups, 1, 1]
    }
}

const fn min3(a: u32, b: u32, c: u32) -> u32 {
    let ab = if a < b { a } else { b };
    if ab < c {
        ab
    } else {
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> SchedulingPolicy {
        SchedulingPolicy::standard()
    }

    #[test]
    fn worker_workgroup_size_clamps_to_max() {
        assert_eq!(policy().worker_workgroup_size(512, 256), 256);
    }

    #[test]
    fn worker_workgroup_size_zero_becomes_one() {
        assert_eq!(policy().worker_workgroup_size(0, 256), 1);
    }

    #[test]
    fn worker_workgroup_size_within_range_preserved() {
        assert_eq!(policy().worker_workgroup_size(128, 256), 128);
    }

    // --- Slot padding ---

    #[test]
    fn padded_slot_count_rounds_up() {
        assert_eq!(policy().padded_slot_count(65, 64), 128);
    }

    #[test]
    fn padded_slot_count_exact_multiple_unchanged() {
        assert_eq!(policy().padded_slot_count(128, 64), 128);
    }

    #[test]
    fn padded_slot_count_minimum_is_one_workgroup() {
        assert_eq!(policy().padded_slot_count(1, 64), 64);
    }

    // --- Dispatch grid ---

    #[test]
    fn dispatch_grid_single_workgroup() {
        let grid = policy().dispatch_grid_for(64, 32, 64);
        assert_eq!(grid, [1, 1, 1]);
    }

    #[test]
    fn dispatch_grid_capped_at_max() {
        let grid = policy().dispatch_grid_for(9999, 999999, 64);
        // Should be capped at max_dispatch_workgroups (1024).
        assert!(grid[0] <= 1024);
    }

    // --- Fusion limit ---

    #[test]
    fn allow_fused_threads_within_multiplier() {
        assert!(policy().allow_fused_threads(100, 100));
        assert!(policy().allow_fused_threads(400, 100)); // 4x
        assert!(!policy().allow_fused_threads(401, 100)); // >4x
    }
}
