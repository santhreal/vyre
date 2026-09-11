//! The one ceiling division from a lane count to a dispatch grid.
//!
//! A dispatch grid is a launch-geometry concept, not a graph one. Keeping the
//! owner here rather than inside a domain module is what lets every domain reach
//! it: `decode` does not enable `graph`, so a `graph`-owned helper is invisible
//! from a decode primitive, and the primitive that could not reach it hand-rolled
//! the arithmetic instead. An owner nobody can reach is not an owner.

/// Blocks needed to give every one of `lanes` items its own invocation, as a
/// one-dimensional dispatch grid.
///
/// Every lane-per-item primitive launches the same shape: `ceil(lanes /
/// lanes_per_group)` groups on x, one on y and z, floored at one group. The floor
/// matters because a zero-length input must still produce a launchable grid: the
/// launcher rejects `grid[axis] == 0` outright, and the kernel bodies already
/// guard every lane against the element count, so one group of bounds-guarded
/// lanes is a no-op while zero groups is a launch failure.
///
/// This replaced six hand-rolled ceiling helpers whose zero cases disagreed: four
/// spelled it `((value - 1) / divisor) + 1`, which underflows at zero and was only
/// safe because each call site pre-floored its input, one split the count into
/// `full_blocks` plus a `tail_block` and then floored the sum, and one returned a
/// grid of zero groups.
#[must_use]
pub const fn lane_grid(lanes: u32, lanes_per_group: u32) -> [u32; 3] {
    let groups = lanes.div_ceil(lanes_per_group);
    [if groups == 0 { 1 } else { groups }, 1, 1]
}

#[cfg(test)]
mod tests {
    use super::lane_grid;

    /// The zero case every hand-rolled copy got differently.
    #[test]
    fn zero_lanes_still_produce_a_launchable_grid() {
        assert_eq!(
            lane_grid(0, 256),
            [1, 1, 1],
            "Fix: a zero-length input must launch one bounds-guarded group; the launcher rejects a zero extent outright."
        );
    }

    #[test]
    fn a_partial_group_rounds_up_rather_than_dropping_its_tail() {
        assert_eq!(lane_grid(1, 256), [1, 1, 1]);
        assert_eq!(lane_grid(256, 256), [1, 1, 1]);
        assert_eq!(
            lane_grid(257, 256),
            [2, 1, 1],
            "Fix: the lane past a whole group needs its own group, or it is never dispatched."
        );
        assert_eq!(lane_grid(512, 256), [2, 1, 1]);
        assert_eq!(lane_grid(513, 256), [3, 1, 1]);
    }

    /// The grid never overflows, so a saturated lane count is still launchable.
    #[test]
    fn a_saturated_lane_count_rounds_up_without_overflow() {
        assert_eq!(lane_grid(u32::MAX, 256), [u32::MAX.div_ceil(256), 1, 1]);
        assert_eq!(lane_grid(u32::MAX, 1), [u32::MAX, 1, 1]);
    }
}
