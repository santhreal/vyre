//! Per-axis launch folding contracts for backend-neutral grid inference.
//!
//! WHY: closes the class "a one-dimensional launch larger than one grid axis
//! holds has no dispatch". A target that publishes the WebGPU minimum of 65535
//! workgroups per axis runs 16 776 960 lanes in blocks of 256 and not one more,
//! and every axis past x sits idle. The fold puts the excess there. It is only
//! sound while the emitted kernel reads a grid-linearized element index, so the
//! `grid_linearized` argument is the whole contract: false must keep the refusal
//! the old lowering earned, true must produce a grid that covers the launch with
//! no axis past its ceiling.
//!
//! The ceilings here are constructed rather than probed, because the number
//! under test is the portability floor every conformant implementation
//! guarantees, not the one the host in front of the suite happens to report.
//!
//! What this does not prove: that a folded grid names the same element on a
//! device. That needs an emitted kernel and a dispatch, and lives in the
//! backend crates.

use vyre_driver::{admit_dispatch_grid, infer_dispatch_grid_for_count, infer_launch_grid};

/// The per-axis workgroup ceiling every conformant WebGPU implementation
/// guarantees. A launch planned against this runs everywhere.
const WEBGPU_MIN_WORKGROUPS_PER_AXIS: u32 = 65_535;
/// The invocations per workgroup the same floor guarantees.
const WEBGPU_MIN_INVOCATIONS_PER_WORKGROUP: u32 = 256;
/// Lanes a single-axis launch covers at that floor: 16 776 960.
const SINGLE_AXIS_CEILING: u32 =
    WEBGPU_MIN_WORKGROUPS_PER_AXIS * WEBGPU_MIN_INVOCATIONS_PER_WORKGROUP;
/// The widest launch the fold runs in one dispatch at that floor. x fills to
/// 16 776 960 lanes, and y opens only as far as a u32 linear index reaches:
/// `2^32 / 16 776 960` is 256 rows, so the grid spans 4 294 901 760 lanes.
const WIDEST_FOLDED_LAUNCH: u32 = SINGLE_AXIS_CEILING * 256;

const FLOOR: [u32; 3] = [WEBGPU_MIN_WORKGROUPS_PER_AXIS; 3];
const BLOCK: [u32; 3] = [WEBGPU_MIN_INVOCATIONS_PER_WORKGROUP, 1, 1];

/// Lanes a grid of `grid` blocks of `workgroup` invocations covers.
fn lanes(grid: [u32; 3], workgroup: [u32; 3]) -> u64 {
    (0..3).fold(1_u64, |lanes, axis| {
        lanes * u64::from(grid[axis]) * u64::from(workgroup[axis])
    })
}

/// The linear index the emitted kernel computes for one invocation, mirroring
/// `x + y * x_extent + z * x_extent * y_extent` over the launched grid.
fn linear_index(invocation: [u32; 3], grid: [u32; 3], workgroup: [u32; 3]) -> u64 {
    let x_extent = u64::from(grid[0]) * u64::from(workgroup[0]);
    let y_extent = u64::from(grid[1]) * u64::from(workgroup[1]);
    u64::from(invocation[0])
        + u64::from(invocation[1]) * x_extent
        + u64::from(invocation[2]) * x_extent * y_extent
}

#[test]
fn a_launch_that_fits_one_axis_is_never_folded() {
    for count in [1, 256, 257, 1_000_000, SINGLE_AXIS_CEILING] {
        let unfolded = infer_dispatch_grid_for_count(count, BLOCK)
            .expect("Fix: a 1D block must have an inferable grid.");
        for linearized in [false, true] {
            let planned = infer_launch_grid(count, BLOCK, FLOOR, linearized, "test")
                .expect("Fix: a launch inside the per-axis ceiling must be planned.");
            assert_eq!(
                planned, unfolded,
                "Fix: a launch of {count} lane(s) fits one axis at the WebGPU floor, so it must \
                 keep the single-axis grid whatever the index space says. linearized={linearized}"
            );
        }
    }
}

#[test]
fn one_lane_past_the_single_axis_ceiling_folds_onto_y() {
    let count = SINGLE_AXIS_CEILING + 1;
    let grid = infer_launch_grid(count, BLOCK, FLOOR, true, "test")
        .expect("Fix: a launch one lane past one axis must fold onto the next.");
    assert_eq!(
        grid,
        [WEBGPU_MIN_WORKGROUPS_PER_AXIS, 2, 1],
        "Fix: the fold fills x to its ceiling before opening y, so the x extent stays the row \
         stride the emitted index multiplies y by."
    );
    assert!(
        lanes(grid, BLOCK) >= u64::from(count),
        "Fix: the folded grid must cover every lane the launch asked for."
    );
}

#[test]
fn a_folded_grid_stays_inside_every_ceiling_and_the_index_space() {
    let index_space = u64::from(u32::MAX) + 1;
    for count in [
        SINGLE_AXIS_CEILING + 1,
        SINGLE_AXIS_CEILING * 2,
        WIDEST_FOLDED_LAUNCH,
    ] {
        let grid = infer_launch_grid(count, BLOCK, FLOOR, true, "test")
            .unwrap_or_else(|error| panic!("Fix: {count} lane(s) must fold at the floor: {error}"));
        for axis in 0..3 {
            assert!(
                grid[axis] <= FLOOR[axis],
                "Fix: folded grid {grid:?} for {count} lane(s) puts {} workgroups on axis {axis}, \
                 past the {} ceiling it was planned against.",
                grid[axis],
                FLOOR[axis]
            );
        }
        assert!(
            lanes(grid, BLOCK) >= u64::from(count),
            "Fix: folded grid {grid:?} covers {} lane(s), short of the {count} asked for.",
            lanes(grid, BLOCK)
        );
        assert!(
            lanes(grid, BLOCK) <= index_space,
            "Fix: folded grid {grid:?} spans {} invocations, past the {index_space} a u32 linear \
             index reaches. The invocations past it wrap onto live elements.",
            lanes(grid, BLOCK)
        );
    }
}

/// WHY: the element index the emitted kernel computes is a u32, so the grid the
/// fold produces is bounded by the index space as well as by the ceilings. A
/// launch past it must be refused rather than folded into a grid whose last rows
/// wrap onto elements the first rows own.
#[test]
fn a_launch_past_the_u32_index_space_is_refused() {
    let error = infer_launch_grid(WIDEST_FOLDED_LAUNCH + 1, BLOCK, FLOOR, true, "test-backend")
        .expect_err("Fix: a launch past what a u32 index reaches must be refused, not wrapped.");
    let message = error.to_string();
    assert!(
        message.contains(&WIDEST_FOLDED_LAUNCH.to_string()) && message.contains("Fix:"),
        "Fix: the refusal must name the lanes one dispatch covers: {message}"
    );
}

/// WHY: the fold is only a launch if the linear index the kernel computes is a
/// bijection onto the element space over the whole grid. A stride taken from the
/// wrong axis, or a fold that opens y before x is full, aliases two invocations
/// onto one element and the wrong answer is a silent store, not an error.
#[test]
fn every_invocation_of_a_folded_grid_reaches_a_distinct_element() {
    // A small ceiling makes the whole invocation space enumerable while keeping
    // the arithmetic identical: the fold reads a ceiling, never a device.
    let ceiling = [4, 3, 2];
    let block = [8, 1, 1];
    let count = 4 * 8 * 3;
    let grid = infer_launch_grid(count, block, ceiling, true, "test")
        .expect("Fix: a launch past one axis must fold at any ceiling.");
    assert!(grid[1] > 1, "Fix: this case must fold onto y, got {grid:?}");
    let mut seen = vec![false; usize::try_from(lanes(grid, block)).expect("enumerable grid")];
    for z in 0..grid[2] * block[2] {
        for y in 0..grid[1] * block[1] {
            for x in 0..grid[0] * block[0] {
                let index = usize::try_from(linear_index([x, y, z], grid, block))
                    .expect("Fix: an enumerable grid must index a host vector.");
                assert!(
                    !seen[index],
                    "Fix: invocation [{x}, {y}, {z}] of grid {grid:?} reaches element {index}, \
                     which another invocation already reached. A folded launch that aliases two \
                     invocations onto one element stores the wrong answer without failing."
                );
                seen[index] = true;
            }
        }
    }
    assert!(
        seen.iter()
            .take(usize::try_from(count).expect("small count"))
            .all(|reached| *reached),
        "Fix: every element below the launch count must be reached by exactly one invocation."
    );
}

/// WHY: folding moves which invocation id names an element, so a kernel that
/// reads the x axis alone must keep the refusal. Folding it would recompute the
/// first row of elements once per row of the grid and report success.
#[test]
fn a_per_axis_kernel_past_the_ceiling_is_refused_by_name() {
    let count = SINGLE_AXIS_CEILING + 1;
    let error = infer_launch_grid(count, BLOCK, FLOOR, false, "test-backend")
        .expect_err("Fix: a per-axis index space past the ceiling must be refused, not folded.");
    let message = error.to_string();
    assert!(
        message.contains("65536")
            && message.contains("65535")
            && message.contains('x')
            && message.contains("test-backend")
            && message.contains("Fix:"),
        "Fix: the refusal must name the axis, the extent asked for, the ceiling and the backend: \
         {message}"
    );
}

/// WHY: the fold has its own ceiling, and past it the honest answer is still a
/// refusal. A launch that saturated silently would run a fraction of its lanes.
#[test]
fn a_launch_past_every_axis_is_refused_naming_the_capacity() {
    // One axis of 2 workgroups and nothing beyond it: 512 lanes in total.
    let ceiling = [2, 1, 1];
    let error = infer_launch_grid(4096, BLOCK, ceiling, true, "test-backend")
        .expect_err("Fix: a launch past every axis must be refused, not saturated.");
    let message = error.to_string();
    assert!(
        message.contains("512")
            && message.contains("4096")
            && message.contains("test-backend")
            && message.contains("Fix:"),
        "Fix: the refusal must name the lanes asked for and the lanes one dispatch covers: \
         {message}"
    );
}

/// WHY: a caller that pinned a grid asked for that exact shape. Reshaping it
/// would answer a different launch than the one submitted, so the pinned path
/// judges and never folds.
#[test]
fn a_pinned_grid_is_judged_and_never_folded() {
    let over = [WEBGPU_MIN_WORKGROUPS_PER_AXIS + 1, 1, 1];
    let error = admit_dispatch_grid(over, FLOOR, "test-backend")
        .expect_err("Fix: a pinned grid past the ceiling must be refused.");
    assert!(
        error.to_string().contains("65536"),
        "Fix: the refusal must name the extent the caller pinned: {error}"
    );
    let at_ceiling = [WEBGPU_MIN_WORKGROUPS_PER_AXIS, 1, 1];
    assert_eq!(
        admit_dispatch_grid(at_ceiling, FLOOR, "test-backend")
            .expect("Fix: a pinned grid at exactly the ceiling must be admitted."),
        at_ceiling,
        "Fix: admission returns the grid it judged."
    );
}

/// WHY: a zero ceiling means no device was probed. Inventing one there refuses
/// launches the target accepts, and folding against one produces a grid nobody
/// asked for.
#[test]
fn an_unprobed_axis_neither_refuses_nor_folds() {
    let count = SINGLE_AXIS_CEILING + 1;
    let unfolded = infer_dispatch_grid_for_count(count, BLOCK)
        .expect("Fix: a 1D block must have an inferable grid.");
    for linearized in [false, true] {
        assert_eq!(
            infer_launch_grid(count, BLOCK, [0; 3], linearized, "test")
                .expect("Fix: an unprobed ceiling must not refuse a launch."),
            unfolded,
            "Fix: with no ceiling to plan against, the launch keeps the grid inference derives. \
             linearized={linearized}"
        );
    }
}
