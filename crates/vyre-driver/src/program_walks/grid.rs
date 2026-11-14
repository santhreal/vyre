//! Backend-neutral dispatch-grid inference.

use vyre_foundation::ir::Program;

use crate::backend::{BackendError, DispatchConfig};
use crate::binding::BindingPlan;
use crate::program_walks::dispatch_element_count_for_program;

/// Infer a concrete workgroup grid from a program ABI and dispatch inputs.
///
/// Explicit [`DispatchConfig::grid_override`] always wins. Otherwise this uses
/// the largest non-shared binding element count as the logical lane count and
/// derives a deterministic 1D/2D/3D grid from the effective workgroup shape.
///
/// # Errors
///
/// Returns when the program/input ABI cannot be planned or when inferred grid
/// dimensions overflow `u32`.
pub fn infer_dispatch_grid(
    program: &Program,
    inputs: &[Vec<u8>],
    config: &DispatchConfig,
) -> Result<[u32; 3], BackendError> {
    if let Some(grid) = config.grid_override {
        return Ok(grid);
    }
    let plan = BindingPlan::from_program(program, inputs)?;
    let element_count = dispatch_element_count_for_program(program, &plan.bindings);
    infer_dispatch_grid_for_count(
        element_count,
        config
            .launch_workgroup()
            .unwrap_or(program.workgroup_size()),
    )
}

/// Infer a launch grid for a known logical element count and workgroup shape.
///
/// The grid covers the axis decomposition
/// [`vyre_foundation::axis_coverage`] states for the shape, so a program
/// dispatched here launches the shape a compiled artifact of the same program
/// records.
///
/// # Errors
///
/// Returns if any workgroup axis is zero or an inferred grid axis cannot fit
/// in `u32`.
pub fn infer_dispatch_grid_for_count(
    element_count: u32,
    workgroup: [u32; 3],
) -> Result<[u32; 3], BackendError> {
    if workgroup.contains(&0) {
        return Err(BackendError::new(
            "workgroup dimensions must be non-zero. Fix: set Program::workgroup_size and DispatchConfig::workgroup_override to positive values.",
        ));
    }
    let coverage = vyre_foundation::axis_coverage(u64::from(element_count), workgroup);
    let mut grid = [1_u32; 3];
    for (axis, slot) in grid.iter_mut().enumerate() {
        *slot = ceil_div_u64(coverage[axis], u64::from(workgroup[axis]))?;
    }
    Ok(grid)
}

/// Refuse a launch grid that exceeds the target's per-axis workgroup ceiling.
///
/// Every graphics-derived target publishes a maximum workgroup count PER AXIS
/// that is far below `u32::MAX`, and a dispatch past it is not a slow launch: the
/// API rejects the command, and a rejection inside a recorded command buffer
/// surfaces as a validation abort rather than as a value this crate can return.
/// So the grid is judged against the ceiling the device reported, before the
/// command is recorded, and the refusal names the axis, the extent asked for and
/// the ceiling, because those three are what a caller needs to reshape the launch.
///
/// A zero ceiling on an axis means the caller has no ceiling to enforce there (no
/// device was probed), and that axis passes: inventing one here would refuse
/// launches the target accepts.
///
/// This judges a grid a caller pinned. A grid the launch planner infers reaches
/// the ceiling through [`infer_launch_grid`], which folds the excess across the
/// remaining axes when the kernel reads a grid-linearized index.
///
/// # Errors
///
/// Returns when any axis of `grid` exceeds its ceiling.
pub fn admit_dispatch_grid(
    grid: [u32; 3],
    max_per_axis: [u32; 3],
    backend_id: &str,
) -> Result<[u32; 3], BackendError> {
    for (axis, extent) in grid.iter().copied().enumerate() {
        let ceiling = max_per_axis[axis];
        if ceiling != 0 && extent > ceiling {
            let axis_name = ["x", "y", "z"][axis];
            return Err(BackendError::InvalidProgram {
                fix: format!(
                    "Fix: dispatch grid {grid:?} asks for {extent} workgroups on axis {axis_name}, above the {ceiling} that `{backend_id}` reported as its per-axis maximum. Reshape the launch so no axis exceeds {ceiling}: give the program a larger workgroup, split the work into several dispatches, or set DispatchConfig::grid_override to a shape whose axes all fit. A program cannot declare a grid the target rejects."
                ),
            });
        }
    }
    Ok(grid)
}

/// Infer the launch grid for `element_count` lanes, folding across grid axes
/// when one axis cannot hold the launch.
///
/// A one-dimensional program whose workgroup count exceeds the per-axis ceiling
/// has a launch the target accepts: the excess belongs on y, then on z. Folding
/// it there moves which invocation id a lane observes, so it is sound only when
/// the emitted kernel reads its element index linearized over the whole grid,
/// which `grid_linearized` states. A kernel that reads the x axis alone keeps its
/// single-axis grid and is refused past the ceiling, because folding it would
/// recompute the first row of elements once per row of the grid.
///
/// A launch that fits one axis is never folded: it gets exactly the grid
/// [`infer_dispatch_grid_for_count`] derives, whatever `grid_linearized` says.
///
/// # Errors
///
/// Returns when the workgroup shape is degenerate, when the launch does not fit
/// even folded across every axis, or when a kernel that indexes per axis asks
/// for more workgroups on one axis than the target admits.
pub fn infer_launch_grid(
    element_count: u32,
    workgroup: [u32; 3],
    max_per_axis: [u32; 3],
    grid_linearized: bool,
    backend_id: &str,
) -> Result<[u32; 3], BackendError> {
    let grid = infer_dispatch_grid_for_count(element_count, workgroup)?;
    let fits = grid
        .iter()
        .zip(max_per_axis)
        .all(|(extent, ceiling)| ceiling == 0 || *extent <= ceiling);
    if fits || !grid_linearized {
        return admit_dispatch_grid(grid, max_per_axis, backend_id);
    }
    fold_launch_grid(element_count, workgroup, max_per_axis, backend_id)
}

/// Spread `element_count` lanes across every grid axis, filling each axis to its
/// ceiling before opening the next.
///
/// Filling in order is what keeps the fold readable from inside the kernel: the
/// x extent is the row stride the linearized index multiplies y by, and the
/// kernel reads it from the workgroup count the launch states rather than from a
/// constant, so one compiled module serves every folded shape.
fn fold_launch_grid(
    element_count: u32,
    workgroup: [u32; 3],
    max_per_axis: [u32; 3],
    backend_id: &str,
) -> Result<[u32; 3], BackendError> {
    let count = u64::from(element_count.max(1));
    // A linear index is a u32, so the whole folded grid may span at most 2^32
    // invocations however many the ceilings would otherwise admit. An
    // invocation past that computes a wrapped index, and a wrapped index passes
    // a bounds guard that cannot see the wrap and stores over a live element.
    let index_space = u64::from(u32::MAX) + 1;
    let mut grid = [1_u32; 3];
    // Lanes every axis below this one already covers, so the extent this axis
    // needs is the whole count divided by what one step of it adds.
    let mut covered = 1_u64;
    for axis in 0..3 {
        let ceiling = match max_per_axis[axis] {
            0 => u64::from(u32::MAX),
            ceiling => u64::from(ceiling),
        };
        let lanes_per_group = covered.saturating_mul(u64::from(workgroup[axis])).max(1);
        let addressable = (index_space / lanes_per_group).max(1);
        let taken = count
            .div_ceil(lanes_per_group)
            .max(1)
            .min(ceiling)
            .min(addressable);
        grid[axis] = u32::try_from(taken).map_err(|_| {
            BackendError::new(format!(
                "folded dispatch grid axis {axis} extent {taken} overflowed u32. Fix: split the Program into smaller dispatches."
            ))
        })?;
        covered = lanes_per_group.saturating_mul(taken);
        if covered >= count {
            return Ok(grid);
        }
    }
    // Every axis is filled to the smaller of its ceiling and what a u32 index
    // reaches, so `covered` is the largest launch this target runs in one
    // dispatch. Saturating here would run a fraction of the lanes and report
    // success.
    Err(BackendError::InvalidProgram {
        fix: format!(
            "Fix: a launch of {element_count} element(s) needs more invocations than `{backend_id}` admits in one dispatch. A workgroup of {workgroup:?} across at most {max_per_axis:?} workgroups per axis covers {covered} lane(s). Split the work into several dispatches, or give the program a larger workgroup."
        ),
    })
}

fn ceil_div_u64(value: u64, divisor: u64) -> Result<u32, BackendError> {
    let divided = value.div_ceil(divisor).max(1);
    u32::try_from(divided).map_err(|_| {
        BackendError::new(
            "inferred dispatch grid dimension overflowed u32. Fix: split the Program into smaller dispatches.",
        )
    })
}

// Inline: covers `ceil_cuberoot_u64`, `ceil_sqrt_u64`, which no integration test can name.
#[cfg(test)]
mod n6_tests {
    use super::*;

    #[test]
    fn a_multi_axis_grid_is_exact_at_the_u32_element_boundary() {
        assert_eq!(
            infer_dispatch_grid_for_count(u32::MAX, [1, 2, 1])
                .expect("a two-axis grid at the element boundary must plan"),
            [65_536, 32_768, 1]
        );
        assert_eq!(
            infer_dispatch_grid_for_count(u32::MAX, [2, 2, 2])
                .expect("a three-axis grid at the element boundary must plan"),
            [813, 813, 813]
        );
    }
}
