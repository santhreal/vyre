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

/// Infer a grid size for a program based on its largest statically-known
/// non-shared binding and its workgroup size.
///
/// Bench cases and backends can use this when no explicit grid_override is provided.
///
/// # Errors
///
/// Returns when the program ABI cannot be planned or if inferred dimensions
/// overflow `u32`.
pub fn auto_grid(
    program: &Program,
    backend: &dyn crate::backend::VyreBackend,
) -> Result<[u32; 3], BackendError> {
    crate::validation::validate_program_for_backend(backend, program, &DispatchConfig::default())?;
    let plan = BindingPlan::build(program)?;
    let element_count = dispatch_element_count_for_program(program, &plan.bindings);

    infer_dispatch_grid_for_count(element_count, program.workgroup_size())
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

// ---------------------------------------------------------------------------
// N6 power-of-2 dispatch grid coercion + tail-mask
// ---------------------------------------------------------------------------

/// Result of coercing a logical element count up to the next power of two.
///
/// Backends that opt into the N6 substrate dispatch over `rounded_count`
/// lanes (so every workgroup is uniform-shape, no boundary divergence on
/// the last workgroup) and have the kernel guard each store with the
/// tail-mask predicate `lane_id < original_count`. Threads beyond the
/// original count no-op their stores.
///
/// The win is on tail handling for attention/softmax/reduce shapes where
/// the workload is not a multiple of the workgroup size  -  without
/// coercion the last workgroup runs with masked-out lanes that still
/// incur scheduling cost; with coercion every workgroup is identical
/// and the masked-out lanes are skipped via the predicate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TailMaskPolicy {
    /// Logical element count requested by the caller.
    pub original_count: u32,
    /// Element count after rounding up to the next power of two. Equal
    /// to `original_count` when it is already a power of two.
    pub rounded_count: u32,
    /// Convenience: `rounded_count - original_count`. Lanes in this
    /// suffix range must be predicated off by the kernel.
    pub tail_lanes: u32,
}

impl TailMaskPolicy {
    /// True when no rounding was needed; the dispatch can run as-is
    /// without a tail-mask predicate.
    #[must_use]
    pub fn is_aligned(&self) -> bool {
        self.tail_lanes == 0
    }
}

/// N6: round `element_count` up to the next power of two. Returns a
/// [`TailMaskPolicy`] that the lower/emit layer consumes to insert a
/// `lane_id < original_count` predicate around each store. Pure
/// arithmetic; no I/O.
///
/// `element_count == 0` is treated as 0 (rounded_count = 0, no tail).
/// `element_count == 1` rounds to 1 (already pow2).
/// `element_count` beyond `1 << 31` cannot be rounded inside `u32`; callers
/// that need to distinguish that condition must use
/// [`try_coerce_to_pow2_with_tail_mask`]. This legacy wrapper preserves the
/// original shape on overflow instead of panicking.
#[must_use]
pub fn coerce_to_pow2_with_tail_mask(element_count: u32) -> TailMaskPolicy {
    match try_coerce_to_pow2_with_tail_mask(element_count) {
        Ok(policy) => policy,
        Err(_error) => TailMaskPolicy {
            original_count: element_count,
            rounded_count: element_count,
            tail_lanes: 0,
        },
    }
}

/// Fallible N6 power-of-two dispatch-grid coercion.
///
/// # Errors
/// Returns when `element_count` cannot be rounded up inside `u32`.
pub fn try_coerce_to_pow2_with_tail_mask(
    element_count: u32,
) -> Result<TailMaskPolicy, BackendError> {
    if element_count == 0 {
        return Ok(TailMaskPolicy {
            original_count: 0,
            rounded_count: 0,
            tail_lanes: 0,
        });
    }
    let rounded = next_pow2_u32_checked(element_count)?;
    Ok(TailMaskPolicy {
        original_count: element_count,
        rounded_count: rounded,
        tail_lanes: rounded - element_count,
    })
}

fn next_pow2_u32_checked(value: u32) -> Result<u32, BackendError> {
    if value.is_power_of_two() {
        return Ok(value);
    }
    if value > (1u32 << 31) {
        return Err(BackendError::new(format!(
            "cannot round element_count={value} up to a power-of-two u32 grid without overflow. Fix: split the workload before grid-shape planning; do not silently saturate or fall back to an under-dispatching shape."
        )));
    }
    Ok(value.next_power_of_two())
}

// Inline: covers `ceil_cuberoot_u64`, `ceil_sqrt_u64`, which no integration test can name.
#[cfg(test)]
mod n6_tests {
    use super::*;

    #[test]
    fn already_pow2_is_identity_with_no_tail() {
        let p = coerce_to_pow2_with_tail_mask(64);
        assert_eq!(p.original_count, 64);
        assert_eq!(p.rounded_count, 64);
        assert_eq!(p.tail_lanes, 0);
        assert!(p.is_aligned());
    }

    #[test]
    fn non_pow2_rounds_up_and_reports_tail() {
        let p = coerce_to_pow2_with_tail_mask(100);
        assert_eq!(p.original_count, 100);
        assert_eq!(p.rounded_count, 128);
        assert_eq!(p.tail_lanes, 28);
        assert!(!p.is_aligned());
    }

    #[test]
    fn one_is_pow2_no_tail() {
        let p = coerce_to_pow2_with_tail_mask(1);
        assert_eq!(p.rounded_count, 1);
        assert_eq!(p.tail_lanes, 0);
    }

    #[test]
    fn zero_passes_through_with_no_tail() {
        let p = coerce_to_pow2_with_tail_mask(0);
        assert_eq!(p.rounded_count, 0);
        assert_eq!(p.tail_lanes, 0);
        assert!(p.is_aligned());
    }

    #[test]
    fn large_value_below_2_31_rounds_normally() {
        let p = coerce_to_pow2_with_tail_mask(1_000_000_000);
        // 2^30 = 1_073_741_824
        assert_eq!(p.rounded_count, 1u32 << 30);
        assert_eq!(p.tail_lanes, (1u32 << 30) - 1_000_000_000);
    }

    #[test]
    fn value_above_2_31_errors_instead_of_saturating() {
        let error = try_coerce_to_pow2_with_tail_mask(u32::MAX)
            .expect_err("oversized power-of-two coercion must fail loudly");
        let message = error.to_string();
        assert!(
            message.contains("Fix:"),
            "oversized grid-shape error must be actionable"
        );
    }

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
