//! Grid index space lowering contracts.
//!
//! WHY: closes the class "a launch folded across grid axes reads the wrong
//! element". A one-dimensional launch larger than one grid axis holds is only
//! dispatchable if the emitted kernel derives its element index from the whole
//! grid, so `GridIndexSpace` decides an arithmetic shape and not a label. Two
//! failures are equally silent: a per-axis lowering under a folded launch
//! recomputes the first row of elements once per row of the grid, and a
//! linearized lowering under a launch that was not folded is only correct
//! because y and z are zero. Both return `Ok` with wrong or accidentally right
//! bytes, so the arithmetic is asserted here rather than the enum.
//!
//! What this does not prove: that the grid the launch planner folds to is the
//! grid these strides were computed for. That pairing is a driver contract.

use super::*;

use naga::{BinaryOperator, Expression};
use vyre_lower::descriptor_builder::op;
use vyre_lower::GridIndexSpace;

/// A store of a literal at `global_invocation_id[axis]`: the smallest descriptor
/// whose emitted index arithmetic is the whole subject.
fn store_at_global_axis(id: &str, axis: u32, grid_index: GridIndexSpace) -> KernelDescriptor {
    descriptor(id)
        .slots([global_rw(0, DataType::U32, "out").with_count(8)])
        .dispatch(256, 1, 1)
        .grid_index(grid_index)
        .body(
            body()
                .ops([
                    op(KernelOpKind::GlobalInvocationId, [axis], 0),
                    lit(1, 1),
                    effect(KernelOpKind::StoreGlobal, [0, 0, 1]),
                ])
                .literals([LiteralValue::U32(0), LiteralValue::U32(7)]),
        )
        .build()
}

/// Number of arguments the entry point takes with `builtin`.
fn builtin_arg_count(module: &naga::Module, builtin: BuiltIn) -> usize {
    module.entry_points[0]
        .function
        .arguments
        .iter()
        .filter(|argument| argument.binding == Some(Binding::BuiltIn(builtin)))
        .count()
}

/// Number of binary expressions in the entry point applying `operator`.
fn binary_count(module: &naga::Module, operator: BinaryOperator) -> usize {
    module.entry_points[0]
        .function
        .expressions
        .iter()
        .filter(|(_, expression)| {
            matches!(expression, Expression::Binary { op, .. } if *op == operator)
        })
        .count()
}

/// WHY: every launch that fits one grid axis keeps this lowering, so it is the
/// shape the whole backend is measured against. A per-axis index that acquired
/// grid arithmetic would recompile every cached module and pay a multiply per
/// lane for a fold nobody asked for.
#[test]
fn a_per_axis_index_reads_one_component_and_no_grid_extent() {
    let module = emit(&store_at_global_axis(
        "per-axis",
        0,
        GridIndexSpace::PerAxis,
    ))
    .expect("Fix: a per-axis descriptor must emit.");
    assert_valid_wgsl(&module, "per-axis global index");
    assert_eq!(
        builtin_arg_count(&module, BuiltIn::NumWorkGroups),
        0,
        "Fix: a per-axis index never reads the grid's extent, so the entry point must not take \
         the workgroup-count builtin. Taking it changes the emitted signature of every kernel."
    );
}

/// WHY: the folded launch is only correct if the emitted index is
/// `x + y * x_extent + z * x_extent * y_extent` with each extent read from the
/// launched grid. A stride taken from a constant serves one folded shape and
/// silently mis-addresses every other, and a missing z term drops the third
/// axis without failing.
///
/// The counts are differential against the per-axis lowering of the same
/// descriptor, so the case measures the arithmetic the index space adds rather
/// than whatever a store already owes.
#[test]
fn a_grid_linearized_index_reads_both_strides_from_the_launched_grid() {
    let per_axis = emit(&store_at_global_axis(
        "per-axis",
        0,
        GridIndexSpace::PerAxis,
    ))
    .expect("Fix: a per-axis descriptor must emit.");
    let module = emit(&store_at_global_axis(
        "linearized",
        0,
        GridIndexSpace::GridLinearized,
    ))
    .expect("Fix: a grid-linearized descriptor must emit.");
    assert_valid_wgsl(&module, "grid-linearized global index");
    assert_eq!(
        builtin_arg_count(&module, BuiltIn::NumWorkGroups),
        1,
        "Fix: the strides are the launched workgroup count times the declared workgroup size, so \
         the entry point must take the workgroup-count builtin exactly once. Without it one \
         compiled module cannot serve every grid the planner folds to."
    );
    // x_extent = num_workgroups.x * 256, row = y * x_extent,
    // plane_stride = x_extent * y_extent, plane = z * plane_stride.
    // y_extent needs no multiply because the declared workgroup size on y is 1.
    assert_eq!(
        binary_count(&module, BinaryOperator::Multiply),
        binary_count(&per_axis, BinaryOperator::Multiply) + 4,
        "Fix: the linearized index is four multiplies: the x invocation extent, the y row \
         offset, the z plane stride, and the z plane offset. A different count means a stride \
         was dropped or duplicated."
    );
    assert_eq!(
        binary_count(&module, BinaryOperator::Add),
        binary_count(&per_axis, BinaryOperator::Add) + 2,
        "Fix: the linearized index sums three terms, which is two adds. Fewer means an axis was \
         dropped from the index."
    );
}

/// WHY: a workgroup size above one on y or z multiplies that axis's invocation
/// extent, and reading the workgroup count alone there would place every row
/// after the first at a fraction of its offset.
#[test]
fn a_grid_linearized_index_scales_every_axis_by_its_declared_workgroup_size() {
    let flat = emit(&store_at_global_axis(
        "flat",
        0,
        GridIndexSpace::GridLinearized,
    ))
    .expect("Fix: a 1D workgroup must emit under a linearized index.");
    let mut desc = store_at_global_axis("tiled", 0, GridIndexSpace::GridLinearized);
    desc.dispatch.workgroup_size = [64, 4, 1];
    let tiled = emit(&desc).expect("Fix: a 2D workgroup must emit under a linearized index.");
    assert_valid_wgsl(&tiled, "grid-linearized index over a 2D workgroup");
    assert_eq!(
        binary_count(&tiled, BinaryOperator::Multiply),
        binary_count(&flat, BinaryOperator::Multiply) + 1,
        "Fix: a workgroup wider than one invocation on y owes that axis its own multiply. \
         Without it the y invocation extent is the workgroup count alone and every plane after \
         the first is placed a factor of the workgroup height too low."
    );
}

/// WHY: folding moves which invocation id names an element, so a kernel that
/// reads y, z, or the workgroup id observes the fold itself. Emitting it under a
/// linearized index would return an id that no longer describes the element the
/// lane owns, and the store would land somewhere plausible. The refusal is what
/// keeps the launch planner from folding such a kernel.
#[test]
fn a_kernel_that_observes_the_grid_shape_is_refused_a_linearized_index() {
    for (what, desc) in [
        (
            "the global invocation id on y",
            store_at_global_axis("gid-y", 1, GridIndexSpace::GridLinearized),
        ),
        (
            "the global invocation id on z",
            store_at_global_axis("gid-z", 2, GridIndexSpace::GridLinearized),
        ),
        ("the workgroup id", {
            let mut desc = store_at_global_axis("wgid", 0, GridIndexSpace::GridLinearized);
            desc.body.ops.push(op(KernelOpKind::WorkgroupId, [1], 3));
            desc
        }),
    ] {
        assert!(
            emit(&desc).is_err(),
            "Fix: a descriptor reading {what} under a grid-linearized index must be refused, \
             because the id it would read no longer names the element the lane owns."
        );
    }
}

/// WHY: the refusal above must be the emitter's, spelled so a caller knows which
/// of the two launch decisions to change.
#[test]
fn the_refusal_names_both_ways_out() {
    let error = emit(&store_at_global_axis(
        "gid-y",
        1,
        GridIndexSpace::GridLinearized,
    ))
    .expect_err(
        "Fix: reading the global invocation id on y under a folded launch is not lowerable.",
    );
    let message = error.to_string();
    assert!(
        message.contains("Fix:") && message.contains("one grid axis") && message.contains("x axis"),
        "Fix: the refusal must state both ways out, planning the launch on one axis or addressing \
         from the x axis alone: {message}"
    );
}

/// WHY: the same kernels the linearized space refuses must still emit under the
/// per-axis space, or the refusal is measuring the ops rather than the index
/// space and takes every multi-axis kernel off the backend.
#[test]
fn a_kernel_that_observes_the_grid_shape_emits_under_a_per_axis_index() {
    for axis in 0..3 {
        let module = emit(&store_at_global_axis(
            "per-axis",
            axis,
            GridIndexSpace::PerAxis,
        ))
        .unwrap_or_else(|error| {
            panic!("Fix: axis {axis} must emit under a per-axis index: {error}")
        });
        assert_valid_wgsl(&module, "per-axis multi-axis read");
    }
    let mut desc = store_at_global_axis("wgid", 0, GridIndexSpace::PerAxis);
    desc.body.ops.push(op(KernelOpKind::WorkgroupId, [1], 3));
    let module = emit(&desc).expect("Fix: a workgroup-id read must emit under a per-axis index.");
    assert_valid_wgsl(&module, "per-axis workgroup id read");
}
