//! Reading packed rows out of a resident e-graph device image: checking the
//! image against the kernel view that describes it, and comparing two rows
//! structurally.

use crate::egraph_device_image::CudaEGraphDeviceKernelView;
use vyre_foundation::optimizer::eqsat_gpu::GpuEGraphDeviceImage;

use super::CudaEGraphKernelPlanError;

pub(super) fn validate_image_view_matches(
    image: &GpuEGraphDeviceImage,
    view: CudaEGraphDeviceKernelView,
) -> Result<(), CudaEGraphKernelPlanError> {
    if image.layout().row_count() != view.row_count() {
        return Err(CudaEGraphKernelPlanError::ImageViewMismatch {
            field: "row count",
            image: image.layout().row_count(),
            view: view.row_count(),
        });
    }
    if image.layout().child_count() != view.child_count() {
        return Err(CudaEGraphKernelPlanError::ImageViewMismatch {
            field: "child count",
            image: image.layout().child_count(),
            view: view.child_count(),
        });
    }
    if image.layout().eclass_group_count() != view.eclass_group_count() {
        return Err(CudaEGraphKernelPlanError::ImageViewMismatch {
            field: "eclass group count",
            image: image.layout().eclass_group_count(),
            view: view.eclass_group_count(),
        });
    }
    Ok(())
}

pub(super) fn packed_rows_structurally_equal(
    image: &GpuEGraphDeviceImage,
    left_row: u32,
    right_row: u32,
) -> Result<bool, CudaEGraphKernelPlanError> {
    let left = left_row as usize;
    let right = right_row as usize;
    let row_count = image.layout().row_count();
    if left >= row_count {
        return Err(CudaEGraphKernelPlanError::ImageColumnOutOfBounds {
            column: "rows",
            row: left_row,
            start: left,
            end: left.saturating_add(1),
            len: row_count,
        });
    }
    if right >= row_count {
        return Err(CudaEGraphKernelPlanError::ImageColumnOutOfBounds {
            column: "rows",
            row: right_row,
            start: right,
            end: right.saturating_add(1),
            len: row_count,
        });
    }
    if image.row_signatures()[left] != image.row_signatures()[right] {
        return Ok(false);
    }
    if image.row_language_op_ids()[left] != image.row_language_op_ids()[right] {
        return Ok(false);
    }
    if image.row_children_lens()[left] != image.row_children_lens()[right] {
        return Ok(false);
    }

    let left_children = packed_row_children(image, left_row)?;
    let right_children = packed_row_children(image, right_row)?;
    Ok(left_children == right_children)
}

pub(super) fn packed_row_children(
    image: &GpuEGraphDeviceImage,
    row: u32,
) -> Result<&[u32], CudaEGraphKernelPlanError> {
    let row_index = row as usize;
    let start = image.row_children_offsets()[row_index] as usize;
    let len = image.row_children_lens()[row_index] as usize;
    let end = start
        .checked_add(len)
        .ok_or(CudaEGraphKernelPlanError::CountOverflow {
            field: "packed row child span end",
        })?;
    let children = image.children();
    if end > children.len() {
        return Err(CudaEGraphKernelPlanError::ImageColumnOutOfBounds {
            column: "children",
            row,
            start,
            end,
            len: children.len(),
        });
    }
    Ok(&children[start..end])
}
