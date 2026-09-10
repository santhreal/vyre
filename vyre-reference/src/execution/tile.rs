//! Canonical shared execution logic for Tile operations.
//!
//! Provides the single-homed reference implementation for tile load, store,
//! element conversion, matrix multiplication, and reduction across both
//! sequential and hashmap execution engines.

use vyre_foundation::ir::{row_major_coords, Layout, SubgroupReduceOp, Tile};

use crate::error::ReferenceError;

use crate::oob::{self, Buffer};
use crate::value::Value;

/// Convert a reference [`Value`] to a flattened element vector.
pub(crate) fn to_elements(val: &Value) -> Vec<Value> {
    match val {
        Value::Array(e) => e.clone(),
        s => vec![s.clone()],
    }
}

/// Load tile elements from a backing buffer with layout translation.
///
/// # Errors
/// Propagates an out-of-bounds refusal from any element the tile reads.
pub(crate) fn load_elements(
    target: &Buffer,
    origin_coords: &[u32],
    tile_type: &Tile,
    layout: &Layout,
) -> Result<Vec<Value>, ReferenceError> {
    let total_elements = tile_type.element_count();
    let mut elements = vec![Value::Float(0.0); total_elements];

    let mut strides = vec![1u32; tile_type.extents.len()];
    for i in (0..tile_type.extents.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * tile_type.extents[i + 1];
    }

    if tile_type.extents.is_empty() {
        let global_idx = origin_coords.first().copied().unwrap_or(0);
        let val = oob::load(target, global_idx)?;
        elements = vec![val];
    } else if tile_type.extents.len() == 1 {
        let n = tile_type.extents[0];
        let base = origin_coords.first().copied().unwrap_or(0);
        for i in 0..n {
            let global_idx = base + i;
            let val = oob::load(target, global_idx)?;
            let local_idx = layout.linear_index(&[i], &tile_type.extents);
            if local_idx < elements.len() {
                elements[local_idx] = val;
            }
        }
    } else if tile_type.extents.len() == 2 {
        let rows = tile_type.extents[0];
        let cols = tile_type.extents[1];
        let r_base = origin_coords.first().copied().unwrap_or(0);
        let c_base = origin_coords.get(1).copied().unwrap_or(0);
        for r in 0..rows {
            for c in 0..cols {
                let global_idx = (r_base + r) * cols + (c_base + c);
                let val = oob::load(target, global_idx)?;
                let local_idx = layout.linear_index(&[r, c], &tile_type.extents);
                if local_idx < elements.len() {
                    elements[local_idx] = val;
                }
            }
        }
    } else {
        for idx in 0..total_elements {
            let coords = row_major_coords(idx as u32, &tile_type.extents);
            let mut global_idx = 0u32;
            for (i, &c) in coords.iter().enumerate() {
                let base = origin_coords.get(i).copied().unwrap_or(0);
                global_idx += (base + c) * strides[i];
            }
            let val = oob::load(target, global_idx)?;
            let local_idx = layout.linear_index(&coords, &tile_type.extents);
            if local_idx < elements.len() {
                elements[local_idx] = val;
            }
        }
    }
    Ok(elements)
}

/// Store tile elements sequentially into a backing buffer starting at `origin_coords`.
///
/// # Errors
/// Propagates an out-of-bounds refusal from any element the tile writes.
pub(crate) fn store_elements(
    target: &mut Buffer,
    origin_coords: &[u32],
    elements: &[Value],
) -> Result<(), ReferenceError> {
    let base = origin_coords.first().copied().unwrap_or(0);
    for (i, elem) in elements.iter().enumerate() {
        let global_idx = base + (i as u32);
        oob::store(target, global_idx, elem)?;
    }
    Ok(())
}

/// Accumulate a matrix product `A x B` into accumulator tile elements.
///
/// Shapes come from the declared [`Tile`] of each operand. The element count
/// alone does not determine a 2-D shape: 256 elements is 16x16, 32x8, 64x4 and
/// 256x1, and this used to pick between them by rounding the square root of
/// the operand length and special-casing two literal 16x16 and 16x8 sizes. A
/// program whose tile shape fell outside those cases got a product of some
/// other shape and the oracle certified it.
///
/// # Errors
/// Refuses when an operand tile is not rank 2, when its declared extents do
/// not account for its elements, when the inner extents disagree, or when an
/// element carries no number.
pub(crate) fn matmul(
    acc_elems: &mut Vec<Value>,
    acc_shape: &Tile,
    a_elems: &[Value],
    a_shape: &Tile,
    b_elems: &[Value],
    b_shape: &Tile,
) -> Result<(), ReferenceError> {
    let (m, k) = matrix_extents("a", a_shape, a_elems.len())?;
    let (b_rows, n) = matrix_extents("b", b_shape, b_elems.len())?;
    if b_rows != k {
        return Err(ReferenceError::incomplete_dispatch_semantics(format!(
            "tile matmul inner extents disagree: a is {m}x{k} and b is {b_rows}x{n}. \
             Fix: declare b with {k} rows."
        )));
    }
    let (acc_rows, acc_cols) = matrix_extents("acc", acc_shape, acc_elems.len())?;
    if (acc_rows, acc_cols) != (m, n) {
        return Err(ReferenceError::incomplete_dispatch_semantics(format!(
            "tile matmul accumulator is {acc_rows}x{acc_cols} but the product is {m}x{n}. \
             Fix: declare the accumulator tile with extents [{m}, {n}]."
        )));
    }

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f64;
            for p in 0..k {
                sum += numeric_element("a", a_elems, i * k + p)?
                    * numeric_element("b", b_elems, p * n + j)?;
            }
            let acc_idx = i * n + j;
            let prev = numeric_element("acc", acc_elems, acc_idx)?;
            acc_elems[acc_idx] = Value::Float(prev + sum);
        }
    }
    Ok(())
}

/// Reduce tile elements along an axis or globally.
///
/// The row and column counts come from the declared [`Tile`], for the reason
/// [`matmul`] gives. The operator semantics come from [`SubgroupReduceOp`],
/// which owns them for every host-side evaluation, so a new operator is a
/// build failure in the spec crate rather than an unannounced sum here.
///
/// # Errors
/// Refuses when the declared extents do not account for the elements, when
/// `axis` names no declared extent, when an element carries no number, or when
/// a bitwise operator is applied to a float tile.
pub(crate) fn reduce(
    elements: &[Value],
    shape: &Tile,
    op: SubgroupReduceOp,
    axis: u32,
) -> Result<Vec<Value>, ReferenceError> {
    let extents = declared_extents("reduce input", shape, elements.len())?;
    let axis_index = usize::try_from(axis).map_err(|_| {
        ReferenceError::incomplete_dispatch_semantics(format!(
            "tile reduce axis {axis} exceeds the addressable range. Fix: reduce over a declared axis."
        ))
    })?;
    if axis_index >= extents.len() {
        return Err(ReferenceError::incomplete_dispatch_semantics(format!(
            "tile reduce over axis {axis} of a rank-{} tile. Fix: reduce over an axis the tile declares.",
            extents.len()
        )));
    }

    match extents.len() {
        1 => Ok(vec![reduce_slice(elements, shape, op)?]),
        2 => {
            let (rows, cols) = (extents[0], extents[1]);
            if axis_index == 1 {
                (0..rows)
                    .map(|r| reduce_slice(&elements[r * cols..(r + 1) * cols], shape, op))
                    .collect()
            } else {
                (0..cols)
                    .map(|c| {
                        let column: Vec<Value> =
                            (0..rows).map(|r| elements[r * cols + c].clone()).collect();
                        reduce_slice(&column, shape, op)
                    })
                    .collect()
            }
        }
        rank => Err(ReferenceError::incomplete_dispatch_semantics(format!(
            "tile reduce over a rank-{rank} tile has no reference semantics. \
             Fix: reduce a rank-1 or rank-2 tile."
        ))),
    }
}

/// Fold one contiguous run of tile elements under `op`.
///
/// The declared element type selects the arithmetic: an integer tile folds
/// through [`SubgroupReduceOp::reduce_u32`], a float tile folds from
/// [`SubgroupReduceOp::f32_identity`] through
/// [`SubgroupReduceOp::combine_f32`], canonicalizing each step the way the
/// subgroup reduction does. A bitwise operator has no float fold and is
/// refused rather than reinterpreted through an integer cast.
fn reduce_slice(
    slice: &[Value],
    shape: &Tile,
    op: SubgroupReduceOp,
) -> Result<Value, ReferenceError> {
    if op.is_bitwise() {
        if is_float_element(&shape.element) {
            return Err(ReferenceError::type_mismatch(format!(
                "bitwise tile reduce `{}` over a {:?} tile. Fix: reduce an integer tile, or use an arithmetic operator.",
                op.as_str(),
                shape.element
            )));
        }
        let mut lanes = Vec::with_capacity(slice.len());
        for (index, element) in slice.iter().enumerate() {
            lanes.push(element.try_as_u32().ok_or_else(|| {
                ReferenceError::type_mismatch(format!(
                    "bitwise tile reduce element {index} is {element:?}, which carries no u32. \
                     Fix: reduce a tile of integer elements."
                ))
            })?);
        }
        return Ok(Value::U32(op.reduce_u32(lanes)));
    }

    let identity = op.f32_identity().ok_or_else(|| {
        ReferenceError::incomplete_dispatch_semantics(format!(
            "tile reduce operator `{}` has no float fold. Fix: add reference semantics for it.",
            op.as_str()
        ))
    })?;
    let mut acc = crate::execution::typed_ops::canonical_f32(identity);
    for index in 0..slice.len() {
        let lane = crate::execution::typed_ops::canonical_f32(
            numeric_element("reduce input", slice, index)? as f32,
        );
        let combined = op.combine_f32(acc, lane).ok_or_else(|| {
            ReferenceError::incomplete_dispatch_semantics(format!(
                "tile reduce operator `{}` has no float fold. Fix: add reference semantics for it.",
                op.as_str()
            ))
        })?;
        acc = crate::execution::typed_ops::canonical_f32(combined);
    }
    Ok(Value::Float(f64::from(acc)))
}

/// True when a tile of `element` holds floating-point values.
///
/// The match has no catch-all so a new element type states its own answer.
fn is_float_element(element: &vyre_foundation::ir::DataType) -> bool {
    use vyre_foundation::ir::DataType as Dt;
    match element {
        Dt::F16 | Dt::BF16 | Dt::F32 | Dt::F64 | Dt::F8E4M3 | Dt::F8E5M2 | Dt::FP4 | Dt::NF4 => {
            true
        }
        Dt::U8
        | Dt::U16
        | Dt::U32
        | Dt::I8
        | Dt::I16
        | Dt::I32
        | Dt::I64
        | Dt::U64
        | Dt::I4
        | Dt::Bool
        | Dt::Bytes
        | Dt::Vec2U32
        | Dt::Vec4U32
        | Dt::Array { .. }
        | Dt::Vec { .. }
        | Dt::Tensor
        | Dt::TensorShaped { .. }
        | Dt::SparseCsr { .. }
        | Dt::SparseCoo { .. }
        | Dt::SparseBsr { .. }
        | Dt::DeviceMesh { .. }
        | Dt::Quantized { .. }
        | Dt::Handle(_)
        | Dt::Opaque(_) => false,
    }
}

/// The declared extents of `shape`, checked against the elements present.
fn declared_extents(
    role: &str,
    shape: &Tile,
    len: usize,
) -> Result<Vec<usize>, ReferenceError> {
    if shape.extents.is_empty() {
        return Err(ReferenceError::incomplete_dispatch_semantics(format!(
            "tile {role} declares no extents. Fix: declare the tile with its extents."
        )));
    }
    let extents: Vec<usize> = shape.extents.iter().map(|e| *e as usize).collect();
    let declared: usize = extents.iter().product();
    if declared != len {
        return Err(ReferenceError::incomplete_dispatch_semantics(format!(
            "tile {role} declares {:?} ({declared} elements) but holds {len}. \
             Fix: declare extents that account for every element.",
            shape.extents
        )));
    }
    Ok(extents)
}

/// The rows and columns of a rank-2 operand.
fn matrix_extents(role: &str, shape: &Tile, len: usize) -> Result<(usize, usize), ReferenceError> {
    let extents = declared_extents(role, shape, len)?;
    match extents.as_slice() {
        [rows, cols] => Ok((*rows, *cols)),
        other => Err(ReferenceError::incomplete_dispatch_semantics(format!(
            "tile matmul operand `{role}` is rank {} but matmul takes rank-2 operands. \
             Fix: declare `{role}` with two extents.",
            other.len()
        ))),
    }
}

/// The number one tile element carries, or a refusal.
fn numeric_element(role: &str, elements: &[Value], index: usize) -> Result<f64, ReferenceError> {
    let element = elements.get(index).ok_or_else(|| {
        ReferenceError::incomplete_dispatch_semantics(format!(
            "tile `{role}` element {index} is past the {} elements it holds. \
             Fix: declare extents that match the tile contents.",
            elements.len()
        ))
    })?;
    element.try_as_f64().ok_or_else(|| {
        ReferenceError::type_mismatch(format!(
            "tile `{role}` element {index} is {element:?}, which carries no number. \
             Fix: populate the tile with numeric elements."
        ))
    })
}
