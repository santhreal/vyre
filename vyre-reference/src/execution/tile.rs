//! Canonical shared execution logic for Tile operations.
//!
//! Provides the single-homed reference implementation for tile load, store,
//! element conversion, matrix multiplication, and reduction across both
//! sequential and hashmap execution engines.

use vyre_foundation::ir::{Layout, SubgroupReduceOp, Tile};

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
pub(crate) fn load_elements(
    target: &Buffer,
    origin_coords: &[u32],
    tile_type: &Tile,
    layout: &Layout,
) -> Vec<Value> {
    let total_elements = tile_type.element_count();
    let mut elements = vec![Value::Float(0.0); total_elements];

    let mut strides = vec![1u32; tile_type.extents.len()];
    for i in (0..tile_type.extents.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * tile_type.extents[i + 1];
    }

    if tile_type.extents.is_empty() {
        let global_idx = origin_coords.first().copied().unwrap_or(0);
        let val = oob::load(target, global_idx);
        elements = vec![val];
    } else if tile_type.extents.len() == 1 {
        let n = tile_type.extents[0];
        let base = origin_coords.first().copied().unwrap_or(0);
        for i in 0..n {
            let global_idx = base + i;
            let val = oob::load(target, global_idx);
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
                let val = oob::load(target, global_idx);
                let local_idx = layout.linear_index(&[r, c], &tile_type.extents);
                if local_idx < elements.len() {
                    elements[local_idx] = val;
                }
            }
        }
    } else {
        for idx in 0..total_elements {
            let mut coords = Vec::with_capacity(tile_type.extents.len());
            let mut temp = idx as u32;
            for &extent in tile_type.extents.iter().rev() {
                coords.push(temp % extent);
                temp /= extent;
            }
            coords.reverse();
            let mut global_idx = 0u32;
            for (i, &c) in coords.iter().enumerate() {
                let base = origin_coords.get(i).copied().unwrap_or(0);
                global_idx += (base + c) * strides[i];
            }
            let val = oob::load(target, global_idx);
            let local_idx = layout.linear_index(&coords, &tile_type.extents);
            if local_idx < elements.len() {
                elements[local_idx] = val;
            }
        }
    }
    elements
}

/// Store tile elements sequentially into a backing buffer starting at `origin_coords`.
pub(crate) fn store_elements(target: &mut Buffer, origin_coords: &[u32], elements: &[Value]) {
    let base = origin_coords.first().copied().unwrap_or(0);
    for (i, elem) in elements.iter().enumerate() {
        let global_idx = base + (i as u32);
        oob::store(target, global_idx, elem);
    }
}

/// Accumulate a matrix product `A x B` into accumulator tile elements.
pub(crate) fn matmul(acc_elems: &mut Vec<Value>, a_elems: &[Value], b_elems: &[Value]) {
    let a_len = a_elems.len();
    let b_len = b_elems.len();
    let (m, k, n) = if a_len == 16 * 16 && b_len == 16 * 8 {
        (16, 16, 8)
    } else if a_len == 16 * 8 && b_len == 8 * 16 {
        (16, 8, 16)
    } else {
        let k = (a_len as f64).sqrt().round() as usize;
        let k = if k == 0 { 1 } else { k };
        let m = a_len / k;
        let n = if k > 0 { b_len / k } else { 1 };
        (m.max(1), k.max(1), n.max(1))
    };

    if acc_elems.len() < m * n {
        acc_elems.resize(m * n, Value::Float(0.0));
    }

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f64;
            for p in 0..k {
                let a_idx = i * k + p;
                let b_idx = p * n + j;
                let a_num = a_elems
                    .get(a_idx)
                    .and_then(|v| v.try_as_f64())
                    .unwrap_or(0.0);
                let b_num = b_elems
                    .get(b_idx)
                    .and_then(|v| v.try_as_f64())
                    .unwrap_or(0.0);
                sum += a_num * b_num;
            }
            let acc_idx = i * n + j;
            let prev = acc_elems
                .get(acc_idx)
                .and_then(|v| v.try_as_f64())
                .unwrap_or(0.0);
            if acc_idx < acc_elems.len() {
                acc_elems[acc_idx] = Value::Float(prev + sum);
            }
        }
    }
}

/// Reduce tile elements along an axis or globally.
pub(crate) fn reduce(elements: &[Value], op: SubgroupReduceOp, axis: u32) -> Vec<Value> {
    if elements.is_empty() {
        return vec![Value::Float(0.0)];
    }

    let reduce_slice = |slice: &[Value]| -> f64 {
        if slice.is_empty() {
            return 0.0;
        }
        let mut acc = slice[0].try_as_f64().unwrap_or(0.0);
        for elem in slice.iter().skip(1) {
            let val = elem.try_as_f64().unwrap_or(0.0);
            acc = match op {
                SubgroupReduceOp::Add => acc + val,
                SubgroupReduceOp::Mul => acc * val,
                SubgroupReduceOp::Min => acc.min(val),
                SubgroupReduceOp::Max => acc.max(val),
                SubgroupReduceOp::And => ((acc as u64) & (val as u64)) as f64,
                SubgroupReduceOp::Or => ((acc as u64) | (val as u64)) as f64,
                SubgroupReduceOp::Xor => ((acc as u64) ^ (val as u64)) as f64,
                _ => acc + val,
            };
        }
        acc
    };

    let total = elements.len();
    let dim = (total as f64).sqrt().round() as usize;
    let (rows, cols) = if dim * dim == total && dim > 0 {
        (dim, dim)
    } else if total % 2 == 0 {
        (total / 2, 2)
    } else {
        (total, 1)
    };

    if axis == 1 && rows > 0 && cols > 0 && rows * cols == total {
        let mut res = Vec::with_capacity(rows);
        for r in 0..rows {
            let slice = &elements[r * cols..(r + 1) * cols];
            res.push(Value::Float(reduce_slice(slice)));
        }
        res
    } else if axis == 0 && rows > 0 && cols > 0 && rows * cols == total {
        let mut res = Vec::with_capacity(cols);
        for c in 0..cols {
            let col_vals: Vec<Value> = (0..rows).map(|r| elements[r * cols + c].clone()).collect();
            res.push(Value::Float(reduce_slice(&col_vals)));
        }
        res
    } else {
        vec![Value::Float(reduce_slice(elements))]
    }
}
