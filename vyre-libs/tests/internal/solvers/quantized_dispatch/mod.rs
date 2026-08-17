mod batched_matmul_contracts;
mod batched_matmul_top1_contracts;
mod batched_matvec_contracts;
mod dot_contracts;
mod generated_contracts;
mod matvec_contracts;
mod unpack_contracts;

use super::*;
use vyre_foundation::program_dispatch::{DispatchError, ProgramDispatcher};

struct QuantizedDispatcher;

impl ProgramDispatcher for QuantizedDispatcher {
    fn dispatch(
        &self,
        _program: &Program,
        inputs: &[Vec<u8>],
        grid_override: Option<[u32; 3]>,
    ) -> Result<Vec<Vec<u8>>, DispatchError> {
        assert_eq!(grid_override, Some([1, 1, 1]));
        assert_eq!(inputs.len(), 2);
        let packed = crate::dispatch_buffers::read_u32s(&inputs[0]);
        let lane_count = inputs[1].len() / std::mem::size_of::<i32>();
        let mut out = Vec::new();
        unpack_i4x8_cpu_into(&packed, lane_count as u32, &mut out);
        Ok(vec![vyre_primitives::wire::pack_i32_slice(&out)])
    }
}

struct QuantizedDotDispatcher;

impl ProgramDispatcher for QuantizedDotDispatcher {
    fn dispatch(
        &self,
        _program: &Program,
        inputs: &[Vec<u8>],
        grid_override: Option<[u32; 3]>,
    ) -> Result<Vec<Vec<u8>>, DispatchError> {
        assert_eq!(grid_override, Some([1, 1, 1]));
        // Four input-consuming buffers (lhs/rhs/lhs_scale/rhs_scale RO); `out` is backend-allocated.
        assert_eq!(inputs.len(), 4);
        let lhs = crate::dispatch_buffers::read_u32s(&inputs[0]);
        let rhs = crate::dispatch_buffers::read_u32s(&inputs[1]);
        let lhs_scale = crate::dispatch_buffers::read_f32s(&inputs[2])[0];
        let rhs_scale = crate::dispatch_buffers::read_f32s(&inputs[3])[0];
        let logical_lane_count = (lhs.len() as u32 - 1) * 8
            + if lhs.last().copied().unwrap_or(0) == 0 {
                8
            } else {
                8
            };
        let lane_count = logical_lane_count.min((lhs.len() as u32) * 8);
        let out = i4x8_dot_f32_scaled_cpu(&lhs, &rhs, lhs_scale, rhs_scale, lane_count);
        Ok(vec![vyre_primitives::wire::pack_f32_slice(&[out])])
    }
}

struct QuantizedMatvecDispatcher;

impl ProgramDispatcher for QuantizedMatvecDispatcher {
    fn dispatch(
        &self,
        _program: &Program,
        inputs: &[Vec<u8>],
        grid_override: Option<[u32; 3]>,
    ) -> Result<Vec<Vec<u8>>, DispatchError> {
        // Three input-consuming buffers (weights/x/row_scales RO); `out` is backend-allocated.
        assert_eq!(inputs.len(), 3);
        let weights = crate::dispatch_buffers::read_u32s(&inputs[0]);
        let x = crate::dispatch_buffers::read_f32s(&inputs[1]);
        let row_scales = crate::dispatch_buffers::read_f32s(&inputs[2]);
        let rows = row_scales.len() as u32;
        let cols = x.len() as u32;
        assert_eq!(grid_override, Some([rows, 1, 1]));
        let out = i4x8_matvec_f32_scaled_cpu(&weights, &x, &row_scales, rows, cols);
        Ok(vec![vyre_primitives::wire::pack_f32_slice(&out)])
    }
}

struct QuantizedBatchedMatvecDispatcher;

impl ProgramDispatcher for QuantizedBatchedMatvecDispatcher {
    fn dispatch(
        &self,
        _program: &Program,
        inputs: &[Vec<u8>],
        grid_override: Option<[u32; 3]>,
    ) -> Result<Vec<Vec<u8>>, DispatchError> {
        // Three input-consuming buffers (weights/x_batches/row_scales RO); `out` is backend-allocated.
        assert_eq!(inputs.len(), 3);
        let weights = crate::dispatch_buffers::read_u32s(&inputs[0]);
        let x_batches = crate::dispatch_buffers::read_f32s(&inputs[1]);
        let row_scales = crate::dispatch_buffers::read_f32s(&inputs[2]);
        let Some([rows, batch, 1]) = grid_override else {
            panic!("Fix: batched matvec dispatch must launch with [rows, batch, 1].");
        };
        let cols = x_batches
            .len()
            .checked_div(batch as usize)
            .expect("Fix: fake batched matvec dispatcher requires nonzero batch")
            as u32;
        assert_eq!(rows as usize, row_scales.len());
        let out = i4x8_batched_matvec_f32_scaled_cpu(
            &weights,
            &x_batches,
            &row_scales,
            batch,
            rows,
            cols,
        );
        Ok(vec![vyre_primitives::wire::pack_f32_slice(&out)])
    }
}

struct QuantizedBatchedMatmulDispatcher;

impl ProgramDispatcher for QuantizedBatchedMatmulDispatcher {
    fn dispatch(
        &self,
        _program: &Program,
        inputs: &[Vec<u8>],
        grid_override: Option<[u32; 3]>,
    ) -> Result<Vec<Vec<u8>>, DispatchError> {
        // Four input-consuming buffers (weights/activations/row_scales/batch_scales RO); `out` is
        // backend-allocated.
        assert_eq!(inputs.len(), 4);
        let weights = crate::dispatch_buffers::read_u32s(&inputs[0]);
        let activations = crate::dispatch_buffers::read_u32s(&inputs[1]);
        let row_scales = crate::dispatch_buffers::read_f32s(&inputs[2]);
        let batch_scales = crate::dispatch_buffers::read_f32s(&inputs[3]);
        let rows = row_scales.len() as u32;
        let batch = batch_scales.len() as u32;
        let Some([grid_x, 1, 1]) = grid_override else {
            panic!(
                "Fix: batched matmul dispatch must launch one-dimensional 64-wide workgroup grid."
            );
        };
        assert_eq!(grid_x, ceil_div_u32(batch * rows, 64));
        let words_per_activation = activations.len() / batch as usize;
        let cols = (words_per_activation as u32) * 8;
        let out = i4x8_batched_matmul_f32_scaled_cpu(
            &weights,
            &activations,
            &row_scales,
            &batch_scales,
            batch,
            rows,
            cols,
        );
        Ok(vec![vyre_primitives::wire::pack_f32_slice(&out)])
    }
}

struct QuantizedBatchedMatmulTop1Dispatcher;

impl ProgramDispatcher for QuantizedBatchedMatmulTop1Dispatcher {
    fn dispatch(
        &self,
        _program: &Program,
        inputs: &[Vec<u8>],
        grid_override: Option<[u32; 3]>,
    ) -> Result<Vec<Vec<u8>>, DispatchError> {
        // Four input-consuming buffers (weights/activations/row_scales/batch_scales RO); the single
        // `out` buffer is backend-allocated.
        assert_eq!(inputs.len(), 4);
        let weights = crate::dispatch_buffers::read_u32s(&inputs[0]);
        let activations = crate::dispatch_buffers::read_u32s(&inputs[1]);
        let row_scales = crate::dispatch_buffers::read_f32s(&inputs[2]);
        let batch_scales = crate::dispatch_buffers::read_f32s(&inputs[3]);
        let rows = row_scales.len() as u32;
        let batch = batch_scales.len() as u32;
        assert_eq!(grid_override, Some([ceil_div_u32(batch, 64), 1, 1]));
        let words_per_activation = activations.len() / batch as usize;
        let cols = (words_per_activation as u32) * 8;
        let (scores, indices) = i4x8_batched_matmul_top1_f32_scaled_cpu(
            &weights,
            &activations,
            &row_scales,
            &batch_scales,
            batch,
            rows,
            cols,
        );
        // Model the real backend: ONE `batch*2` f32 output buffer, scores in the first `batch`
        // words then indices-as-f32 in the next `batch`: exactly what the
        // `i4x8_batched_matmul_top1_f32_scaled` kernel writes into `out`.
        let mut packed = Vec::with_capacity(batch as usize * 2);
        packed.extend_from_slice(&scores);
        packed.extend(indices.iter().map(|&i| i as f32));
        Ok(vec![vyre_primitives::wire::pack_f32_slice(&packed)])
    }
}

/// Every quantized dispatch entry point keys its `ProgramCache` on the shape it
/// built the Program for: two dispatches at one shape must build one Program,
/// and a third at a different shape must build a second.
///
/// `dispatch` receives the caller-owned scratch and `true` when it should use
/// the changed shape; `builds` reads that scratch's build counter.
fn assert_program_cache_keys_on_shape<S: Default>(
    entry_point: &str,
    shape_field: &str,
    builds: impl Fn(&S) -> usize,
    mut dispatch: impl FnMut(&mut S, bool),
) {
    let mut scratch = S::default();
    dispatch(&mut scratch, false);
    dispatch(&mut scratch, false);
    assert_eq!(
        builds(&scratch),
        1,
        "Fix: repeated same-shape {entry_point} dispatch must reuse the primitive Program."
    );

    dispatch(&mut scratch, true);
    assert_eq!(
        builds(&scratch),
        2,
        "Fix: {entry_point} dispatch must rebuild the primitive Program only when {shape_field} changes."
    );
}

/// Dispatch results must match the CPU oracle bit for bit, length included.
/// Zipping the two and comparing the overlap passes on a result that is short,
/// which is the failure a backend readback bug produces.
fn assert_f32_bits_eq(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(
        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "Fix: {label} must match the CPU oracle bit for bit, over the same number of values."
    );
}

fn pack_i4_rows(rows: &[&[i32]]) -> Vec<u32> {
    let mut packed = Vec::new();
    for row in rows {
        packed.extend(pack_i4x8_cpu(row));
    }
    packed
}

/// Every batched INT4 entry point validates the same five shape preconditions before it builds a
/// Program, and reports each one with the fragment paired below.
fn assert_rejects_batched_shape_errors<T>(
    call: impl Fn(&[u32], &[u32], &[f32], &[f32], u32, u32, u32) -> Result<T, DispatchError>,
) {
    let weights = pack_i4_rows(&[&[-1, 2, 3, -4, 5, -6, 7, -8]]);
    let activations = pack_i4_rows(&[&[7, 5, 3, 1, -1, -3, -5, -7], &[-8, -6, -4, -2, 0, 2, 4, 6]]);
    let row_scales = [0.5];
    let batch_scales = [0.25, 0.375];
    let cases: [(&str, &[u32], &[u32], &[f32], &[f32], u32, &str); 5] = [
        (
            "zero batch",
            &weights,
            &activations,
            &row_scales,
            &batch_scales,
            0,
            "batch > 0",
        ),
        (
            "missing weights",
            &[],
            &activations,
            &row_scales,
            &batch_scales,
            2,
            "weights_packed.len()",
        ),
        (
            "short activations",
            &weights,
            &activations[..1],
            &row_scales,
            &batch_scales,
            2,
            "activation_batches_packed.len()",
        ),
        (
            "missing row scale",
            &weights,
            &activations,
            &[],
            &batch_scales,
            2,
            "row_scales.len() == rows",
        ),
        (
            "missing batch scale",
            &weights,
            &activations,
            &row_scales,
            &batch_scales[..1],
            2,
            "batch_scales.len() == batch",
        ),
    ];

    for (label, weights, activations, row_scales, batch_scales, batch, fragment) in cases {
        let Err(err) = call(weights, activations, row_scales, batch_scales, batch, 1, 8) else {
            panic!("Fix: {label} must be rejected before dispatch.");
        };
        let message = err.to_string();
        assert!(
            message.contains(fragment),
            "Fix: {label} must report `{fragment}`, reported `{message}`."
        );
    }
}
