//! Tensor-train compression of the dispatch-graph cost tensor.
//!
//! Self-consumer for [#12 `tensor_train_decompose`](vyre_primitives::math::tensor_train_decompose).
//!
//! The dispatch-graph cost tensor (per-Region × per-buffer × per-config
//! cost) grows with the cube of the dispatch size. For a 1k-region
//! Program with 256 configs and 32 buffers, that's 8M f64 cells  -
//! 64MB resident in the autotuner. TT-decomposition compresses this
//! along each mode (region / buffer / config) into a small set of
//! "core" tensors with TT-rank that bounds the approximation error.
//!
//! Used by:
//! - The differentiable autotuner: store costs in TT form so the
//!   derivative loop reads compressed cores instead of full tensor.
//! - The cost-model self-consumer: TT-compressed cost lookup is O(1)
//!   per query vs O(n) for raw tensor traversal.

use crate::dispatch_buffers::{
    decode_f32_output_exact, ensure_input_slots, write_f32_slice_le_bytes, write_zero_bytes,
};
use crate::optimizer::dispatcher::{DispatchError, OptimizerDispatcher};
use vyre_primitives::math::tensor_train_decompose::tensor_train_decompose_step;

/// Compressed cost tensor in tensor-train form.
///
/// `cores[k]` is the k-th TT core; the original cost tensor is
/// reconstructed by chained matrix-vector contraction
/// `T(i_1, ..., i_d) = ∏ cores[k][r_k, i_k, r_{k+1}]`.
#[derive(Debug, Clone)]
pub struct CompressedCostTensor {
    /// TT cores in dispatch-graph mode order.
    pub cores: Vec<Vec<f64>>,
    /// Per-mode dimensions (e.g. [n_regions, n_buffers, n_configs]).
    pub dims: Vec<u32>,
    /// TT-ranks (length `dims.len() + 1`, with `ranks[0] = ranks[d] = 1`).
    pub ranks: Vec<u32>,
}

/// f32 compressed cost tensor produced by the dispatchable TT-SVD step primitive.
#[derive(Debug, Clone, PartialEq)]
pub struct CompressedF32CostTensor {
    /// f32 TT cores in dispatch-graph mode order.
    pub cores: Vec<Vec<f32>>,
    /// Per-mode dimensions.
    pub dims: Vec<u32>,
    /// TT-ranks.
    pub ranks: Vec<u32>,
}

/// Caller-owned GPU dispatch scratch for f32 tensor-train compression.
#[derive(Debug, Default)]
pub struct TensorTrainCompressionGpuScratch {
    current: Vec<f32>,
    remainder: Vec<f32>,
    inputs: Vec<Vec<u8>>,
}

/// Compress an f32 cost tensor through dispatchable TT-SVD steps.
///
/// This path chains [`tensor_train_decompose_step`] (a real per-mode truncated SVD in f32) once per
/// non-final mode and stores the final remainder as the last core. It is the GPU-dispatchable
/// production path; [`reference_compress_cost_tensor`] remains the f64 CPU reference TT-SVD.
pub fn compress_cost_tensor_f32_via(
    dispatcher: &dyn OptimizerDispatcher,
    tensor_f32: &[f32],
    dims: &[u32],
    target_ranks: &[u32],
) -> Result<CompressedF32CostTensor, DispatchError> {
    let mut cores = Vec::with_capacity(dims.len());
    compress_cost_tensor_f32_via_into(dispatcher, tensor_f32, dims, target_ranks, &mut cores)?;
    Ok(CompressedF32CostTensor {
        cores,
        dims: dims.to_vec(),
        ranks: target_ranks.to_vec(),
    })
}

/// Compress an f32 cost tensor through dispatchable TT-SVD steps into caller-owned core storage.
pub fn compress_cost_tensor_f32_via_into(
    dispatcher: &dyn OptimizerDispatcher,
    tensor_f32: &[f32],
    dims: &[u32],
    target_ranks: &[u32],
    cores_out: &mut Vec<Vec<f32>>,
) -> Result<(), DispatchError> {
    let mut scratch = TensorTrainCompressionGpuScratch::default();
    compress_cost_tensor_f32_via_with_scratch_into(
        dispatcher,
        tensor_f32,
        dims,
        target_ranks,
        &mut scratch,
        cores_out,
    )
}

/// Compress an f32 cost tensor through dispatchable TT-SVD steps into
/// caller-owned dispatch scratch and core storage.
pub fn compress_cost_tensor_f32_via_with_scratch_into(
    dispatcher: &dyn OptimizerDispatcher,
    tensor_f32: &[f32],
    dims: &[u32],
    target_ranks: &[u32],
    scratch: &mut TensorTrainCompressionGpuScratch,
    cores_out: &mut Vec<Vec<f32>>,
) -> Result<(), DispatchError> {
    use crate::observability::{bump, tensor_train_compression_calls};
    bump(&tensor_train_compression_calls);

    validate_tt_shape(tensor_f32, dims, target_ranks)?;
    if dims.is_empty() {
        cores_out.truncate(0);
        return Ok(());
    }
    if dims.len() == 1 {
        ensure_core_slot(cores_out, 0);
        cores_out[0].clear();
        cores_out[0].extend_from_slice(tensor_f32);
        cores_out.truncate(1);
        return Ok(());
    }

    scratch.current.clear();
    scratch.current.extend_from_slice(tensor_f32);
    let mut r_prev = target_ranks[0];
    for mode in 0..(dims.len() - 1) {
        let nk = dims[mode];
        let r_next = target_ranks[mode + 1];
        let input_rows = checked_mul_u32(r_prev, nk, "r_prev", "nk")?;
        let input_rows_usize = input_rows as usize;
        if input_rows_usize == 0 || scratch.current.len() % input_rows_usize != 0 {
            return Err(DispatchError::BadInputs(format!(
                "Fix: compress_cost_tensor_f32_via mode {mode} expected current.len() divisible by r_prev*nk={input_rows}, got {}.",
                scratch.current.len()
            )));
        }
        let rem = u32::try_from(scratch.current.len() / input_rows_usize).map_err(|_| {
            DispatchError::BadInputs(format!(
                "Fix: compress_cost_tensor_f32_via mode {mode} remainder column count exceeds u32."
            ))
        })?;
        let core_words = checked_product3_usize(r_prev, nk, r_next, "core")?;
        let rem_words = checked_mul_usize(r_next, rem, "remainder")?;
        let program = tensor_train_decompose_step(
            "input_matrix",
            "u_out",
            "rem_out",
            r_prev,
            nk,
            rem,
            r_next,
        );
        // Real-backend dispatch-input contract (vyre-driver `role_for_buffer`): one input per
        // INPUT-CONSUMING buffer in buffer order. The step declares six such buffers. `input_matrix`
        // RO (0) then the plain-ReadWrite `u_out` (1, m*r_next), `rem_out` (2, r_next*rem), and the
        // internal `tt_ata`/`tt_evec` Gram+eigenvector (3,4, rem*rem each) and `tt_eval` (5, rem)
        // scratch. Every plain-RW buffer needs a zero-filled input slot for its initial contents (the
        // lane-0-serial kernel writes them all); passing only `input_matrix` would fail the backend's
        // strict `validate_input_lengths` count check.
        let rem_usize = rem as usize;
        let gram_words = checked_mul_usize(rem, rem, "gram")?;
        let f32_bytes = std::mem::size_of::<f32>();
        let bytes = |words: usize| words * f32_bytes;
        ensure_input_slots(&mut scratch.inputs, 6);
        write_f32_slice_le_bytes(&mut scratch.inputs[0], &scratch.current);
        write_zero_bytes(&mut scratch.inputs[1], bytes(core_words));
        write_zero_bytes(&mut scratch.inputs[2], bytes(rem_words));
        write_zero_bytes(&mut scratch.inputs[3], bytes(gram_words));
        write_zero_bytes(&mut scratch.inputs[4], bytes(gram_words));
        write_zero_bytes(&mut scratch.inputs[5], bytes(rem_usize));
        let outputs = dispatcher.dispatch(&program, &scratch.inputs[..6], Some([1, 1, 1]))?;
        // The kernel's writable buffers in binding order are u_out (core), rem_out (S·Vᵀ), then the
        // internal scratch; a faithful backend returns at least the first two, which are what we
        // decode. Trailing scratch outputs are ignored.
        if outputs.len() < 2 {
            return Err(DispatchError::BackendError(format!(
                "Fix: compress_cost_tensor_f32_via expected at least the u_out + rem_out outputs, got {}.",
                outputs.len()
            )));
        }
        ensure_core_slot(cores_out, mode);
        decode_f32_output_exact(
            &outputs[0],
            core_words,
            "compress_cost_tensor_f32_via u_out",
            &mut cores_out[mode],
        )?;
        decode_f32_output_exact(
            &outputs[1],
            rem_words,
            "compress_cost_tensor_f32_via rem_out",
            &mut scratch.remainder,
        )?;
        std::mem::swap(&mut scratch.current, &mut scratch.remainder);
        r_prev = r_next;
    }
    let last = dims.len() - 1;
    ensure_core_slot(cores_out, last);
    cores_out[last].clear();
    cores_out[last].extend_from_slice(&scratch.current);
    cores_out.truncate(dims.len());
    if scratch.current.capacity() < tensor_f32.len() {
        scratch
            .current
            .try_reserve_exact(tensor_f32.len() - scratch.current.capacity())
            .map_err(|error| {
                DispatchError::BackendError(format!(
                    "Fix: compress_cost_tensor_f32_via could not retain current scratch capacity for {} word(s): {error}.",
                    tensor_f32.len()
                ))
            })?;
    }
    Ok(())
}

fn ensure_core_slot(cores: &mut Vec<Vec<f32>>, slot: usize) {
    while cores.len() <= slot {
        cores.push(Vec::new());
    }
}

fn validate_tt_shape(tensor: &[f32], dims: &[u32], ranks: &[u32]) -> Result<(), DispatchError> {
    if dims.iter().any(|&dim| dim == 0) {
        return Err(DispatchError::BadInputs(
            "Fix: compress_cost_tensor_f32_via requires all dims > 0.".to_string(),
        ));
    }
    if ranks.len() != dims.len() + 1 {
        return Err(DispatchError::BadInputs(format!(
            "Fix: compress_cost_tensor_f32_via expected ranks.len() == dims.len()+1 == {}, got {}.",
            dims.len() + 1,
            ranks.len()
        )));
    }
    if ranks.first().copied().unwrap_or(0) != 1 || ranks.last().copied().unwrap_or(0) != 1 {
        return Err(DispatchError::BadInputs(
            "Fix: compress_cost_tensor_f32_via requires boundary ranks ranks[0] == ranks[d] == 1."
                .to_string(),
        ));
    }
    if ranks.iter().any(|&rank| rank == 0) {
        return Err(DispatchError::BadInputs(
            "Fix: compress_cost_tensor_f32_via requires all ranks > 0.".to_string(),
        ));
    }
    let expected = dims
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim as usize))
        .ok_or_else(|| {
            DispatchError::BadInputs(
                "Fix: compress_cost_tensor_f32_via dims product overflows usize.".to_string(),
            )
        })?;
    if tensor.len() != expected {
        return Err(DispatchError::BadInputs(format!(
            "Fix: compress_cost_tensor_f32_via expected tensor_f32.len() == dims product == {expected}, got {}.",
            tensor.len()
        )));
    }
    Ok(())
}

fn checked_mul_u32(
    left: u32,
    right: u32,
    left_name: &str,
    right_name: &str,
) -> Result<u32, DispatchError> {
    left.checked_mul(right).ok_or_else(|| {
        DispatchError::BadInputs(format!(
            "Fix: compress_cost_tensor_f32_via {left_name}*{right_name} overflows u32: {left}*{right}."
        ))
    })
}

fn checked_mul_usize(left: u32, right: u32, context: &str) -> Result<usize, DispatchError> {
    checked_mul_u32(left, right, "left", "right")
        .map(|value| value as usize)
        .map_err(|_| {
            DispatchError::BadInputs(format!(
                "Fix: compress_cost_tensor_f32_via {context} word count overflows usize."
            ))
        })
}

fn checked_product3_usize(a: u32, b: u32, c: u32, context: &str) -> Result<usize, DispatchError> {
    let ab = checked_mul_u32(a, b, "a", "b")?;
    checked_mul_u32(ab, c, "a*b", "c")
        .map(|value| value as usize)
        .map_err(|_| {
            DispatchError::BadInputs(format!(
                "Fix: compress_cost_tensor_f32_via {context} word count overflows usize."
            ))
        })
}

/// Approximate the original cost tensor's compression ratio:
/// `(1 - tt_size / original_size)`  -  a value in `[0, 1]` where 0
/// means no compression and 1 means full elimination.
#[must_use]
pub fn compression_ratio(compressed: &CompressedCostTensor) -> f64 {
    let original_size: usize = if compressed.dims.is_empty() {
        0
    } else {
        compressed.dims.iter().map(|d| *d as usize).product()
    };
    if original_size == 0 {
        return 0.0;
    }
    let tt_size: usize = compressed.cores.iter().map(Vec::len).sum();
    1.0 - (tt_size as f64) / (original_size as f64)
}

/// Total entries the TT representation stores. Useful for
/// observability  -  emit alongside cache size metrics so operators
/// can verify TT compression is actually shrinking memory.
#[must_use]
pub fn tt_storage_size(compressed: &CompressedCostTensor) -> usize {
    compressed.cores.iter().map(Vec::len).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    // The dispatchable f32 TT-SVD path is validated end-to-end (through a real reference-eval
    // dispatch, reconstructing the tensor) in
    // `tests/tensor_train_compress_via_reference_parity.rs`: not by a mock dispatcher here.

    #[test]
    fn compresses_3_mode_tensor() {
        // 2×3×2 cost tensor flattened row-major = 12 entries.
        let dims = vec![2u32, 3, 2];
        let target_ranks = vec![1u32, 2, 2, 1];
        let tensor: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let compressed = reference_compress_cost_tensor(&tensor, &dims, &target_ranks);
        assert_eq!(compressed.cores.len(), 3); // d cores
        assert_eq!(compressed.dims, dims);
    }

    #[test]
    fn compression_ratio_is_in_unit_interval() {
        let dims = vec![4u32, 4];
        let target_ranks = vec![1u32, 2, 1];
        let tensor = vec![1.0; 16];
        let compressed = reference_compress_cost_tensor(&tensor, &dims, &target_ranks);
        let ratio = compression_ratio(&compressed);
        assert!(
            (-1.0..=1.0).contains(&ratio),
            "ratio out of expected range: {ratio}"
        );
    }

    #[test]
    fn production_source_does_not_call_cpu_tensor_train_decompose_helper() {
        let source = include_str!("tensor_train_compression.rs");
        let cutoff = [
            source.find("#[cfg(test)]"),
            source.find("/// Parity-only f64 TT-SVD CPU oracle"),
        ]
        .into_iter()
        .flatten()
        .min()
        .expect("Fix: source includes an explicit non-production cutoff marker");
        let production_source = &source[..cutoff];
        assert!(
            !production_source.contains("cpu_ref(")
                && !production_source.contains("reference_compress_cost_tensor("),
            "Fix: tensor-train compression production paths must dispatch tensor_train_decompose_step, not CPU TT-SVD helpers."
        );
    }

    #[test]
    fn tt_storage_size_returns_sum() {
        let compressed = CompressedCostTensor {
            cores: vec![vec![1.0; 4], vec![1.0; 8], vec![1.0; 4]],
            dims: vec![2, 4, 2],
            ranks: vec![1, 2, 2, 1],
        };
        assert_eq!(tt_storage_size(&compressed), 16);
    }

    #[test]
    fn empty_dims_handled() {
        let compressed = CompressedCostTensor {
            cores: Vec::new(),
            dims: Vec::new(),
            ranks: vec![1],
        };
        assert_eq!(tt_storage_size(&compressed), 0);
        assert_eq!(compression_ratio(&compressed), 0.0);
    }
}

/// Parity-only f64 TT-SVD CPU oracle for compressing a flat cost tensor.
///
/// Production callers must use [`compress_cost_tensor_f32_via`] or
/// [`compress_cost_tensor_f32_via_with_scratch_into`], which dispatch
/// [`tensor_train_decompose_step`] through the selected backend.
///
/// # Panics
///
/// Panics if `target_ranks.len() != dims.len() + 1`, if the boundary
/// ranks are not 1, or if `tensor.len()` doesn't match the dim
/// product.
#[cfg(any(test, feature = "cpu-parity"))]
#[must_use]

pub fn reference_compress_cost_tensor(
    tensor: &[f64],
    dims: &[u32],
    target_ranks: &[u32],
) -> CompressedCostTensor {
    use crate::observability::{bump, tensor_train_compression_calls};
    bump(&tensor_train_compression_calls);
    let cores = vyre_primitives::math::tensor_train_decompose::cpu_ref(tensor, dims, target_ranks);
    CompressedCostTensor {
        cores,
        dims: dims.to_vec(),
        ranks: target_ranks.to_vec(),
    }
}
