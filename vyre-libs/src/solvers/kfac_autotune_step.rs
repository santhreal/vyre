//! K-FAC step inside vyre's natural-gradient autotuner.
//!
//! Replaces standard gradient descent on dispatch-graph continuous
//! variables (e.g. tile sizes, fusion probabilities) with Fisher-
//! preconditioned updates.
//!
//! Dispatches the `crate::math::kfac_block_inverse` primitive
//! to invert the block-diagonal Fisher information matrix of the
//! autotuner's policy network.

use crate::math::kfac_block_inverse::kfac_block_inverse;
use vyre_foundation::ir::Program;

use crate::dispatch_buffers::{
    decode_f32_output_exact, ensure_input_slots, write_f32_slice_le_bytes, write_zero_bytes,
};
use vyre_megakernel::{
    execute_single_program, SemanticExecutionError, SemanticExecutionPolicy, SemanticExecutor,
};

/// Canonical op ID for the autotune step.
pub const OP_ID: &str = "vyre-libs::self_substrate::kfac_autotune_step";

/// Caller-owned GPU dispatch scratch for K-FAC autotune steps.
#[derive(Debug, Default)]
pub struct KfacAutotuneGpuScratch {
    inputs: Vec<Vec<u8>>,
}

/// Compile a Program that inverts the Fisher block-diagonal matrix.
///
/// `n` is the size of each block (e.g. number of parameters in a layer).
/// `num_blocks` is the number of independent layers/blocks.
#[must_use]
pub fn kfac_autotune_step_program(
    blocks_out: &str,
    blocks_in: &str,
    scratch: &str,
    num_blocks: u32,
    n: u32,
) -> Program {
    use crate::telemetry::{bump, kfac_autotune_step_calls};
    bump(&kfac_autotune_step_calls);
    kfac_block_inverse(blocks_out, blocks_in, scratch, num_blocks, n)
}

/// GPU dispatch wrapper around [`kfac_autotune_step_program`].
/// Returns the inverted block-diagonal Fisher matrix for the supplied
/// blocks.
///
/// # Errors
///
/// Propagates dispatch failures and rejects malformed dimensions or readback.
pub fn kfac_autotune_step_via(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    blocks_in: &[f32],
    num_blocks: u32,
    n: u32,
) -> Result<Vec<f32>, SemanticExecutionError> {
    let mut out = Vec::new();
    kfac_autotune_step_via_into(dispatcher, policy, blocks_in, num_blocks, n, &mut out)?;
    Ok(out)
}

/// GPU dispatch wrapper around [`kfac_autotune_step_program`] into caller-owned
/// output storage.
///
/// # Errors
///
/// Propagates dispatch failures and rejects malformed dimensions or readback.
pub fn kfac_autotune_step_via_into(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    blocks_in: &[f32],
    num_blocks: u32,
    n: u32,
    out: &mut Vec<f32>,
) -> Result<(), SemanticExecutionError> {
    let mut scratch = KfacAutotuneGpuScratch::default();
    kfac_autotune_step_via_with_scratch_into(
        dispatcher,
        policy,
        blocks_in,
        num_blocks,
        n,
        &mut scratch,
        out,
    )
}

/// GPU dispatch wrapper around [`kfac_autotune_step_program`] into
/// caller-owned dispatch and output storage.
///
/// # Errors
///
/// Propagates dispatch failures and rejects malformed dimensions or readback.
pub fn kfac_autotune_step_via_with_scratch_into(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    blocks_in: &[f32],
    num_blocks: u32,
    n: u32,
    scratch: &mut KfacAutotuneGpuScratch,
    out: &mut Vec<f32>,
) -> Result<(), SemanticExecutionError> {
    if num_blocks == 0 || n == 0 {
        return Err(SemanticExecutionError::InvalidRequest(format!(
        "Fix: kfac_autotune_step_via requires num_blocks > 0 and n > 0; got num_blocks={num_blocks}, n={n}."
    )));
    }
    let block_cells = n.checked_mul(n).ok_or_else(|| {
        SemanticExecutionError::InvalidRequest(format!(
            "Fix: kfac_autotune_step_via block size overflows n*n for n={n}."
        ))
    })?;
    let total_cells = num_blocks.checked_mul(block_cells).ok_or_else(|| {
    SemanticExecutionError::InvalidRequest(format!(
        "Fix: kfac_autotune_step_via total size overflows num_blocks*n*n for num_blocks={num_blocks}, n={n}."
    ))
})? as usize;
    if blocks_in.len() != total_cells {
        return Err(SemanticExecutionError::InvalidRequest(format!(
        "Fix: kfac_autotune_step_via expected blocks_in.len() == num_blocks*n*n == {total_cells}, got {}.",
        blocks_in.len()
    )));
    }

    let program = kfac_autotune_step_program("blocks_out", "blocks_in", "scratch", num_blocks, n);
    let byte_len = total_cells
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            SemanticExecutionError::InvalidRequest(format!(
                "Fix: kfac_autotune_step_via byte size overflows usize for {total_cells} cells."
            ))
        })?;
    ensure_input_slots(&mut scratch.inputs, 3);
    write_zero_bytes(&mut scratch.inputs[0], byte_len);
    write_f32_slice_le_bytes(&mut scratch.inputs[1], blocks_in);
    write_zero_bytes(&mut scratch.inputs[2], byte_len);
    let outputs = execute_single_program(
        dispatcher,
        crate::dispatch_buffers::HOST_WRAPPER_NODE,
        program,
        &scratch.inputs,
        policy,
    )
    .map(|output| output.outputs)?;
    if outputs.is_empty() {
        return Err(SemanticExecutionError::Backend(format!(
            "Fix: kfac_autotune_step_via expected at least the blocks_out output buffer, got {}.",
            outputs.len()
        )));
    }
    decode_f32_output_exact(&outputs[0], total_cells, "kfac_autotune_step_via", out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch_buffers::f32_slice_to_le_bytes;
    use crate::fixture_bytes::eval_bytes;
    use vyre_reference::composition_witness::kfac_block_inverse_witness as reference_kfac_block_inverse;

    struct KfacDispatcher;

    impl SemanticExecutor for KfacDispatcher {
        fn execute(
            &self,
            request: &vyre_megakernel::SemanticExecutionRequest<'_>,
        ) -> Result<vyre_megakernel::SemanticExecutionOutput, SemanticExecutionError> {
            let inputs = crate::test_parity_oracles::canonical_inputs(request)?;
            let ordered = (|| -> Result<Vec<Vec<u8>>, SemanticExecutionError> {
                assert_eq!(inputs.len(), 3);
                assert_eq!(inputs[0].len(), inputs[1].len());
                assert_eq!(inputs[2].len(), inputs[1].len());
                let blocks_in = crate::dispatch_buffers::read_f32s(&inputs[1]);
                let out = reference_kfac_block_inverse(&blocks_in, 1, 2);
                Ok(vec![f32_slice_to_le_bytes(&out)])
            })();
            let mut ordered = ordered?;
            let output_count = request.logical().graph().nodes()[0].outputs.len();
            if ordered.len() < output_count {
                ordered.resize(output_count, Vec::new());
            }
            crate::test_parity_oracles::semantic_output(request, ordered)
        }
    }

    #[test]
    fn test_kfac_program_shape() {
        let p = kfac_autotune_step_program("bo", "bi", "s", 10, 16);
        assert_eq!(p.buffers().len(), 3, "Expects exactly 3 buffers");
        assert!(p.buffers().iter().any(|b| b.name() == "bi"));
    }

    #[test]
    fn test_kfac_autotune_fisher_block() {
        // Non-trivial vyre IR shape: 2 blocks of size 2x2.
        // Block 1: Identity
        // Block 2: Diagonal [2, 4] -> inverse is [0.5, 0.25]
        let num_blocks = 2;
        let n = 2;
        let blocks_in = vec![
            1.0, 0.0, 0.0, 1.0, // block 0
            2.0, 0.0, 0.0, 4.0, // block 1
        ];

        let out = reference_kfac_block_inverse(&blocks_in, num_blocks, n);

        assert_eq!(out[0..4], vec![1.0, 0.0, 0.0, 1.0]);
        assert_eq!(out[4..8], vec![0.5, 0.0, 0.0, 0.25]);
    }

    #[test]
    fn test_kfac_autotune_dense_block() {
        // Dense block
        let num_blocks = 1;
        let n = 2;
        let blocks_in = vec![4.0, 3.0, 3.0, 2.0];
        // determinant = 4*2 - 3*3 = 8 - 9 = -1
        // inverse = [-2, 3; 3, -4]

        let out = reference_kfac_block_inverse(&blocks_in, num_blocks, n);

        assert_eq!(out, vec![-2.0, 3.0, 3.0, -4.0]);
    }

    #[test]
    fn test_multi_layer_kfac_composition() {
        let p1 = kfac_autotune_step_program("bo1", "bi1", "s1", 1, 4);
        let p2 = kfac_autotune_step_program("bo2", "bi2", "s2", 1, 4);
        let p3 = kfac_autotune_step_program("bo3", "bi3", "s3", 1, 4);

        let final_p =
            crate::test_parity_oracles::wrap_program_sequence(&[&p1, &p2, &p3], [256, 1, 1]);
        crate::solvers::test_helpers::assert_min_region_count(&final_p, 3);
    }

    #[test]
    fn test_end_to_end_kfac_parity() {
        let blocks_in = vec![2.0, 0.0, 0.0, 4.0];
        let p = kfac_autotune_step_program("bo", "bi", "s", 1, 2);

        let to_value = |data: &[f32]| vyre_primitives::wire::pack_f32_slice(data);

        let inputs = vec![
            to_value(&[0.0; 4]),
            to_value(&blocks_in),
            to_value(&[0.0; 4]),
        ];

        let results = eval_bytes("kfac_autotune_step", &p, inputs.clone());
        let actual_bytes = results[0].clone();
        let actual_out: Vec<f32> = actual_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(actual_out, vec![0.5, 0.0, 0.0, 0.25]);
    }

    #[test]
    fn kfac_autotune_step_via_dispatches_primitive() {
        let blocks_in = vec![2.0, 0.0, 0.0, 4.0];

        let out = kfac_autotune_step_via(
            &KfacDispatcher,
            &crate::test_parity_oracles::policy(),
            &blocks_in,
            1,
            2,
        )
        .unwrap();

        assert_eq!(out, vec![0.5, 0.0, 0.0, 0.25]);
    }

    #[test]
    fn kfac_autotune_step_via_into_reuses_output() {
        let blocks_in = vec![2.0, 0.0, 0.0, 4.0];
        let mut out = Vec::with_capacity(8);
        let ptr = out.as_ptr();

        kfac_autotune_step_via_into(
            &KfacDispatcher,
            &crate::test_parity_oracles::policy(),
            &blocks_in,
            1,
            2,
            &mut out,
        )
        .unwrap();

        assert_eq!(out.as_ptr(), ptr);
        assert_eq!(out, vec![0.5, 0.0, 0.0, 0.25]);
    }

    #[test]
    fn kfac_autotune_step_via_with_scratch_reuses_dispatch_and_output_storage() {
        let blocks_in = vec![2.0, 0.0, 0.0, 4.0];
        let mut scratch = KfacAutotuneGpuScratch::default();
        let mut out = Vec::with_capacity(4);

        kfac_autotune_step_via_with_scratch_into(
            &KfacDispatcher,
            &crate::test_parity_oracles::policy(),
            &blocks_in,
            1,
            2,
            &mut scratch,
            &mut out,
        )
        .unwrap();

        let input_capacities = scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>();
        let out_capacity = out.capacity();

        kfac_autotune_step_via_with_scratch_into(
            &KfacDispatcher,
            &crate::test_parity_oracles::policy(),
            &blocks_in,
            1,
            2,
            &mut scratch,
            &mut out,
        )
        .unwrap();

        crate::solvers::test_helpers::assert_scratch_capacities_preserved(
            &scratch.inputs,
            &input_capacities,
        );
        assert_eq!(out.capacity(), out_capacity);
        assert_eq!(out, vec![0.5, 0.0, 0.0, 0.25]);
    }

    #[test]
    fn kfac_autotune_step_via_rejects_bad_shape() {
        let err = kfac_autotune_step_via(
            &KfacDispatcher,
            &crate::test_parity_oracles::policy(),
            &[1.0, 0.0],
            1,
            2,
        )
        .unwrap_err();

        assert!(matches!(err, SemanticExecutionError::InvalidRequest(_)));
    }
}
