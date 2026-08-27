//! Scheduler/cache mask algebra via `vyre-primitives::bitset`.
//!
//! Fusion groups, dirty-region filters, resident-cache reuse masks, and
//! invalidation frontiers are all packed bitsets. This module centralizes the
//! common mask operations so self-substrate users consume the same primitive
//! programs that downstream users consume instead of re-implementing bit twiddles
//! in each optimizer pass.

use super::decode_first_output;
use crate::bitset::{
    and::bitset_and, clear_bit::bitset_clear_bit, contains::bitset_contains, equal::bitset_equal,
    not::bitset_not, or::bitset_or, set_bit::bitset_set_bit, subset_of::bitset_subset_of,
    test_bit::bitset_test_bit, xor::bitset_xor,
};
use crate::dispatch_buffers::{ensure_input_slots, write_u32_slice_le_bytes, write_zero_bytes};
use vyre_megakernel::{
    execute_single_program, SemanticExecutionError, SemanticExecutionPolicy, SemanticExecutor,
};

/// Caller-owned dispatch scratch for bitset mask algebra.
#[derive(Debug, Default)]
pub struct BitsetMaskAlgebraGpuScratch {
    inputs: Vec<Vec<u8>>,
}

/// Mask operation selector for two-input bitset algebra.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BitsetMaskBinaryOp {
    /// `lhs & rhs`.
    And,
    /// `lhs | rhs`.
    Or,
    /// `lhs ^ rhs`.
    Xor,
}

/// Apply one binary mask operation through the primitive GPU program.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when input lengths differ, the word count exceeds
/// the primitive index space, semantic execution fails, or readback is malformed.
pub fn mask_binary_via(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    op: BitsetMaskBinaryOp,
    lhs: &[u32],
    rhs: &[u32],
) -> Result<Vec<u32>, SemanticExecutionError> {
    let mut out = Vec::new();
    mask_binary_via_into(dispatcher, policy, op, lhs, rhs, &mut out)?;
    Ok(out)
}

/// Apply one binary mask operation into caller-owned output storage.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when validation, execution, or readback fails.
pub fn mask_binary_via_into(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    op: BitsetMaskBinaryOp,
    lhs: &[u32],
    rhs: &[u32],
    out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    let mut scratch = BitsetMaskAlgebraGpuScratch::default();
    mask_binary_via_with_scratch_into(dispatcher, policy, op, lhs, rhs, &mut scratch, out)
}

/// Apply one binary mask operation using caller-owned dispatch scratch.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when validation, execution, or readback fails.
pub fn mask_binary_via_with_scratch_into(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    op: BitsetMaskBinaryOp,
    lhs: &[u32],
    rhs: &[u32],
    scratch: &mut BitsetMaskAlgebraGpuScratch,
    out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    use crate::telemetry::{bitset_mask_algebra_calls, bump};
    bump(&bitset_mask_algebra_calls);

    if lhs.len() != rhs.len() {
        return Err(SemanticExecutionError::InvalidRequest(format!(
            "Fix: mask_binary_via requires lhs.len() == rhs.len(), got {} and {}.",
            lhs.len(),
            rhs.len()
        )));
    }
    if lhs.is_empty() {
        out.clear();
        return Ok(());
    }
    let words = checked_words(lhs.len(), "mask_binary_via")?;
    let program = match op {
        BitsetMaskBinaryOp::And => bitset_and("lhs", "rhs", "out", words),
        BitsetMaskBinaryOp::Or => bitset_or("lhs", "rhs", "out", words),
        BitsetMaskBinaryOp::Xor => bitset_xor("lhs", "rhs", "out", words),
    };
    ensure_input_slots(&mut scratch.inputs, 2);
    write_u32_slice_le_bytes(&mut scratch.inputs[0], lhs);
    write_u32_slice_le_bytes(&mut scratch.inputs[1], rhs);
    let outputs = execute_single_program(
        dispatcher,
        crate::dispatch_buffers::HOST_WRAPPER_NODE,
        program,
        &scratch.inputs,
        policy,
    )
    .map(|output| output.outputs)?;
    decode_first_output(&outputs, lhs.len(), "mask_binary_via", out)
}

/// Compute `lhs & rhs` through the bitset primitive.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when validation, execution, or readback fails.
pub fn mask_and_via(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    lhs: &[u32],
    rhs: &[u32],
) -> Result<Vec<u32>, SemanticExecutionError> {
    mask_binary_via(dispatcher, policy, BitsetMaskBinaryOp::And, lhs, rhs)
}

/// Compute `lhs | rhs` through the bitset primitive.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when validation, execution, or readback fails.
pub fn mask_or_via(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    lhs: &[u32],
    rhs: &[u32],
) -> Result<Vec<u32>, SemanticExecutionError> {
    mask_binary_via(dispatcher, policy, BitsetMaskBinaryOp::Or, lhs, rhs)
}

/// Compute `lhs ^ rhs` through the bitset primitive.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when validation, execution, or readback fails.
pub fn mask_xor_via(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    lhs: &[u32],
    rhs: &[u32],
) -> Result<Vec<u32>, SemanticExecutionError> {
    mask_binary_via(dispatcher, policy, BitsetMaskBinaryOp::Xor, lhs, rhs)
}

/// Compute `!input` through the bitset primitive.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when the word count exceeds the primitive index
/// space, semantic execution fails, or readback is malformed.
pub fn mask_not_via(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    input: &[u32],
) -> Result<Vec<u32>, SemanticExecutionError> {
    let mut scratch = BitsetMaskAlgebraGpuScratch::default();
    let mut out = Vec::new();
    mask_not_via_with_scratch_into(dispatcher, policy, input, &mut scratch, &mut out)?;
    Ok(out)
}

/// Compute `!input` through the bitset primitive using caller-owned scratch.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when validation, execution, or readback fails.
pub fn mask_not_via_with_scratch_into(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    input: &[u32],
    scratch: &mut BitsetMaskAlgebraGpuScratch,
    out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    use crate::telemetry::{bitset_mask_algebra_calls, bump};
    bump(&bitset_mask_algebra_calls);

    if input.is_empty() {
        out.clear();
        return Ok(());
    }
    let words = checked_words(input.len(), "mask_not_via")?;
    let program = bitset_not("input", "out", words);
    ensure_input_slots(&mut scratch.inputs, 1);
    write_u32_slice_le_bytes(&mut scratch.inputs[0], input);
    let outputs = execute_single_program(
        dispatcher,
        crate::dispatch_buffers::HOST_WRAPPER_NODE,
        program,
        &scratch.inputs,
        policy,
    )
    .map(|output| output.outputs)?;
    decode_first_output(&outputs, input.len(), "mask_not_via", out)
}

/// Test exact equality through the bitset primitive.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when validation, execution, or scalar readback fails.
pub fn mask_equal_via(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    lhs: &[u32],
    rhs: &[u32],
) -> Result<bool, SemanticExecutionError> {
    scalar_binary_predicate_via(dispatcher, policy, "mask_equal_via", lhs, rhs, bitset_equal)
}

/// Test subset relation through the bitset primitive.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when validation, execution, or scalar readback fails.
pub fn mask_subset_of_via(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    lhs: &[u32],
    rhs: &[u32],
) -> Result<bool, SemanticExecutionError> {
    scalar_binary_predicate_via(
        dispatcher,
        policy,
        "mask_subset_of_via",
        lhs,
        rhs,
        bitset_subset_of,
    )
}

/// Test whether a bit is present using the index-buffer `contains` primitive.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when the word count exceeds primitive limits,
/// semantic execution fails, or scalar readback is malformed.
pub fn mask_contains_via(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    input: &[u32],
    bit_idx: u32,
) -> Result<bool, SemanticExecutionError> {
    use crate::telemetry::{bitset_mask_algebra_calls, bump};
    bump(&bitset_mask_algebra_calls);

    let words = checked_words(input.len(), "mask_contains_via")?;
    let program = bitset_contains("input", "index", "out", words);
    let mut scratch = BitsetMaskAlgebraGpuScratch::default();
    ensure_input_slots(&mut scratch.inputs, 3);
    write_u32_slice_le_bytes(&mut scratch.inputs[0], input);
    write_u32_slice_le_bytes(&mut scratch.inputs[1], &[bit_idx]);
    write_zero_bytes(&mut scratch.inputs[2], std::mem::size_of::<u32>());
    let outputs = execute_single_program(
        dispatcher,
        crate::dispatch_buffers::HOST_WRAPPER_NODE,
        program,
        &scratch.inputs,
        policy,
    )
    .map(|output| output.outputs)?;
    decode_scalar_bool(&outputs, "mask_contains_via")
}

/// Test a compile-time bit index using the scalar `test_bit` primitive.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when semantic execution or scalar readback fails.
pub fn mask_test_bit_via(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    input: &[u32],
    bit_idx: u32,
) -> Result<bool, SemanticExecutionError> {
    use crate::telemetry::{bitset_mask_algebra_calls, bump};
    bump(&bitset_mask_algebra_calls);

    if (bit_idx / 32) as usize >= input.len() {
        return Ok(false);
    }
    let program = bitset_test_bit("input", bit_idx, "out", input.len() as u32);
    let mut scratch = BitsetMaskAlgebraGpuScratch::default();
    ensure_input_slots(&mut scratch.inputs, 2);
    write_u32_slice_le_bytes(&mut scratch.inputs[0], input);
    write_zero_bytes(&mut scratch.inputs[1], std::mem::size_of::<u32>());
    let outputs = execute_single_program(
        dispatcher,
        crate::dispatch_buffers::HOST_WRAPPER_NODE,
        program,
        &scratch.inputs,
        policy,
    )
    .map(|output| output.outputs)?;
    decode_scalar_bool(&outputs, "mask_test_bit_via")
}

/// Set one bit in a cache/frontier mask through the bitset primitive.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when the word count exceeds primitive limits,
/// semantic execution fails, or readback is malformed.
pub fn mask_set_bit_via(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    target: &[u32],
    bit_idx: u32,
) -> Result<Vec<u32>, SemanticExecutionError> {
    scalar_mutate_bit_via(
        dispatcher,
        policy,
        "mask_set_bit_via",
        target,
        bit_idx,
        bitset_set_bit,
    )
}

/// Clear one bit in a cache/frontier mask through the bitset primitive.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when the word count exceeds primitive limits,
/// semantic execution fails, or readback is malformed.
pub fn mask_clear_bit_via(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    target: &[u32],
    bit_idx: u32,
) -> Result<Vec<u32>, SemanticExecutionError> {
    scalar_mutate_bit_via(
        dispatcher,
        policy,
        "mask_clear_bit_via",
        target,
        bit_idx,
        bitset_clear_bit,
    )
}

fn scalar_binary_predicate_via(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    context: &'static str,
    lhs: &[u32],
    rhs: &[u32],
    build: fn(&str, &str, &str, u32) -> vyre_foundation::ir::Program,
) -> Result<bool, SemanticExecutionError> {
    use crate::telemetry::{bitset_mask_algebra_calls, bump};
    bump(&bitset_mask_algebra_calls);

    if lhs.len() != rhs.len() {
        return Err(SemanticExecutionError::InvalidRequest(format!(
            "Fix: {context} requires lhs.len() == rhs.len(), got {} and {}.",
            lhs.len(),
            rhs.len()
        )));
    }
    let words = checked_words(lhs.len(), context)?;
    let program = build("lhs", "rhs", "out", words);
    let mut scratch = BitsetMaskAlgebraGpuScratch::default();
    ensure_input_slots(&mut scratch.inputs, 3);
    write_u32_slice_le_bytes(&mut scratch.inputs[0], lhs);
    write_u32_slice_le_bytes(&mut scratch.inputs[1], rhs);
    write_zero_bytes(&mut scratch.inputs[2], std::mem::size_of::<u32>());
    let outputs = execute_single_program(
        dispatcher,
        crate::dispatch_buffers::HOST_WRAPPER_NODE,
        program,
        &scratch.inputs,
        policy,
    )
    .map(|output| output.outputs)?;
    decode_scalar_bool(&outputs, context)
}

fn scalar_mutate_bit_via(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    context: &'static str,
    target: &[u32],
    bit_idx: u32,
    build: fn(&str, u32, u32) -> vyre_foundation::ir::Program,
) -> Result<Vec<u32>, SemanticExecutionError> {
    use crate::telemetry::{bitset_mask_algebra_calls, bump};
    bump(&bitset_mask_algebra_calls);

    if (bit_idx / 32) as usize >= target.len() {
        return Ok(target.to_vec());
    }
    let words = checked_words(target.len(), context)?;
    let program = build("target", bit_idx, words);
    let mut scratch = BitsetMaskAlgebraGpuScratch::default();
    ensure_input_slots(&mut scratch.inputs, 1);
    write_u32_slice_le_bytes(&mut scratch.inputs[0], target);
    let outputs = execute_single_program(
        dispatcher,
        crate::dispatch_buffers::HOST_WRAPPER_NODE,
        program,
        &scratch.inputs,
        policy,
    )
    .map(|output| output.outputs)?;
    let mut out = Vec::new();
    decode_first_output(&outputs, target.len(), context, &mut out)?;
    Ok(out)
}

fn checked_words(len: usize, context: &'static str) -> Result<u32, SemanticExecutionError> {
    u32::try_from(len).map_err(|_| {
        SemanticExecutionError::InvalidRequest(format!(
            "Fix: {context} received {len} words, which exceeds the u32 GPU index space."
        ))
    })
}

fn decode_scalar_bool(
    outputs: &[Vec<u8>],
    context: &'static str,
) -> Result<bool, SemanticExecutionError> {
    let mut out = Vec::new();
    decode_first_output(outputs, 1, context, &mut out)?;
    Ok(out[0] != 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch_buffers::u32_slice_to_le_bytes;

    use vyre_reference::composition_witness::{
        bitset_and_witness as reference_mask_and,
        bitset_clear_bit_witness as reference_mask_clear_bit,
        bitset_contains_witness as reference_mask_contains,
        bitset_equal_witness as reference_mask_equal, bitset_not_witness as reference_mask_not,
        bitset_or_witness as reference_mask_or, bitset_set_bit_witness as reference_mask_set_bit,
        bitset_subset_of_witness as reference_mask_subset_of,
        bitset_xor_witness as reference_mask_xor,
    };

    fn reference_mask_test_bit(input: &[u32], bit_idx: u32) -> bool {
        reference_mask_contains(input, bit_idx)
    }

    struct MaskDispatcher;

    impl SemanticExecutor for MaskDispatcher {
        fn execute(
            &self,
            request: &vyre_megakernel::SemanticExecutionRequest<'_>,
        ) -> Result<vyre_megakernel::SemanticExecutionOutput, SemanticExecutionError> {
            let program = &request.logical().graph().nodes()[0].program;
            let inputs = crate::test_parity_oracles::canonical_inputs(request)?;
            let compute_ordered = || -> Result<Vec<Vec<u8>>, SemanticExecutionError> {
                let op_id = program
                    .entry
                    .iter()
                    .find_map(|node| match node {
                        vyre_foundation::ir::Node::Region { generator, .. } => {
                            Some(generator.as_str())
                        }
                        _ => None,
                    })
                    .expect("Fix: primitive program should contain region generator");
                match op_id {
                    crate::bitset::and::OP_ID => binary(&inputs, |a, b| a & b),
                    crate::bitset::or::OP_ID => binary(&inputs, |a, b| a | b),
                    crate::bitset::xor::OP_ID => binary(&inputs, |a, b| a ^ b),
                    crate::bitset::not::OP_ID => {
                        let input = crate::dispatch_buffers::read_u32s(&inputs[0]);
                        Ok(vec![u32_slice_to_le_bytes(
                            &input.iter().map(|word| !word).collect::<Vec<_>>(),
                        )])
                    }
                    crate::bitset::equal::OP_ID => {
                        let lhs = crate::dispatch_buffers::read_u32s(&inputs[0]);
                        let rhs = crate::dispatch_buffers::read_u32s(&inputs[1]);
                        Ok(vec![u32_slice_to_le_bytes(&[u32::from(lhs == rhs)])])
                    }
                    crate::bitset::subset_of::OP_ID => {
                        let lhs = crate::dispatch_buffers::read_u32s(&inputs[0]);
                        let rhs = crate::dispatch_buffers::read_u32s(&inputs[1]);
                        let ok = lhs.iter().zip(rhs.iter()).all(|(a, b)| (a & !b) == 0);
                        Ok(vec![u32_slice_to_le_bytes(&[u32::from(ok)])])
                    }
                    crate::bitset::contains::OP_ID => {
                        let input = crate::dispatch_buffers::read_u32s(&inputs[0]);
                        let index = crate::dispatch_buffers::read_u32s(&inputs[1])[0];
                        Ok(vec![u32_slice_to_le_bytes(&[u32::from(
                            reference_mask_contains(&input, index),
                        )])])
                    }
                    crate::bitset::test_bit::OP_ID => Ok(vec![u32_slice_to_le_bytes(&[1])]),
                    crate::bitset::set_bit::OP_ID => {
                        let target = crate::dispatch_buffers::read_u32s(&inputs[0]);
                        let target = reference_mask_set_bit(&target, 1);
                        Ok(vec![u32_slice_to_le_bytes(&target)])
                    }
                    crate::bitset::clear_bit::OP_ID => {
                        let target = crate::dispatch_buffers::read_u32s(&inputs[0]);
                        let target = reference_mask_clear_bit(&target, 1);
                        Ok(vec![u32_slice_to_le_bytes(&target)])
                    }
                    other => panic!("unexpected primitive op id {other}"),
                }
            };
            let ordered = compute_ordered()?;
            crate::test_parity_oracles::semantic_output(request, ordered)
        }
    }

    fn binary(
        inputs: &[Vec<u8>],
        op: impl Fn(u32, u32) -> u32,
    ) -> Result<Vec<Vec<u8>>, SemanticExecutionError> {
        let lhs = crate::dispatch_buffers::read_u32s(&inputs[0]);
        let rhs = crate::dispatch_buffers::read_u32s(&inputs[1]);
        let out = lhs
            .iter()
            .zip(rhs.iter())
            .map(|(a, b)| op(*a, *b))
            .collect::<Vec<_>>();
        Ok(vec![u32_slice_to_le_bytes(&out)])
    }

    #[test]
    fn reference_mask_algebra_matches_primitives_exactly() {
        let lhs = [0xF0F0u32, 0xAAAA_AAAA];
        let rhs = [0x0FF0u32, 0xFFFF_0000];

        assert_eq!(reference_mask_and(&lhs, &rhs), vec![0x00F0, 0xAAAA_0000]);
        assert_eq!(reference_mask_or(&lhs, &rhs), vec![0xFFF0, 0xFFFF_AAAA]);
        assert_eq!(reference_mask_xor(&lhs, &rhs), vec![0xFF00, 0x5555_AAAA]);
        assert_eq!(reference_mask_not(&lhs), vec![!0xF0F0u32, !0xAAAA_AAAA]);
        assert!(reference_mask_equal(&lhs, &lhs));
        assert!(!reference_mask_equal(&lhs, &rhs));
        assert!(reference_mask_subset_of(&[0b0011], &[0b1111]));
        assert!(reference_mask_contains(&[0b1010], 1));
        assert!(reference_mask_test_bit(&[0b1010], 1));
        assert_eq!(reference_mask_set_bit(&[0], 1), vec![0b10]);
        assert_eq!(reference_mask_clear_bit(&[0b11], 1), vec![0b01]);
    }

    #[test]
    fn binary_dispatch_uses_primitive_programs() {
        let lhs = [0xF0F0u32, 0xAAAA_AAAA];
        let rhs = [0x0FF0u32, 0xFFFF_0000];

        assert_eq!(
            mask_and_via(
                &MaskDispatcher,
                &crate::test_parity_oracles::policy(),
                &lhs,
                &rhs
            )
            .unwrap(),
            reference_mask_and(&lhs, &rhs)
        );
        assert_eq!(
            mask_or_via(
                &MaskDispatcher,
                &crate::test_parity_oracles::policy(),
                &lhs,
                &rhs
            )
            .unwrap(),
            reference_mask_or(&lhs, &rhs)
        );
        assert_eq!(
            mask_xor_via(
                &MaskDispatcher,
                &crate::test_parity_oracles::policy(),
                &lhs,
                &rhs
            )
            .unwrap(),
            reference_mask_xor(&lhs, &rhs)
        );
    }

    #[test]
    fn unary_and_scalar_dispatch_use_primitive_programs() {
        assert_eq!(
            mask_not_via(
                &MaskDispatcher,
                &crate::test_parity_oracles::policy(),
                &[0x0F0F_F0F0],
            )
            .unwrap(),
            reference_mask_not(&[0x0F0F_F0F0])
        );
        assert!(mask_equal_via(
            &MaskDispatcher,
            &crate::test_parity_oracles::policy(),
            &[1, 2],
            &[1, 2],
        )
        .unwrap());
        assert!(mask_subset_of_via(
            &MaskDispatcher,
            &crate::test_parity_oracles::policy(),
            &[0b0011],
            &[0b1111],
        )
        .unwrap());
        assert!(mask_contains_via(
            &MaskDispatcher,
            &crate::test_parity_oracles::policy(),
            &[0b1010],
            1,
        )
        .unwrap());
        assert!(mask_test_bit_via(
            &MaskDispatcher,
            &crate::test_parity_oracles::policy(),
            &[0b1010],
            1,
        )
        .unwrap());
        assert_eq!(
            mask_set_bit_via(
                &MaskDispatcher,
                &crate::test_parity_oracles::policy(),
                &[0],
                1,
            )
            .unwrap(),
            vec![0b10]
        );
        assert_eq!(
            mask_clear_bit_via(
                &MaskDispatcher,
                &crate::test_parity_oracles::policy(),
                &[0b11],
                1,
            )
            .unwrap(),
            vec![0b01]
        );
    }

    #[test]
    fn scratch_binary_path_reuses_output_capacity() {
        let mut scratch = BitsetMaskAlgebraGpuScratch::default();
        let mut out = Vec::with_capacity(4);
        mask_binary_via_with_scratch_into(
            &MaskDispatcher,
            &crate::test_parity_oracles::policy(),
            BitsetMaskBinaryOp::And,
            &[0xFFFF],
            &[0x00FF],
            &mut scratch,
            &mut out,
        )
        .unwrap();
        let out_capacity = out.capacity();
        let input_capacities = scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>();

        mask_binary_via_with_scratch_into(
            &MaskDispatcher,
            &crate::test_parity_oracles::policy(),
            BitsetMaskBinaryOp::Or,
            &[0xF000],
            &[0x000F],
            &mut scratch,
            &mut out,
        )
        .unwrap();

        assert_eq!(out.capacity(), out_capacity);
        assert_eq!(
            scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>(),
            input_capacities
        );
        assert_eq!(out, vec![0xF00F]);
    }

    #[test]
    fn length_mismatch_is_actionable() {
        let err = mask_and_via(
            &MaskDispatcher,
            &crate::test_parity_oracles::policy(),
            &[1],
            &[1, 2],
        )
        .unwrap_err();
        assert!(err.to_string().contains("Fix: mask_binary_via requires"));
    }
}
