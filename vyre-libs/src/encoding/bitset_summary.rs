//! Bitset summary substrate consumer.
//!
//! Wires `crate::bitset::popcount` and several companion bitset
//! operations into the dispatch path so the optimizer / cache invalidator can
//! summarize how saturated their reachability / alias / dirty-set bitsets are
//! without each pass re-implementing popcount inline.

use crate::dispatch_buffers::{
    decode_u32_output_exact, ensure_input_slots, write_u32_slice_le_bytes, write_zero_bytes,
};
use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_megakernel::{
    execute_single_program, SemanticExecutionError, SemanticExecutionPolicy, SemanticExecutor,
};

/// Maximum word count where the sum of bit counts is guaranteed not to wrap a u32 reduction accumulator.
const MAX_TOTAL_SET_BITS_WORDS: usize = (u32::MAX / 32) as usize;

fn validate_total_set_bits_len(len: usize) -> Result<u32, SemanticExecutionError> {
    if len > MAX_TOTAL_SET_BITS_WORDS {
        return Err(SemanticExecutionError::InvalidRequest(format!(
        "Fix: total_set_bits_via input has {len} words, which exceeds the max capacity of {MAX_TOTAL_SET_BITS_WORDS} words (u32::MAX bits) supported by single-pass u32 GPU popcount reduction; shard the bitset before summarizing."
    )));
    }
    u32::try_from(len).map_err(|_| {
        SemanticExecutionError::InvalidRequest(format!(
            "Fix: total_set_bits_via input has {len} words, which exceeds the u32 GPU index space."
        ))
    })
}

/// Canonical op id for bitset saturation ratio.
const SATURATION_RATIO_OP_ID: &str = "vyre-libs::encoding::bitset_saturation_ratio";

/// Build a Program that computes the saturation ratio of a packed bitset on GPU.
#[must_use]
fn bitset_saturation_ratio(input: &str, output: &str, words: u32) -> Program {
    let tile = 256u32;
    let chunks = words.div_ceil(tile);
    let total_bits = (words as f32) * 32.0;
    let phase = crate::builder::reduction::ReductionPhase {
        accumulate: crate::builder::strided_accumulate_child(
            SATURATION_RATIO_OP_ID,
            tile,
            chunks,
            words,
            "sat_acc",
            Expr::f32(0.0),
            "sat_scratch",
            |idx, acc| {
                Expr::add(
                    acc,
                    Expr::cast(DataType::F32, Expr::popcount(Expr::load(input, idx))),
                )
            },
        ),
        reductions: vec![crate::reduce::workgroup_tree::sum_f32_child(
            SATURATION_RATIO_OP_ID,
            tile,
            "sat_scratch",
            crate::reduce::workgroup_tree::WorkgroupReductionScope::FirstWorkgroup,
        )],
        publish: vec![Node::Store {
            buffer: output.into(),
            index: Expr::u32(0),
            value: Expr::div(
                Expr::load("sat_scratch", Expr::u32(0)),
                Expr::f32(total_bits),
            ),
        }],
    };
    crate::builder::reduction::ReductionComposer::new(
        SATURATION_RATIO_OP_ID,
        vec![
            BufferDecl::storage(input, 0, BufferAccess::ReadOnly, DataType::U32).with_count(words),
            BufferDecl::workgroup("sat_scratch", tile, DataType::F32),
            BufferDecl::output(output, 1, DataType::F32).with_count(1),
        ],
        [tile, 1, 1],
    )
    .with_phase(phase)
    .build()
}
/// Caller-owned GPU dispatch scratch for bitset-summary kernels.
#[derive(Debug, Default)]
pub struct BitsetSummaryGpuScratch {
    inputs: Vec<Vec<u8>>,
    decoded_u32: Vec<u32>,
}

/// GPU dispatch wrapper around the primitive per-word popcount program.
///
/// # Errors
///
/// Propagates dispatcher errors or malformed readback.
pub fn per_word_popcount_via(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    input: &[u32],
) -> Result<Vec<u32>, SemanticExecutionError> {
    let mut out = Vec::new();
    per_word_popcount_via_into(dispatcher, policy, input, &mut out)?;
    Ok(out)
}

/// GPU dispatch wrapper around the primitive per-word popcount program into
/// caller-owned output storage.
///
/// # Errors
///
/// Propagates dispatcher errors or malformed readback.
pub fn per_word_popcount_via_into(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    input: &[u32],
    out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    let mut scratch = BitsetSummaryGpuScratch::default();
    per_word_popcount_via_with_scratch_into(dispatcher, policy, input, &mut scratch, out)
}

/// GPU dispatch wrapper around the primitive per-word popcount program into
/// caller-owned dispatch and output storage.
///
/// # Errors
///
/// Propagates dispatcher errors or malformed readback.
pub fn per_word_popcount_via_with_scratch_into(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    input: &[u32],
    scratch: &mut BitsetSummaryGpuScratch,
    out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    if input.is_empty() {
        out.clear();
        return Ok(());
    }
    let word_count = u32::try_from(input.len()).map_err(|_| {
        SemanticExecutionError::InvalidRequest(format!(
            "Fix: per_word_popcount_via input has {} words, which exceeds the u32 GPU index space.",
            input.len()
        ))
    })?;
    let program = crate::bitset::popcount::bitset_popcount("input", "count_words", word_count);
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
    let [out_buf, ..] = match outputs.as_slice() {
        [out_buf, ..] => [out_buf],
        [] => {
            return Err(SemanticExecutionError::Backend(format!(
                "Fix: per_word_popcount_via expected at least one output buffer, got {}.",
                outputs.len()
            )));
        }
    };
    decode_u32_output_exact(out_buf, input.len(), "per_word_popcount_via", out)
}

/// GPU-backed total set-bit count.
///
/// # Errors
///
/// Propagates popcount reduction dispatch errors.
pub fn total_set_bits_via(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    input: &[u32],
) -> Result<u64, SemanticExecutionError> {
    let mut scratch = BitsetSummaryGpuScratch::default();
    total_set_bits_via_with_scratch_into(dispatcher, policy, input, &mut scratch)
}

/// GPU-backed total set-bit count using caller-owned scratch.
///
/// # Errors
///
/// Propagates popcount reduction dispatch errors.
pub fn total_set_bits_via_with_scratch_into(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    input: &[u32],
    scratch: &mut BitsetSummaryGpuScratch,
) -> Result<u64, SemanticExecutionError> {
    if input.is_empty() {
        return Ok(0);
    }
    let word_count = validate_total_set_bits_len(input.len())?;
    let program = crate::reduce::count::reduce_count("input", "out", word_count);
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
    let [out_buf, ..] = match outputs.as_slice() {
        [out_buf, ..] => [out_buf],
        [] => {
            return Err(SemanticExecutionError::Backend(format!(
                "Fix: total_set_bits_via expected at least one output buffer, got {}.",
                outputs.len()
            )));
        }
    };
    decode_u32_output_exact(out_buf, 1, "total_set_bits_via", &mut scratch.decoded_u32)?;
    Ok(u64::from(scratch.decoded_u32[0]))
}

/// GPU-backed saturation ratio.
///
/// # Errors
///
/// Propagates popcount dispatch errors.
pub fn saturation_ratio_via(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    input: &[u32],
) -> Result<f64, SemanticExecutionError> {
    let mut scratch = BitsetSummaryGpuScratch::default();
    saturation_ratio_via_with_scratch_into(dispatcher, policy, input, &mut scratch)
}

/// GPU-backed saturation ratio using caller-owned scratch.
///
/// # Errors
///
/// Propagates popcount reduction dispatch errors.
pub fn saturation_ratio_via_with_scratch_into(
    dispatcher: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    input: &[u32],
    scratch: &mut BitsetSummaryGpuScratch,
) -> Result<f64, SemanticExecutionError> {
    if input.is_empty() {
        return Ok(0.0);
    }
    let word_count = validate_total_set_bits_len(input.len())?;
    let program = bitset_saturation_ratio("input", "out", word_count);
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
    let [out_buf, ..] = match outputs.as_slice() {
        [out_buf, ..] => [out_buf],
        [] => {
            return Err(SemanticExecutionError::Backend(format!(
                "Fix: saturation_ratio_via expected at least one output buffer, got {}.",
                outputs.len()
            )));
        }
    };
    let [b0, b1, b2, b3, ..] = match out_buf.as_slice() {
        [b0, b1, b2, b3, ..] => [*b0, *b1, *b2, *b3],
        _ => {
            return Err(SemanticExecutionError::Backend(format!(
                "Fix: saturation_ratio_via expected at least 4 output bytes for f32, got {}.",
                out_buf.len()
            )));
        }
    };
    let val = f32::from_le_bytes([b0, b1, b2, b3]);
    Ok(f64::from(val))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch_buffers::u32_slice_to_le_bytes;

    use vyre_reference::composition_witness::{
        bitset_popcount_witness as per_word_popcount,
        bitset_popcount_witness_into as per_word_popcount_into,
    };

    fn total_set_bits(input: &[u32]) -> u64 {
        per_word_popcount(input).iter().map(|&w| u64::from(w)).sum()
    }

    fn saturation_ratio(input: &[u32]) -> f64 {
        if input.is_empty() {
            return 0.0;
        }
        let capacity_bits = (input.len() as u64) * 32;
        let set = total_set_bits(input);
        (set as f64) / (capacity_bits as f64)
    }

    struct PopcountDispatcher;

    impl SemanticExecutor for PopcountDispatcher {
        fn execute(
            &self,
            request: &vyre_megakernel::SemanticExecutionRequest<'_>,
        ) -> Result<vyre_megakernel::SemanticExecutionOutput, SemanticExecutionError> {
            let program = &request.logical().graph().nodes()[0].program;
            let inputs = crate::test_parity_oracles::canonical_inputs(request)?;
            let ordered = (|| -> Result<Vec<Vec<u8>>, SemanticExecutionError> {
                let op_id = crate::test_parity_oracles::region_operation_id(program)?;
                if op_id == crate::reduce::count::OP_ID {
                    assert_eq!(inputs.len(), 2);
                    let input = crate::dispatch_buffers::read_u32s(&inputs[0]);
                    assert_eq!(inputs[1].len(), std::mem::size_of::<u32>());
                    let total: u32 = input.iter().map(|word| word.count_ones()).sum();
                    return Ok(vec![u32_slice_to_le_bytes(&[total])]);
                }
                assert_eq!(inputs.len(), 1);
                let input = crate::dispatch_buffers::read_u32s(&inputs[0]);
                if op_id == SATURATION_RATIO_OP_ID {
                    let total: u32 = input.iter().map(|word| word.count_ones()).sum();
                    let capacity = (input.len() * 32) as f32;
                    let ratio = (total as f32) / capacity;
                    return Ok(vec![vyre_primitives::wire::pack_f32_slice(&[ratio])]);
                }

                let out: Vec<u32> = input.iter().map(|word| word.count_ones()).collect();
                Ok(vec![u32_slice_to_le_bytes(&out)])
            })()?;
            crate::test_parity_oracles::semantic_output(request, ordered)
        }
    }

    #[test]
    fn empty_input_yields_empty_summary() {
        let v = per_word_popcount(&[]);
        assert!(v.is_empty());
        assert_eq!(total_set_bits(&[]), 0);
        assert_eq!(saturation_ratio(&[]), 0.0);
    }

    #[test]
    fn full_word_is_thirty_two_bits() {
        let v = per_word_popcount(&[0xFFFF_FFFFu32]);
        assert_eq!(v, vec![32u32]);
        assert_eq!(total_set_bits(&[0xFFFF_FFFF]), 32);
        assert!((saturation_ratio(&[0xFFFF_FFFF]) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn mixed_words_count_correctly() {
        // 0b1111 = 4 bits, 0b101 = 2 bits.
        let v = per_word_popcount(&[0b1111u32, 0b101]);
        assert_eq!(v, vec![4, 2]);
        assert_eq!(total_set_bits(&[0b1111, 0b101]), 6);
    }

    #[test]
    fn popcount_into_reuses_capacity() {
        let mut out = Vec::with_capacity(8);
        per_word_popcount_into(&[0b1111u32, 0xFFFF_FFFF], &mut out);
        let capacity = out.capacity();
        assert_eq!(out, vec![4, 32]);

        per_word_popcount_into(&[0b1010u32], &mut out);
        assert_eq!(out.capacity(), capacity);
        assert_eq!(out, vec![2]);
    }

    /// Closure-bar: substrate output equals primitive output exactly.
    #[test]
    fn matches_primitive_directly() {
        let input = vec![0u32, 1, 0xFFFF_FFFF, 0xAAAA_AAAA, 0x12345678];
        assert_eq!(per_word_popcount(&input), vec![0, 1, 32, 16, 13]);
    }

    /// Adversarial: half-saturated bitset yields ratio 0.5.
    #[test]
    fn half_saturation_ratio() {
        // 0xAAAA_AAAA has 16 bits set out of 32.
        let r = saturation_ratio(&[0xAAAA_AAAAu32]);
        assert!((r - 0.5).abs() < 1e-9, "expected 0.5, got {r}");
    }

    /// Adversarial: a bitset that's 32 entries wide but only one bit
    /// set has saturation ≈ 1/(32*32).
    #[test]
    fn single_bit_in_large_bitset() {
        let mut input = vec![0u32; 32];
        input[5] = 1;
        let r = saturation_ratio(&input);
        let expected = 1.0 / 1024.0;
        assert!((r - expected).abs() < 1e-9);
    }

    /// Idempotence: per_word_popcount on the same input is
    /// deterministic.
    #[test]
    fn deterministic_summary() {
        let input = vec![0xCAFE_BABEu32, 0x1234_5678];
        let a = per_word_popcount(&input);
        let b = per_word_popcount(&input);
        assert_eq!(a, b);
    }

    #[test]
    fn per_word_popcount_via_dispatches_primitive() {
        let input = vec![0u32, 1, 0xFFFF_FFFF, 0xAAAA_AAAA];
        let out = per_word_popcount_via(
            &PopcountDispatcher,
            &crate::test_parity_oracles::policy(),
            &input,
        )
        .unwrap();
        assert_eq!(out, vec![0, 1, 32, 16]);
    }

    #[test]
    fn per_word_popcount_via_into_reuses_output() {
        let mut out = Vec::with_capacity(8);
        let ptr = out.as_ptr();
        per_word_popcount_via_into(
            &PopcountDispatcher,
            &crate::test_parity_oracles::policy(),
            &[0b1011],
            &mut out,
        )
        .unwrap();
        assert_eq!(out, vec![3]);
        assert_eq!(out.as_ptr(), ptr);
    }

    #[test]
    fn per_word_popcount_via_with_scratch_reuses_dispatch_and_output_storage() {
        let mut scratch = BitsetSummaryGpuScratch::default();
        let mut out = Vec::with_capacity(4);

        per_word_popcount_via_with_scratch_into(
            &PopcountDispatcher,
            &crate::test_parity_oracles::policy(),
            &[0b1011, 0xFFFF_FFFF],
            &mut scratch,
            &mut out,
        )
        .unwrap();

        let input_capacities = scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>();
        let out_capacity = out.capacity();

        per_word_popcount_via_with_scratch_into(
            &PopcountDispatcher,
            &crate::test_parity_oracles::policy(),
            &[0b0101, 0xAAAA_AAAA],
            &mut scratch,
            &mut out,
        )
        .unwrap();

        assert_eq!(
            scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>(),
            input_capacities
        );
        assert_eq!(out.capacity(), out_capacity);
        assert_eq!(out, vec![2, 16]);
    }

    #[test]
    fn total_set_bits_via_with_scratch_reuses_dispatch_storage() {
        let mut scratch = BitsetSummaryGpuScratch::default();
        let res1 = total_set_bits_via_with_scratch_into(
            &PopcountDispatcher,
            &crate::test_parity_oracles::policy(),
            &[0b1011, 0xFFFF_FFFF],
            &mut scratch,
        )
        .unwrap();
        assert_eq!(res1, 35);

        let input_capacities = scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>();
        let res2 = total_set_bits_via_with_scratch_into(
            &PopcountDispatcher,
            &crate::test_parity_oracles::policy(),
            &[0b0101, 0xAAAA_AAAA],
            &mut scratch,
        )
        .unwrap();
        assert_eq!(res2, 18);
        assert_eq!(
            scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>(),
            input_capacities
        );
    }

    #[test]
    fn validate_total_set_bits_len_boundary_checks() {
        assert_eq!(
            validate_total_set_bits_len(MAX_TOTAL_SET_BITS_WORDS).unwrap(),
            MAX_TOTAL_SET_BITS_WORDS as u32
        );
        let err = validate_total_set_bits_len(MAX_TOTAL_SET_BITS_WORDS + 1)
            .expect_err("word count exceeding u32::MAX/32 must be rejected with BadInputs");
        assert!(matches!(err, SemanticExecutionError::InvalidRequest(_)));
        assert!(err.to_string().contains("shard the bitset"));
    }

    #[test]
    fn total_and_ratio_via_match_host_contract() {
        let input = vec![0xFFFF_FFFFu32, 0];
        assert_eq!(
            total_set_bits_via(
                &PopcountDispatcher,
                &crate::test_parity_oracles::policy(),
                &input,
            )
            .unwrap(),
            32
        );
        assert!(
            (saturation_ratio_via(
                &PopcountDispatcher,
                &crate::test_parity_oracles::policy(),
                &input,
            )
            .unwrap()
                - 0.5)
                .abs()
                < 1e-6
        );
    }
}
