//! GPU reduction metrics for self-substrate scheduling and telemetry.
//!
//! Optimizer passes repeatedly need scalar summaries: total active work,
//! maximum queue depth, minimum remaining budget, all/any convergence flags,
//! per-segment pressure, and occupancy histograms. This module routes those
//! summaries through `vyre-primitives::reduce` programs instead of open-coding
//! host loops in each pass.

use crate::dispatch_buffers::{ensure_input_slots, write_u32_slice_le_bytes, write_zero_bytes};
use crate::reduce::{
    all::reduce_all, any::reduce_any, count_non_zero::reduce_count_non_zero,
    histogram::histogram_atomic_scatter, max::reduce_max, min::reduce_min,
    segment_reduce::segment_reduce_sum, sum::reduce_sum,
};
use vyre_megakernel::{
    execute_single_program, SemanticExecutionError, SemanticExecutionPolicy, SemanticExecutor,
};

/// Caller-owned scratch for reduction metric dispatches.
#[derive(Debug, Default)]
pub struct ReductionMetricsGpuScratch {
    inputs: Vec<Vec<u8>>,
}

/// Scalar reduction selector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReductionMetric {
    /// Wrapping unsigned sum.
    Sum,
    /// Unsigned maximum.
    Max,
    /// Unsigned minimum.
    Min,
    /// Count non-zero lanes.
    CountNonZero,
    /// Any non-zero lane.
    Any,
    /// Every lane non-zero.
    All,
}

/// Dispatch one scalar reduction metric over a u32 value set.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when input length exceeds the primitive
/// index space, execution fails, or scalar readback is malformed.
pub fn reduce_metric_via(
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    metric: ReductionMetric,
    values: &[u32],
) -> Result<u32, SemanticExecutionError> {
    let mut scratch = ReductionMetricsGpuScratch::default();
    reduce_metric_via_with_scratch(executor, policy, metric, values, &mut scratch)
}

/// Dispatch one scalar reduction metric using caller-owned scratch.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when validation, execution, or readback fails.
pub fn reduce_metric_via_with_scratch(
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    metric: ReductionMetric,
    values: &[u32],
    scratch: &mut ReductionMetricsGpuScratch,
) -> Result<u32, SemanticExecutionError> {
    bump_reduction_metrics_calls();

    let count = checked_len(values.len(), "reduce_metric_via")?;
    let program = match metric {
        ReductionMetric::Sum => reduce_sum("values", "out", count),
        ReductionMetric::Max => reduce_max("values", "out", count),
        ReductionMetric::Min => reduce_min("values", "out", count),
        ReductionMetric::CountNonZero => reduce_count_non_zero("values", "out", count),
        ReductionMetric::Any => reduce_any("values", "out", count),
        ReductionMetric::All => reduce_all("values", "out", count),
    };
    ensure_input_slots(&mut scratch.inputs, 2);
    write_u32_slice_le_bytes(&mut scratch.inputs[0], values);
    write_zero_bytes(&mut scratch.inputs[1], std::mem::size_of::<u32>());
    let outputs = execute_single_program(
        executor,
        crate::dispatch_buffers::HOST_WRAPPER_NODE,
        program,
        &scratch.inputs,
        policy,
    )?
    .outputs;
    decode_scalar(&outputs, "reduce_metric_via")
}

/// Wrapping sum of active work items through the reduce primitive.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when execution or readback fails.
pub fn reduce_sum_via(
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    values: &[u32],
) -> Result<u32, SemanticExecutionError> {
    reduce_metric_via(executor, policy, ReductionMetric::Sum, values)
}

/// Maximum queue depth through the reduce primitive.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when execution or readback fails.
pub fn reduce_max_via(
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    values: &[u32],
) -> Result<u32, SemanticExecutionError> {
    reduce_metric_via(executor, policy, ReductionMetric::Max, values)
}

/// Minimum remaining budget through the reduce primitive.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when execution or readback fails.
pub fn reduce_min_via(
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    values: &[u32],
) -> Result<u32, SemanticExecutionError> {
    reduce_metric_via(executor, policy, ReductionMetric::Min, values)
}

/// Non-zero lane count through the reduce primitive.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when execution or readback fails.
pub fn reduce_count_non_zero_via(
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    values: &[u32],
) -> Result<u32, SemanticExecutionError> {
    reduce_metric_via(executor, policy, ReductionMetric::CountNonZero, values)
}

/// Any-lane convergence predicate through the reduce primitive.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when execution or readback fails.
pub fn reduce_any_via(
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    values: &[u32],
) -> Result<bool, SemanticExecutionError> {
    Ok(reduce_metric_via(executor, policy, ReductionMetric::Any, values)? != 0)
}

/// All-lanes convergence predicate through the reduce primitive.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when execution or readback fails.
pub fn reduce_all_via(
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    values: &[u32],
) -> Result<bool, SemanticExecutionError> {
    Ok(reduce_metric_via(executor, policy, ReductionMetric::All, values)? != 0)
}

/// Per-segment wrapping sum through the segment reduction primitive.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when offsets are malformed, segment count is
/// unsupported by the primitive, execution fails, or readback is malformed.
pub fn segment_reduce_sum_via(
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    input: &[u32],
    segment_offsets: &[u32],
) -> Result<Vec<u32>, SemanticExecutionError> {
    let mut out = Vec::new();
    let mut scratch = ReductionMetricsGpuScratch::default();
    segment_reduce_sum_via_with_scratch_into(
        executor,
        policy,
        input,
        segment_offsets,
        &mut scratch,
        &mut out,
    )?;
    Ok(out)
}

/// Per-segment wrapping sum into caller-owned storage.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when validation, execution, or readback fails.
pub fn segment_reduce_sum_via_with_scratch_into(
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    input: &[u32],
    segment_offsets: &[u32],
    scratch: &mut ReductionMetricsGpuScratch,
    out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    bump_reduction_metrics_calls();

    let num_segments = validate_segment_offsets(input, segment_offsets)?;
    let input_count = u32::try_from(input.len().max(1)).map_err(|_| {
        SemanticExecutionError::InvalidRequest(format!(
            "Fix: segment_reduce_sum_via input length {} exceeds u32.",
            input.len()
        ))
    })?;
    let program = segment_reduce_sum(
        "input",
        "segment_offsets",
        "output",
        input_count,
        num_segments,
    );
    ensure_input_slots(&mut scratch.inputs, 3);
    if input.is_empty() {
        write_zero_bytes(&mut scratch.inputs[0], std::mem::size_of::<u32>());
    } else {
        write_u32_slice_le_bytes(&mut scratch.inputs[0], input);
    }
    write_u32_slice_le_bytes(&mut scratch.inputs[1], segment_offsets);
    write_zero_bytes(
        &mut scratch.inputs[2],
        num_segments as usize * std::mem::size_of::<u32>(),
    );
    let outputs = execute_single_program(
        executor,
        crate::dispatch_buffers::HOST_WRAPPER_NODE,
        program,
        &scratch.inputs,
        policy,
    )?
    .outputs;
    decode_first_output(
        &outputs,
        num_segments as usize,
        "segment_reduce_sum_via",
        out,
    )
}

/// Histogram with input-parallel atomic scatter semantics.
///
/// # Errors
///
/// Returns [`SemanticExecutionError`] when count/bin dimensions are zero or too
/// large, execution fails, or readback is malformed.
pub fn histogram_atomic_scatter_via(
    executor: &dyn SemanticExecutor,
    policy: &SemanticExecutionPolicy,
    input: &[u32],
    num_bins: u32,
) -> Result<Vec<u32>, SemanticExecutionError> {
    bump_reduction_metrics_calls();

    let count = checked_nonzero_len(input.len(), "histogram_atomic_scatter_via")?;
    if num_bins == 0 {
        return Err(SemanticExecutionError::InvalidRequest(
            "Fix: histogram_atomic_scatter_via requires num_bins > 0.".to_string(),
        ));
    }
    let bin_count = num_bins as usize;
    let program = histogram_atomic_scatter("input", "output", count, num_bins);
    let mut scratch = ReductionMetricsGpuScratch::default();
    ensure_input_slots(&mut scratch.inputs, 2);
    write_u32_slice_le_bytes(&mut scratch.inputs[0], input);
    write_zero_bytes(
        &mut scratch.inputs[1],
        bin_count * std::mem::size_of::<u32>(),
    );
    let outputs = execute_single_program(
        executor,
        crate::dispatch_buffers::HOST_WRAPPER_NODE,
        program,
        &scratch.inputs,
        policy,
    )?
    .outputs;
    let mut out = Vec::new();
    decode_first_output(
        &outputs,
        bin_count,
        "histogram_atomic_scatter_via",
        &mut out,
    )?;
    Ok(out)
}

fn bump_reduction_metrics_calls() {
    #[cfg(feature = "telemetry")]
    {
        use crate::telemetry::{bump, reduction_metrics_calls};
        bump(&reduction_metrics_calls);
    }
}

fn checked_len(len: usize, context: &'static str) -> Result<u32, SemanticExecutionError> {
    u32::try_from(len).map_err(|_| {
        SemanticExecutionError::InvalidRequest(format!(
            "Fix: {context} received {len} values, which exceeds the u32 GPU index space."
        ))
    })
}

fn checked_nonzero_len(len: usize, context: &'static str) -> Result<u32, SemanticExecutionError> {
    let count = checked_len(len, context)?;
    if count == 0 {
        return Err(SemanticExecutionError::InvalidRequest(format!(
            "Fix: {context} requires count > 0."
        )));
    }
    Ok(count)
}

fn validate_segment_offsets(
    input: &[u32],
    segment_offsets: &[u32],
) -> Result<u32, SemanticExecutionError> {
    if segment_offsets.len() < 2 {
        return Err(SemanticExecutionError::InvalidRequest(
            "Fix: segment_reduce_sum_via requires at least two CSR offsets.".to_string(),
        ));
    }
    let num_segments = segment_offsets.len() - 1;
    if num_segments > 256 {
        return Err(SemanticExecutionError::InvalidRequest(format!(
            "Fix: segment_reduce_sum_via supports at most 256 segments per primitive dispatch, got {num_segments}."
        )));
    }
    for (segment, pair) in segment_offsets.windows(2).enumerate() {
        let start = pair[0] as usize;
        let end = pair[1] as usize;
        if start > end || end > input.len() {
            return Err(SemanticExecutionError::InvalidRequest(format!(
                "Fix: segment_reduce_sum_via received malformed segment {segment}: start={start}, end={end}, input_len={}.",
                input.len()
            )));
        }
    }
    Ok(num_segments as u32)
}

fn decode_first_output(
    outputs: &[Vec<u8>],
    words: usize,
    context: &'static str,
    out: &mut Vec<u32>,
) -> Result<(), SemanticExecutionError> {
    if outputs.is_empty() {
        return Err(SemanticExecutionError::Backend(format!(
            "Fix: {context} expected at least one output buffer, got 0."
        )));
    }
    crate::dispatch_buffers::decode_u32_output_exact(&outputs[0], words, context, out)
        .map_err(|error| SemanticExecutionError::Backend(error.to_string()))
}

fn decode_scalar(
    outputs: &[Vec<u8>],
    context: &'static str,
) -> Result<u32, SemanticExecutionError> {
    let mut out = Vec::new();
    decode_first_output(outputs, 1, context, &mut out)?;
    Ok(out[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch_buffers::u32_slice_to_le_bytes;
    use crate::test_parity_oracles::{
        canonical_inputs, policy, semantic_output, NeverDispatches, StaticOutputs,
    };
    use vyre_megakernel::{SemanticExecutionOutput, SemanticExecutionRequest};
    use vyre_reference::composition_witness::{
        histogram_witness as reference_histogram_atomic_scatter,
        reduce_all_witness as reference_reduce_all_u32,
        reduce_any_witness as reference_reduce_any_u32,
        reduce_count_non_zero_witness as reference_reduce_count_non_zero,
        reduce_max_witness as reference_reduce_max, reduce_min_witness as reference_reduce_min,
        segment_reduce_sum_witness as primitive_segment_reduce_sum,
        wrapping_sum_witness as reference_reduce_sum,
    };

    fn reference_reduce_any(values: &[u32]) -> bool {
        reference_reduce_any_u32(values) != 0
    }

    fn reference_reduce_all(values: &[u32]) -> bool {
        reference_reduce_all_u32(values) != 0
    }

    struct ReduceExecutor;

    impl SemanticExecutor for ReduceExecutor {
        fn execute(
            &self,
            request: &SemanticExecutionRequest<'_>,
        ) -> Result<SemanticExecutionOutput, SemanticExecutionError> {
            let graph = request.logical().graph();
            let program = &graph
                .nodes()
                .first()
                .ok_or_else(|| {
                    SemanticExecutionError::InvalidRequest(
                        "Fix: reduction semantic executor requires one graph node.".to_string(),
                    )
                })?
                .program;
            let inputs = canonical_inputs(request)?;
            let op_id = program
                .entry
                .iter()
                .find_map(|node| match node {
                    vyre_foundation::ir::Node::Region { generator, .. } => Some(generator.as_str()),
                    _ => None,
                })
                .ok_or_else(|| {
                    SemanticExecutionError::InvalidRequest(
                        "Fix: reduction primitive should expose a region generator.".to_string(),
                    )
                })?;
            let values = crate::dispatch_buffers::read_u32s(required_input(&inputs, 0, op_id)?);
            let ordered = match op_id {
                crate::reduce::sum::OP_ID => {
                    require_input_count(&inputs, 2, op_id)?;
                    scalar(reference_reduce_sum(&values))
                }
                crate::reduce::max::OP_ID => {
                    require_input_count(&inputs, 2, op_id)?;
                    scalar(reference_reduce_max(&values))
                }
                crate::reduce::min::OP_ID => {
                    require_input_count(&inputs, 2, op_id)?;
                    scalar(reference_reduce_min(&values))
                }
                crate::reduce::count_non_zero::OP_ID => {
                    require_input_count(&inputs, 2, op_id)?;
                    scalar(reference_reduce_count_non_zero(&values))
                }
                crate::reduce::any::OP_ID => {
                    require_input_count(&inputs, 2, op_id)?;
                    scalar(reference_reduce_any_u32(&values))
                }
                crate::reduce::all::OP_ID => {
                    require_input_count(&inputs, 2, op_id)?;
                    scalar(reference_reduce_all_u32(&values))
                }
                crate::reduce::segment_reduce::OP_ID => {
                    require_input_count(&inputs, 3, op_id)?;
                    let offsets =
                        crate::dispatch_buffers::read_u32s(required_input(&inputs, 1, op_id)?);
                    vec![u32_slice_to_le_bytes(&primitive_segment_reduce_sum(
                        &values, &offsets,
                    ))]
                }
                crate::reduce::histogram::OP_ID => {
                    require_input_count(&inputs, 2, op_id)?;
                    let bins =
                        required_input(&inputs, 1, op_id)?.len() / std::mem::size_of::<u32>();
                    vec![u32_slice_to_le_bytes(&reference_histogram_atomic_scatter(
                        &values,
                        bins as u32,
                    ))]
                }
                other => {
                    return Err(SemanticExecutionError::InvalidRequest(format!(
                        "Fix: unexpected reduction primitive op id {other}."
                    )));
                }
            };
            semantic_output(request, ordered)
        }
    }

    fn required_input<'a>(
        inputs: &'a [Vec<u8>],
        index: usize,
        op_id: &str,
    ) -> Result<&'a [u8], SemanticExecutionError> {
        inputs.get(index).map(Vec::as_slice).ok_or_else(|| {
            SemanticExecutionError::InvalidRequest(format!(
                "Fix: reduction primitive {op_id} requires input {index}."
            ))
        })
    }

    fn require_input_count(
        inputs: &[Vec<u8>],
        expected: usize,
        op_id: &str,
    ) -> Result<(), SemanticExecutionError> {
        if inputs.len() != expected {
            return Err(SemanticExecutionError::InvalidRequest(format!(
                "Fix: reduction primitive {op_id} requires {expected} semantic inputs, got {}.",
                inputs.len()
            )));
        }
        Ok(())
    }

    fn scalar(value: u32) -> Vec<Vec<u8>> {
        vec![u32_slice_to_le_bytes(&[value])]
    }

    #[test]
    fn reference_reductions_match_primitives_exactly() {
        let values = [1u32, 0, 7, u32::MAX];
        assert_eq!(reference_reduce_sum(&values), 7);
        assert_eq!(reference_reduce_max(&values), u32::MAX);
        assert_eq!(reference_reduce_min(&values), 0);
        assert_eq!(reference_reduce_count_non_zero(&values), 3);
        assert!(reference_reduce_any(&values));
        assert!(!reference_reduce_all(&values));
    }

    #[test]
    fn scalar_reductions_execute_through_primitives() {
        let policy = policy();
        let values = [1u32, 0, 7, 3];
        assert_eq!(
            reduce_sum_via(&ReduceExecutor, &policy, &values).unwrap(),
            11
        );
        assert_eq!(
            reduce_max_via(&ReduceExecutor, &policy, &values).unwrap(),
            7
        );
        assert_eq!(
            reduce_min_via(&ReduceExecutor, &policy, &values).unwrap(),
            0
        );
        assert_eq!(
            reduce_count_non_zero_via(&ReduceExecutor, &policy, &values).unwrap(),
            3
        );
        assert!(reduce_any_via(&ReduceExecutor, &policy, &values).unwrap());
        assert!(!reduce_all_via(&ReduceExecutor, &policy, &values).unwrap());
    }

    #[test]
    fn segment_and_histogram_execute_through_primitives() {
        let policy = policy();
        assert_eq!(
            segment_reduce_sum_via(&ReduceExecutor, &policy, &[1, 2, 3, 4, 5], &[0, 2, 5],)
                .unwrap(),
            vec![3, 12]
        );
        assert_eq!(
            segment_reduce_sum_via(&ReduceExecutor, &policy, &[], &[0, 0, 0]).unwrap(),
            vec![0, 0]
        );
        assert_eq!(
            histogram_atomic_scatter_via(&ReduceExecutor, &policy, &[0, 1, 2, 1, 9], 4,).unwrap(),
            vec![1, 2, 1, 0]
        );
    }

    #[test]
    fn semantic_wrappers_preserve_resource_and_exact_output_contracts() {
        let policy = policy();
        let scalar_executor =
            StaticOutputs::new("scalar reduction", vec![u32_slice_to_le_bytes(&[9])])
                .expecting_inputs(&[2])
                .expecting_input_bytes(1, std::mem::size_of::<u32>());
        assert_eq!(
            reduce_sum_via(&scalar_executor, &policy, &[4, 5]).unwrap(),
            9
        );

        let segment_executor =
            StaticOutputs::new("segment reduction", vec![u32_slice_to_le_bytes(&[3, 12])])
                .expecting_inputs(&[3])
                .expecting_input_bytes(2, 2 * std::mem::size_of::<u32>());
        assert_eq!(
            segment_reduce_sum_via(&segment_executor, &policy, &[1, 2, 3, 4, 5], &[0, 2, 5],)
                .unwrap(),
            vec![3, 12]
        );

        let malformed = StaticOutputs::new("scalar reduction", vec![vec![9, 0, 0, 0, 1]])
            .expecting_inputs(&[2]);
        let error = reduce_sum_via(&malformed, &policy, &[4, 5])
            .expect_err("trailing scalar bytes must be rejected");
        assert!(error.to_string().contains("expected 4 output bytes, got 5"));
    }

    #[test]
    fn generated_large_scalar_reductions_match_oracles() {
        let policy = policy();
        for case in 0..4096u32 {
            let len = 257 + (case.wrapping_mul(31) % 1024) as usize;
            let mut state = 0xA11C_E5CAu32 ^ case.wrapping_mul(0x9E37_79B9);
            let mut values = Vec::with_capacity(len);
            for index in 0..len {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                values.push(if (state.wrapping_add(index as u32)) % 13 == 0 {
                    0
                } else {
                    state
                });
            }

            assert_eq!(
                reduce_sum_via(&ReduceExecutor, &policy, &values).unwrap(),
                reference_reduce_sum(&values),
                "case {case}: sum"
            );
            assert_eq!(
                reduce_max_via(&ReduceExecutor, &policy, &values).unwrap(),
                reference_reduce_max(&values),
                "case {case}: max"
            );
            assert_eq!(
                reduce_min_via(&ReduceExecutor, &policy, &values).unwrap(),
                reference_reduce_min(&values),
                "case {case}: min"
            );
            assert_eq!(
                reduce_count_non_zero_via(&ReduceExecutor, &policy, &values).unwrap(),
                reference_reduce_count_non_zero(&values),
                "case {case}: count_non_zero"
            );
            assert_eq!(
                reduce_any_via(&ReduceExecutor, &policy, &values).unwrap(),
                reference_reduce_any(&values),
                "case {case}: any"
            );
            assert_eq!(
                reduce_all_via(&ReduceExecutor, &policy, &values).unwrap(),
                reference_reduce_all(&values),
                "case {case}: all"
            );
        }
    }

    #[test]
    fn generated_large_histograms_match_oracles() {
        let policy = policy();
        for case in 0..4096u32 {
            let len = 257 + (case.wrapping_mul(17) % 1024) as usize;
            let bins = 1 + case.wrapping_mul(7) % 97;
            let mut state = 0xABCD_EF01u32 ^ case.wrapping_mul(0x85EB_CA6B);
            let mut input = Vec::with_capacity(len);
            for index in 0..len {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let value = if index % 11 == 0 {
                    bins + (state % 19)
                } else {
                    state % bins
                };
                input.push(value);
            }

            assert_eq!(
                histogram_atomic_scatter_via(&ReduceExecutor, &policy, &input, bins).unwrap(),
                reference_histogram_atomic_scatter(&input, bins),
                "case {case}: histogram"
            );
        }
    }

    #[test]
    fn scratch_path_reuses_buffers() {
        let policy = policy();
        let mut scratch = ReductionMetricsGpuScratch::default();
        assert_eq!(
            reduce_metric_via_with_scratch(
                &ReduceExecutor,
                &policy,
                ReductionMetric::CountNonZero,
                &[0, 1, 2],
                &mut scratch,
            )
            .unwrap(),
            2
        );
        let input_capacities = scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>();
        assert_eq!(
            reduce_metric_via_with_scratch(
                &ReduceExecutor,
                &policy,
                ReductionMetric::CountNonZero,
                &[0, 1, 2],
                &mut scratch,
            )
            .unwrap(),
            2
        );
        assert_eq!(
            scratch.inputs.iter().map(Vec::capacity).collect::<Vec<_>>(),
            input_capacities
        );
    }

    #[test]
    fn invalid_segment_offsets_are_actionable() {
        let error = segment_reduce_sum_via(
            &NeverDispatches("invalid offsets must fail before execution"),
            &policy(),
            &[1, 2],
            &[0, 3],
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("Fix: segment_reduce_sum_via received malformed segment"));
    }

    #[test]
    fn zero_bin_histogram_is_rejected_before_execution() {
        let error = histogram_atomic_scatter_via(
            &NeverDispatches("zero bins must fail before execution"),
            &policy(),
            &[1],
            0,
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("Fix: histogram_atomic_scatter_via requires num_bins > 0"));
    }
}
