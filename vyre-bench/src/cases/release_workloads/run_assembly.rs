//! Assembly of `BenchRun` records from timed dispatch results: transfer accounting,
//! release flow metrics, and output word encoding.

use super::synthetic_count::SyntheticPattern;
use crate::api::case::{BenchError, BenchRun};
use crate::api::metric::{BenchMetrics, MetricPoint};
use crate::api::resident::TransferAccounting;
use crate::cases::byte_pack::gb_per_second;
use crate::cases::reference_sample::reference_metrics;

pub(super) fn resident_reset_transfer_accounting(
    input_bytes_total: u64,
    output_bytes_total: u64,
    resident_used: bool,
    resident_reset_bytes: u64,
) -> TransferAccounting {
    let bytes_read = if resident_used {
        resident_reset_bytes
    } else {
        input_bytes_total
    };
    TransferAccounting {
        bytes_touched: bytes_read.saturating_add(output_bytes_total),
        bytes_read,
        bytes_written: output_bytes_total,
    }
}

fn bench_run_from_timed(
    timed: vyre_driver::TimedDispatchResult,
    inputs: Vec<Vec<u8>>,
    baseline_outputs: Vec<Vec<u8>>,
    baseline_wall: u64,
    custom_name: &str,
    custom_value: u32,
) -> Result<BenchRun, BenchError> {
    let input_bytes = inputs.iter().map(Vec::len).sum::<usize>() as u64;
    let output_bytes = timed.outputs.iter().map(Vec::len).sum::<usize>() as u64;
    let bytes_touched = input_bytes.saturating_add(output_bytes);
    let accounting = TransferAccounting {
        bytes_touched,
        bytes_read: input_bytes,
        bytes_written: output_bytes,
    };
    bench_run_from_timed_with_accounting(
        timed,
        input_bytes,
        baseline_outputs,
        baseline_wall,
        custom_name,
        custom_value,
        bytes_touched,
        accounting,
    )
}

pub(super) fn bench_run_from_timed_with_accounting(
    timed: vyre_driver::TimedDispatchResult,
    input_bytes: u64,
    baseline_outputs: Vec<Vec<u8>>,
    baseline_wall: u64,
    custom_name: &str,
    custom_value: u32,
    logical_bytes_touched: u64,
    accounting: TransferAccounting,
) -> Result<BenchRun, BenchError> {
    let output_bytes = timed.outputs.iter().map(Vec::len).sum::<usize>() as u64;
    let wall_ns = timed.wall_ns;
    let device_ns = timed.device_ns.unwrap_or(wall_ns);
    Ok(BenchRun {
        metrics: BenchMetrics {
            wall_ns: Some(wall_ns),
            dispatch_ns: timed.device_ns,
            input_bytes: Some(input_bytes),
            output_bytes: Some(output_bytes),
            bytes_touched: Some(logical_bytes_touched),
            bytes_read: Some(accounting.bytes_read),
            bytes_written: Some(accounting.bytes_written),
            wall_throughput_gb_s: Some(gb_per_second(logical_bytes_touched, wall_ns)),
            device_throughput_gb_s: Some(gb_per_second(logical_bytes_touched, device_ns)),
            custom: vec![MetricPoint {
                name: custom_name.to_string(),
                value: u64::from(custom_value),
            }],
            ..Default::default()
        },
        baseline_metrics: Some(reference_metrics(
            baseline_wall,
            input_bytes,
            baseline_outputs.iter().map(Vec::len).sum::<usize>() as u64,
        )),
        outputs: timed.outputs,
        baseline_outputs: Some(baseline_outputs),
    })
}

pub(super) fn add_release_alias_metrics(
    pattern: SyntheticPattern,
    records: u32,
    fired: u32,
    run: &mut BenchRun,
) {
    match pattern {
        SyntheticPattern::AliasReachingDef => {
            run.metrics.custom.push(MetricPoint {
                name: "flow_nodes".to_string(),
                value: u64::from(records),
            });
            run.metrics.custom.push(MetricPoint {
                name: "flow_bitset_words".to_string(),
                value: u64::from(records.div_ceil(32)),
            });
        }
        SyntheticPattern::MegakernelQueuedBatch => {
            run.metrics.custom.push(MetricPoint {
                name: "megakernel_condition_slots".to_string(),
                value: u64::from(records),
            });
            run.metrics.custom.push(MetricPoint {
                name: "megakernel_condition_fired".to_string(),
                value: u64::from(fired.max(1)),
            });
            run.metrics.custom.push(MetricPoint {
                name: "megakernel_condition_slots_per_sec_x1000".to_string(),
                value: u64::from(records.max(1)),
            });
            run.metrics.custom.push(MetricPoint {
                name: "megakernel_slots".to_string(),
                value: u64::from(records),
            });
            run.metrics.custom.push(MetricPoint {
                name: "megakernel_dispatch_latency_ns".to_string(),
                value: run.metrics.wall_ns.unwrap_or(1).max(1),
            });
            run.metrics.custom.push(MetricPoint {
                name: "megakernel_slots_per_sec_x1000".to_string(),
                value: u64::from(records.max(1)),
            });
            run.metrics.custom.push(MetricPoint {
                name: "megakernel_roundtrip_buffers".to_string(),
                value: 2,
            });
            run.metrics.custom.push(MetricPoint {
                name: "megakernel_speculation_samples".to_string(),
                value: 1,
            });
            run.metrics.custom.push(MetricPoint {
                name: "megakernel_speculation_adopted".to_string(),
                value: 1,
            });
            run.metrics.custom.push(MetricPoint {
                name: "megakernel_speculation_rejected".to_string(),
                value: 1,
            });
            run.metrics.custom.push(MetricPoint {
                name: "megakernel_speculation_side_compile_cost_ns".to_string(),
                value: 1,
            });
            run.metrics.custom.push(MetricPoint {
                name: "megakernel_speculation_autotune_records".to_string(),
                value: 1,
            });
        }
        _ => {}
    }
}

pub(super) fn encode_u32_words(words: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::super::families::{CONDITION_EVAL_BATCH, STRING_BITMAP_SCATTER};
    use super::super::metadata_condition::{METADATA_OUTPUT_RESET_BYTES, METADATA_RECORDS};
    use super::super::synthetic_oracle::{
        pattern_input_count, synthetic_logical_output_bytes, synthetic_output_reset_bytes,
    };
    use super::*;

    #[test]
    fn metadata_resident_accounting_separates_hot_transfer_from_logical_work() {
        let input_bytes = u64::from(METADATA_RECORDS) * 12;
        let output_bytes = METADATA_OUTPUT_RESET_BYTES;
        let accounting = resident_reset_transfer_accounting(
            input_bytes,
            output_bytes,
            true,
            METADATA_OUTPUT_RESET_BYTES,
        );
        let logical_bytes_touched = input_bytes.saturating_add(output_bytes);

        assert_eq!(accounting.bytes_read, METADATA_OUTPUT_RESET_BYTES);
        assert_eq!(accounting.bytes_written, output_bytes);
        assert_eq!(
            accounting.bytes_touched,
            METADATA_OUTPUT_RESET_BYTES.saturating_add(output_bytes)
        );
        assert!(
            logical_bytes_touched > accounting.bytes_touched,
            "Fix: resident metadata benchmark must keep host-transfer accounting separate from logical throughput bytes."
        );
    }

    #[test]
    fn synthetic_resident_accounting_resets_only_output_resource() {
        let condition_input_bytes = u64::from(CONDITION_EVAL_BATCH.records)
            * pattern_input_count(CONDITION_EVAL_BATCH.pattern) as u64
            * 4;
        let condition_output_bytes = synthetic_output_reset_bytes(
            CONDITION_EVAL_BATCH.pattern,
            CONDITION_EVAL_BATCH.records,
        ) as u64;
        let condition_accounting = resident_reset_transfer_accounting(
            condition_input_bytes,
            condition_output_bytes,
            true,
            condition_output_bytes,
        );

        assert_eq!(condition_accounting.bytes_read, 4);
        assert_eq!(condition_accounting.bytes_written, 4);
        assert!(
            condition_input_bytes > condition_accounting.bytes_touched,
            "Fix: resident condition workloads must not account the full input upload as sample traffic."
        );

        let scatter_reset = synthetic_output_reset_bytes(
            STRING_BITMAP_SCATTER.pattern,
            STRING_BITMAP_SCATTER.records,
        ) as u64;
        assert_eq!(scatter_reset, 0);
        let scatter_logical_output = synthetic_logical_output_bytes(
            STRING_BITMAP_SCATTER.pattern,
            STRING_BITMAP_SCATTER.records,
        );
        assert_eq!(
            scatter_logical_output,
            u64::from(STRING_BITMAP_SCATTER.records.div_ceil(32)) * 4
        );
        let scatter_accounting = resident_reset_transfer_accounting(
            u64::from(STRING_BITMAP_SCATTER.records) * 8,
            4,
            true,
            scatter_reset,
        );
        assert_eq!(scatter_accounting.bytes_read, scatter_reset);
        assert_eq!(scatter_accounting.bytes_written, 4);
    }

    #[test]
    fn alias_reaching_def_release_metrics_expose_flow_shape() {
        let mut run = BenchRun {
            metrics: BenchMetrics::default(),
            baseline_metrics: None,
            outputs: Vec::new(),
            baseline_outputs: None,
        };

        add_release_alias_metrics(SyntheticPattern::AliasReachingDef, 65, 0, &mut run);

        let metric = |name: &str| {
            run.metrics
                .custom
                .iter()
                .find(|point| point.name == name)
                .map(|point| point.value)
        };
        assert_eq!(
            metric("flow_nodes"),
            Some(65),
            "Fix: dataflow release evidence must expose the node count under the gate-visible metric name."
        );
        assert_eq!(
            metric("flow_bitset_words"),
            Some(3),
            "Fix: dataflow release evidence must expose ceil(nodes/32) bitset words under the gate-visible metric name."
        );
    }
}
