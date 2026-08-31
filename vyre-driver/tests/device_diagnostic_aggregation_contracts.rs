//! Contracts for `vyre_driver::device_diagnostic_aggregation`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_driver::device_diagnostic_aggregation::{
    plan_device_diagnostic_aggregation, plan_device_diagnostic_aggregation_with_scratch,
    DiagnosticAggregationError, DiagnosticAggregationScratch, DiagnosticCompactRange,
    DiagnosticShard,
};

#[test]
fn diagnostic_aggregation_compacts_sparse_device_diagnostics() {
    let plan = plan_device_diagnostic_aggregation(
        &[
            shard(2, 2_000, 4, 32, 24, 16, 0b010),
            shard(1, 1_000, 2, 32, 24, 16, 0b001),
            shard(3, 4_000, 0, 32, 24, 16, 0),
        ],
        64,
        1_024,
    )
    .expect("Fix: sparse diagnostics should aggregate on device");

    assert_eq!(
        plan.compact_ranges,
        vec![
            DiagnosticCompactRange {
                shard: 1,
                compact_offset: 0,
                records: 2,
                bytes: 48,
            },
            DiagnosticCompactRange {
                shard: 2,
                compact_offset: 48,
                records: 4,
                bytes: 96,
            },
        ]
    );
    assert_eq!(plan.counter_readback_bytes, 48);
    assert_eq!(plan.compact_readback_bytes, 144);
    assert_eq!(plan.host_readback_bytes, 192);
    assert_eq!(plan.raw_candidate_readback_bytes, 224_000);
    assert_eq!(plan.avoided_readback_bytes, 223_808);
    assert!(plan.compression_ratio_bps < 10);
    assert!(plan.requires_device_prefix_scan);
    assert!(plan.final_only_host_readback);
}

#[test]
fn diagnostic_aggregation_caps_overflow_without_host_filtering() {
    let plan = plan_device_diagnostic_aggregation(&[shard(7, 1_000, 10, 32, 16, 8, 0b111)], 3, 128)
        .expect("Fix: overflow should be represented by device-side flags");

    assert_eq!(plan.compact_ranges[0].records, 3);
    assert_eq!(plan.overflow_records, 7);
    assert!(plan.requires_overflow_flag);
    assert_eq!(plan.host_readback_bytes, 56);
    assert!(
        !plan.requires_device_prefix_scan,
        "Fix: a single non-empty diagnostic shard has compact offset zero and must not schedule a device prefix scan."
    );
}

#[test]
fn diagnostic_aggregation_ratio_does_not_saturate_before_division() {
    let plan = plan_device_diagnostic_aggregation(
        &[shard(9, u64::MAX / 32, 1, 32, 16, u64::MAX / 20, 0b001)],
        1,
        u64::MAX,
    )
    .expect("Fix: large diagnostic plans must retain exact ratio arithmetic");

    let expected = (((plan.host_readback_bytes as u128) * 10_000)
        / plan.raw_candidate_readback_bytes as u128) as u32;
    assert_eq!(plan.compression_ratio_bps, expected);
    assert!(plan.compression_ratio_bps > 100);
}

#[test]
fn diagnostic_aggregation_rejects_invalid_or_cpu_shaped_inputs() {
    assert_eq!(
        plan_device_diagnostic_aggregation(
            &[shard(1, 8, 1, 32, 24, 8, 1), shard(1, 8, 1, 32, 24, 8, 1)],
            4,
            1_024,
        )
        .expect_err("duplicate shard should fail"),
        DiagnosticAggregationError::DuplicateShard { shard: 1 }
    );
    assert_eq!(
        plan_device_diagnostic_aggregation(&[shard(2, 8, 9, 32, 24, 8, 1)], 4, 1_024)
            .expect_err("emitted diagnostics cannot exceed candidates"),
        DiagnosticAggregationError::EmittedExceedsCandidates {
            shard: 2,
            emitted_diagnostics: 9,
            candidate_items: 8,
        }
    );
    assert_eq!(
        plan_device_diagnostic_aggregation(&[shard(3, 8, 1, 32, 24, 8, 0)], 4, 1_024)
            .expect_err("diagnostics must retain class mask"),
        DiagnosticAggregationError::MissingSeverityMask { shard: 3 }
    );
    assert_eq!(
        plan_device_diagnostic_aggregation(&[shard(4, 8, 1, 32, 24, 8, 1)], 4, 16)
            .expect_err("over budget plan should fail"),
        DiagnosticAggregationError::OverBudget {
            required_bytes: 32,
            budget_bytes: 16,
        }
    );
}

#[test]
fn diagnostic_aggregation_reports_zero_avoided_bytes_when_counters_exceed_raw_stream() {
    let plan = plan_device_diagnostic_aggregation(
        &[shard(1, 1, 0, 1, 8, 64, 0)],
        1,
        128,
    )
    .expect("Fix: diagnostic aggregation should report negative savings as zero avoided bytes, not fail with underflow");

    assert_eq!(plan.raw_candidate_readback_bytes, 1);
    assert_eq!(plan.host_readback_bytes, 64);
    assert_eq!(plan.avoided_readback_bytes, 0);
    assert_eq!(plan.compression_ratio_bps, 640_000);
    assert!(plan.final_only_host_readback);
}

#[test]
fn diagnostic_aggregation_reuses_caller_owned_shard_planning_scratch() {
    let mut scratch =
        DiagnosticAggregationScratch::try_with_capacity(128).expect("Fix: scratch capacity");
    let wide = (0..128)
        .rev()
        .map(|index| shard(index, 1_024, 1, 32, 16, 8, 1))
        .collect::<Vec<_>>();
    let first = plan_device_diagnostic_aggregation_with_scratch(&wide, 4, 1 << 20, &mut scratch)
        .expect("Fix: wide diagnostic aggregation should plan with reusable scratch");
    let id_capacity = scratch.id_capacity();
    let ordered_index_capacity = scratch.ordered_index_capacity();

    assert_eq!(first.compact_ranges.len(), 128);
    assert_eq!(first.compact_ranges[0].shard, 0);

    let second = plan_device_diagnostic_aggregation_with_scratch(
        &[
            shard(9, 1_000, 0, 32, 24, 16, 0),
            shard(3, 1_000, 7, 32, 24, 16, 1),
        ],
        3,
        1 << 20,
        &mut scratch,
    )
    .expect("Fix: smaller diagnostic aggregation should reuse previous scratch");

    assert_eq!(second.compact_ranges[0].shard, 3);
    assert_eq!(second.overflow_records, 4);
    assert!(scratch.id_capacity() >= id_capacity);
    assert!(scratch.ordered_index_capacity() >= ordered_index_capacity);
}

#[test]
fn generated_diagnostic_aggregation_profiles_preserve_exact_telemetry_for_4096_shapes() {
    let mut scratch = DiagnosticAggregationScratch::default();
    for shard_count in 1u32..=128 {
        for cap in 1u64..=32 {
            let shards = (0..shard_count)
                .rev()
                .map(|id| {
                    let candidates = u64::from((id % 19) + 1) * 8;
                    let emitted = u64::from(id % 7);
                    shard(
                        id,
                        candidates,
                        emitted.min(candidates),
                        16,
                        12,
                        8,
                        if emitted == 0 { 0 } else { 1 << (id % 8) },
                    )
                })
                .collect::<Vec<_>>();

            let plan = plan_device_diagnostic_aggregation_with_scratch(
                &shards,
                cap,
                u64::MAX,
                &mut scratch,
            )
            .expect("Fix: generated diagnostic aggregation profile should plan");

            let expected_raw = shards
                .iter()
                .map(|shard| shard.candidate_items * shard.raw_item_bytes)
                .sum::<u64>();
            let expected_counter = shards.iter().map(|shard| shard.counter_bytes).sum::<u64>();
            let expected_compact = shards
                .iter()
                .map(|shard| shard.emitted_diagnostics.min(cap) * shard.diagnostic_record_bytes)
                .sum::<u64>();
            assert_eq!(plan.raw_candidate_readback_bytes, expected_raw);
            assert_eq!(plan.counter_readback_bytes, expected_counter);
            assert_eq!(plan.compact_readback_bytes, expected_compact);
            assert_eq!(
                plan.host_readback_bytes,
                expected_counter + expected_compact
            );
            assert!(plan
                .compact_ranges
                .windows(2)
                .all(|pair| pair[0].shard < pair[1].shard));
            assert!(plan.final_only_host_readback);
        }
    }
}

fn shard(
    shard: u32,
    candidate_items: u64,
    emitted_diagnostics: u64,
    raw_item_bytes: u64,
    diagnostic_record_bytes: u64,
    counter_bytes: u64,
    severity_mask: u32,
) -> DiagnosticShard {
    DiagnosticShard {
        shard,
        candidate_items,
        emitted_diagnostics,
        raw_item_bytes,
        diagnostic_record_bytes,
        counter_bytes,
        severity_mask,
    }
}
