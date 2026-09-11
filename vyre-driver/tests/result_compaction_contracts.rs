//! Contracts for `vyre_driver::result_compaction`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_driver::result_compaction::{
    plan_result_compaction, plan_result_compaction_with_scratch, CompactResultRecord,
    ResultCompactionError, ResultCompactionScratch, ResultSlot,
};

#[test]
fn result_compaction_packs_small_outputs_and_skips_empty_slots() {
    let plan = plan_result_compaction(&[slot(2, 0, 128), slot(1, 12, 128), slot(3, 24, 256)], 32)
        .expect("Fix: small outputs should compact");

    assert_eq!(
        plan.compact_records,
        vec![
            CompactResultRecord {
                slot: 1,
                compact_offset: 0,
                bytes: 12,
            },
            CompactResultRecord {
                slot: 3,
                compact_offset: 12,
                bytes: 24,
            },
        ]
    );
    assert_eq!(plan.direct_slots, Vec::<u32>::new());
    assert_eq!(plan.full_capacity_bytes, 512);
    assert_eq!(plan.compact_bytes, 36);
    assert_eq!(plan.direct_bytes, 0);
    assert_eq!(plan.selected_readback_bytes, 36);
    assert_eq!(plan.avoided_readback_bytes, 476);
    assert_eq!(plan.avoided_readback_basis_points, 9_296);
}

#[test]
fn result_compaction_keeps_large_outputs_direct() {
    let plan = plan_result_compaction(&[slot(1, 64, 128), slot(2, 512, 1_024)], 128)
        .expect("Fix: mixed outputs should plan");

    assert_eq!(plan.compact_records.len(), 1);
    assert_eq!(plan.direct_slots, vec![2]);
    assert_eq!(plan.full_capacity_bytes, 1_152);
    assert_eq!(plan.compact_bytes, 64);
    assert_eq!(plan.direct_bytes, 512);
    assert_eq!(plan.selected_readback_bytes, 576);
    assert_eq!(plan.avoided_readback_bytes, 576);
    assert_eq!(plan.avoided_readback_basis_points, 5_000);
}

#[test]
fn result_compaction_reports_zero_work_telemetry_without_division() {
    let plan = plan_result_compaction(&[slot(4, 0, 0), slot(9, 0, 0)], 128)
        .expect("Fix: zero-capacity outputs should plan");

    assert!(plan.compact_records.is_empty());
    assert!(plan.direct_slots.is_empty());
    assert_eq!(plan.full_capacity_bytes, 0);
    assert_eq!(plan.compact_bytes, 0);
    assert_eq!(plan.direct_bytes, 0);
    assert_eq!(plan.selected_readback_bytes, 0);
    assert_eq!(plan.avoided_readback_bytes, 0);
    assert_eq!(plan.avoided_readback_basis_points, 0);
}

#[test]
fn result_compaction_rejects_invalid_slots() {
    assert_eq!(
        plan_result_compaction(&[slot(1, 1, 8), slot(1, 1, 8)], 4)
            .expect_err("duplicate slots should fail"),
        ResultCompactionError::DuplicateSlot { slot: 1 }
    );
    assert_eq!(
        plan_result_compaction(&[slot(2, 9, 8)], 4)
            .expect_err("meaningful bytes above capacity should fail"),
        ResultCompactionError::MeaningfulExceedsCapacity {
            slot: 2,
            meaningful_bytes: 9,
            capacity_bytes: 8,
        }
    );
}

#[test]
fn result_compaction_reuses_caller_owned_slot_planning_scratch() {
    let mut scratch =
        ResultCompactionScratch::try_with_capacity(96).expect("Fix: scratch capacity");
    let wide = (0..96)
        .rev()
        .map(|index| slot(index, 8, 64))
        .collect::<Vec<_>>();
    let first = plan_result_compaction_with_scratch(&wide, 16, &mut scratch)
        .expect("Fix: wide compact result set should plan with reusable scratch");
    let id_capacity = scratch.id_capacity();
    let ordered_index_capacity = scratch.ordered_index_capacity();

    assert_eq!(first.compact_records.len(), 96);
    assert_eq!(first.compact_records[0].slot, 0);

    let second = plan_result_compaction_with_scratch(
        &[slot(7, 0, 128), slot(3, 512, 1_024), slot(5, 16, 128)],
        32,
        &mut scratch,
    )
    .expect("Fix: smaller mixed result set should reuse previous scratch");

    assert_eq!(second.compact_records[0].slot, 5);
    assert_eq!(second.direct_slots, vec![3]);
    assert!(scratch.id_capacity() >= id_capacity);
    assert!(scratch.ordered_index_capacity() >= ordered_index_capacity);
}

#[test]
fn generated_result_compaction_profiles_preserve_exact_telemetry_for_4096_shapes() {
    let mut scratch = ResultCompactionScratch::default();
    for slot_count in 1u32..=128 {
        for compact_threshold in 0u64..32 {
            let slots = (0..slot_count)
                .rev()
                .map(|slot_id| {
                    let meaningful = u64::from((slot_id % 17) + 1);
                    ResultSlot {
                        slot: slot_id,
                        meaningful_bytes: meaningful,
                        capacity_bytes: meaningful + compact_threshold + 8,
                    }
                })
                .collect::<Vec<_>>();

            let plan = plan_result_compaction_with_scratch(&slots, compact_threshold, &mut scratch)
                .expect("Fix: generated result compaction profile should plan");

            let expected_full = slots.iter().map(|slot| slot.capacity_bytes).sum::<u64>();
            let expected_selected = slots.iter().map(|slot| slot.meaningful_bytes).sum::<u64>();
            assert_eq!(plan.full_capacity_bytes, expected_full);
            assert_eq!(plan.selected_readback_bytes, expected_selected);
            assert_eq!(
                plan.avoided_readback_bytes,
                expected_full - expected_selected
            );
            assert!(plan
                .compact_records
                .windows(2)
                .all(|pair| pair[0].slot < pair[1].slot));
            assert!(plan.direct_slots.windows(2).all(|pair| pair[0] < pair[1]));
        }
    }
}

fn slot(slot: u32, meaningful_bytes: u64, capacity_bytes: u64) -> ResultSlot {
    ResultSlot {
        slot,
        meaningful_bytes,
        capacity_bytes,
    }
}
