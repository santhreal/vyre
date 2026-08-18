//! Contracts for `vyre_driver::device_work_queue`.
//!
//! Every item under test is public API, so the suite reaches the crate the way
//! a consumer does.

use vyre_driver::device_work_queue::{
    plan_device_work_queue, plan_device_work_queue_backpressure,
    plan_device_work_queue_with_expansion, DeviceWorkQueueDrainStrategy, DeviceWorkQueueError,
    DeviceWorkQueueExpansionProfile, DeviceWorkQueueProfile, WorkQueueHostSync,
};

#[test]
fn device_work_queue_plans_final_only_resident_execution() {
    let plan = plan_device_work_queue(DeviceWorkQueueProfile {
        initial_items: 256,
        queue_capacity: 1_024,
        entry_bytes: 16,
        control_bytes: 128,
        budget_bytes: 32_768,
        host_sync: WorkQueueHostSync::FinalOnly,
    })
    .expect("Fix: valid device work queue should plan");

    assert_eq!(plan.queue_bytes, 16_384);
    assert_eq!(plan.control_bytes, 128);
    assert_eq!(plan.resident_bytes, 16_512);
    assert_eq!(plan.initial_occupancy_bps, 2_500);
    assert!(plan.final_only_host_sync);
}

#[test]
fn device_work_queue_expansion_uses_budgeted_resident_headroom() {
    let plan = plan_device_work_queue_with_expansion(DeviceWorkQueueExpansionProfile {
        initial_items: 4,
        expansion_items: 12,
        entry_bytes: 8,
        control_bytes: 64,
        budget_bytes: 256,
        host_sync: WorkQueueHostSync::FinalOnly,
    })
    .expect("Fix: expansion headroom should fit inside the explicit queue budget");

    assert_eq!(plan.queue_bytes, 128);
    assert_eq!(plan.control_bytes, 64);
    assert_eq!(plan.resident_bytes, 192);
    assert_eq!(
        plan.initial_occupancy_bps, 2_500,
        "Fix: occupancy must use the expanded resident queue capacity"
    );
    assert!(plan.final_only_host_sync);
}

#[test]
fn device_work_queue_expansion_clamps_to_budget_without_dropping_initial_items() {
    let plan = plan_device_work_queue_with_expansion(DeviceWorkQueueExpansionProfile {
        initial_items: 4,
        expansion_items: 100,
        entry_bytes: 8,
        control_bytes: 16,
        budget_bytes: 96,
        host_sync: WorkQueueHostSync::FinalOnly,
    })
    .expect("Fix: queue expansion should use all affordable headroom");

    assert_eq!(plan.queue_bytes, 80);
    assert_eq!(plan.resident_bytes, 96);
    assert_eq!(
        plan.initial_occupancy_bps, 4_000,
        "Fix: initial occupancy should reflect budget-clamped expansion capacity"
    );
}

#[test]
fn device_work_queue_expansion_fails_when_initial_frontier_cannot_fit() {
    assert_eq!(
        plan_device_work_queue_with_expansion(DeviceWorkQueueExpansionProfile {
            initial_items: 8,
            expansion_items: 100,
            entry_bytes: 16,
            control_bytes: 64,
            budget_bytes: 128,
            host_sync: WorkQueueHostSync::FinalOnly,
        })
        .expect_err("initial frontier must fail when it cannot fit the explicit budget"),
        DeviceWorkQueueError::OverBudget {
            required_bytes: 192,
            budget_bytes: 128,
        }
    );
}

#[test]
fn device_work_queue_expansion_rejects_capacity_overflow() {
    assert_eq!(
        plan_device_work_queue_with_expansion(DeviceWorkQueueExpansionProfile {
            initial_items: u64::MAX,
            expansion_items: 1,
            entry_bytes: 1,
            control_bytes: 0,
            budget_bytes: u64::MAX,
            host_sync: WorkQueueHostSync::FinalOnly,
        })
        .expect_err("overflowed expansion capacity must fail before queue planning"),
        DeviceWorkQueueError::ByteCountOverflow {
            field: "queue expansion capacity",
        }
    );
}

#[test]
fn device_work_queue_rejects_host_participation() {
    assert_eq!(
        plan_device_work_queue(DeviceWorkQueueProfile {
            initial_items: 1,
            queue_capacity: 8,
            entry_bytes: 16,
            control_bytes: 64,
            budget_bytes: 1_024,
            host_sync: WorkQueueHostSync::HostParticipates,
        })
        .expect_err("host participation should fail"),
        DeviceWorkQueueError::HostParticipationRejected
    );
}

#[test]
fn device_work_queue_rejects_invalid_capacity_and_budget() {
    assert_eq!(
        plan_device_work_queue(DeviceWorkQueueProfile {
            initial_items: 9,
            queue_capacity: 8,
            entry_bytes: 16,
            control_bytes: 64,
            budget_bytes: 1_024,
            host_sync: WorkQueueHostSync::FinalOnly,
        })
        .expect_err("initial overflow should fail"),
        DeviceWorkQueueError::InitialItemsExceedCapacity {
            initial_items: 9,
            queue_capacity: 8,
        }
    );
    assert_eq!(
        plan_device_work_queue(DeviceWorkQueueProfile {
            initial_items: 1,
            queue_capacity: 8,
            entry_bytes: 16,
            control_bytes: 64,
            budget_bytes: 128,
            host_sync: WorkQueueHostSync::FinalOnly,
        })
        .expect_err("over-budget queue should fail"),
        DeviceWorkQueueError::OverBudget {
            required_bytes: 192,
            budget_bytes: 128,
        }
    );
}

#[test]
fn device_work_queue_occupancy_uses_widened_arithmetic_for_huge_queues() {
    let plan = plan_device_work_queue(DeviceWorkQueueProfile {
        initial_items: u64::MAX,
        queue_capacity: u64::MAX,
        entry_bytes: 1,
        control_bytes: 0,
        budget_bytes: u64::MAX,
        host_sync: WorkQueueHostSync::FinalOnly,
    })
    .expect("Fix: max-sized byte queue should fit exactly");

    assert_eq!(
        plan.initial_occupancy_bps, 10_000,
        "Fix: device work-queue occupancy must not use saturating u64 multiplication before division; full queues must report 10000 bps even near u64::MAX."
    );
}

#[test]
fn device_work_queue_backpressure_chunks_large_resident_queues_without_host_participation() {
    let plan = plan_device_work_queue_backpressure(
        DeviceWorkQueueProfile {
            initial_items: 4_096,
            queue_capacity: 65_536,
            entry_bytes: 16,
            control_bytes: 128,
            budget_bytes: 2 << 20,
            host_sync: WorkQueueHostSync::FinalOnly,
        },
        8_192,
    )
    .expect("Fix: large resident work queue should plan bounded device-side drain chunks");

    assert_eq!(
        plan.strategy,
        DeviceWorkQueueDrainStrategy::ChunkedResidentDrain
    );
    assert_eq!(plan.items_per_chunk, 8_192);
    assert_eq!(plan.chunks, 8);
    assert_eq!(plan.queue.resident_bytes, 1_048_704);
    assert!(plan.final_only_host_sync);
    assert!(plan.queue.final_only_host_sync);
}

#[test]
fn device_work_queue_backpressure_ceil_division_handles_max_capacity() {
    let plan = plan_device_work_queue_backpressure(
        DeviceWorkQueueProfile {
            initial_items: u64::MAX,
            queue_capacity: u64::MAX,
            entry_bytes: 1,
            control_bytes: 0,
            budget_bytes: u64::MAX,
            host_sync: WorkQueueHostSync::FinalOnly,
        },
        65_536,
    )
    .expect("Fix: ceil division for max-capacity queues must not overflow");

    assert_eq!(
        plan.strategy,
        DeviceWorkQueueDrainStrategy::ChunkedResidentDrain
    );
    assert_eq!(plan.queue.queue_bytes, u64::MAX);
    assert_eq!(plan.items_per_chunk, 65_536);
    assert_eq!(plan.chunks, 281_474_976_710_656);
    assert!(plan.final_only_host_sync);
}

#[test]
fn device_work_queue_backpressure_rejects_zero_drain_chunk() {
    let err = plan_device_work_queue_backpressure(
        DeviceWorkQueueProfile {
            initial_items: 1,
            queue_capacity: 8,
            entry_bytes: 16,
            control_bytes: 64,
            budget_bytes: 1_024,
            host_sync: WorkQueueHostSync::FinalOnly,
        },
        0,
    )
    .expect_err("zero drain chunk must fail loudly");

    assert_eq!(err, DeviceWorkQueueError::ZeroDrainChunk);
}

#[test]
fn generated_device_work_queue_profiles_preserve_budget_and_sync_contracts() {
    let mut state = 0xa409_3822_299f_31d0_u64;
    for case_index in 0..2048usize {
        let queue_capacity = 1 + next_u64(&mut state) % 262_144;
        let entry_bytes = 1 + next_u64(&mut state) % 256;
        let initial_items = next_u64(&mut state) % (queue_capacity + 1);
        let control_bytes = next_u64(&mut state) % 4096;
        let queue_bytes = queue_capacity
            .checked_mul(entry_bytes)
            .expect("Fix: generated queue byte count should fit");
        let resident_bytes = queue_bytes
            .checked_add(control_bytes)
            .expect("Fix: generated resident byte count should fit");
        let budget_bytes = resident_bytes + (next_u64(&mut state) % 8192);
        let profile = DeviceWorkQueueProfile {
            initial_items,
            queue_capacity,
            entry_bytes,
            control_bytes,
            budget_bytes,
            host_sync: WorkQueueHostSync::FinalOnly,
        };

        let plan =
            plan_device_work_queue(profile).expect("Fix: generated valid queue profile must plan");
        assert_eq!(plan.queue_bytes, queue_bytes, "case {case_index}");
        assert_eq!(plan.control_bytes, control_bytes, "case {case_index}");
        assert_eq!(plan.resident_bytes, resident_bytes, "case {case_index}");
        assert!(plan.resident_bytes <= budget_bytes, "case {case_index}");
        assert!(plan.initial_occupancy_bps <= 10_000, "case {case_index}");
        assert!(plan.final_only_host_sync, "case {case_index}");

        let drain = 1 + next_u64(&mut state) % queue_capacity;
        let backpressure = plan_device_work_queue_backpressure(profile, drain)
            .expect("Fix: generated valid backpressure profile must plan");
        assert_eq!(backpressure.queue, plan, "case {case_index}");
        assert!(
            backpressure.items_per_chunk <= queue_capacity,
            "case {case_index}"
        );
        assert!(backpressure.chunks >= 1, "case {case_index}");
        assert!(backpressure.final_only_host_sync, "case {case_index}");

        let expansion_items = next_u64(&mut state) % queue_capacity;
        let expansion_budget = resident_bytes + (expansion_items * entry_bytes);
        let expansion = plan_device_work_queue_with_expansion(DeviceWorkQueueExpansionProfile {
            initial_items,
            expansion_items,
            entry_bytes,
            control_bytes,
            budget_bytes: expansion_budget,
            host_sync: WorkQueueHostSync::FinalOnly,
        })
        .expect("Fix: generated valid expansion queue profile must plan");
        assert!(
            expansion.resident_bytes <= expansion_budget,
            "case {case_index}"
        );
        assert!(
            expansion.queue_bytes >= initial_items * entry_bytes,
            "case {case_index}"
        );
        assert!(expansion.final_only_host_sync, "case {case_index}");
    }
}

fn next_u64(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}
