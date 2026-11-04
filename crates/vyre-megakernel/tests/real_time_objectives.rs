//! Tests for versioned real-time objective family, measurement boundaries, and hard-constraint verification.
//!
//! Acceptance criteria:
//! 1. Versioned real-time objective family (deadlines, jitter, queueing, arrival traces, memory ceilings, energy policies).
//! 2. Measurement boundary tracking input-to-visible latency with raw samples, uncertainty, and missed deadlines.
//! 3. Rejection of schedules violating hard deadlines or jitter bounds even if they offer higher average throughput.

use vyre_megakernel::cost::CostBreakdown;
use vyre_megakernel::{
    DeviceFacts, InputToVisibleMeasurement, LatencyPercentile, RealTimeDeadline, RealTimeObjective,
    RealTimeViolation, WorkloadArrivalTrace, REAL_TIME_OBJECTIVE_SCHEMA_VERSION,
};

#[test]
fn real_time_objective_schema_and_presets() {
    let obj_60fps = RealTimeObjective::interactive_60fps();
    assert_eq!(obj_60fps.version(), REAL_TIME_OBJECTIVE_SCHEMA_VERSION);
    assert_eq!(
        obj_60fps.deadline(),
        RealTimeDeadline::InteractiveFrame {
            frame_target_ns: 16_666_667,
            target_fps: 60,
        }
    );
    assert_eq!(obj_60fps.deadline().budget_ns(), 16_666_667);
    assert_eq!(obj_60fps.percentile(), LatencyPercentile::P99);
    assert_eq!(obj_60fps.jitter_limit_ns(), Some(2_000_000));

    let obj_120fps = RealTimeObjective::interactive_120fps();
    assert_eq!(
        obj_120fps.deadline(),
        RealTimeDeadline::InteractiveFrame {
            frame_target_ns: 8_333_333,
            target_fps: 120,
        }
    );
    assert_eq!(obj_120fps.percentile(), LatencyPercentile::P999);
    assert_eq!(obj_120fps.jitter_limit_ns(), Some(1_000_000));

    let hard_rt = RealTimeObjective::hard_real_time(5_000_000, 500_000);
    assert_eq!(hard_rt.deadline().budget_ns(), 5_000_000);
    assert_eq!(hard_rt.deadline().max_jitter_ns(), Some(500_000));
    assert_eq!(hard_rt.percentile(), LatencyPercentile::WorstCase);
}

#[test]
fn real_time_objective_builder_and_arrival_traces() {
    let obj = RealTimeObjective::interactive_60fps()
        .with_memory_ceiling(64 * 1024 * 1024)
        .with_compile_budget_us(15_000)
        .with_arrival_trace(WorkloadArrivalTrace::Burst {
            burst_size: 4,
            interval_ns: 16_666_667,
        });

    assert_eq!(obj.memory_ceiling_bytes(), Some(64 * 1024 * 1024));
}

#[test]
fn hard_constraint_rejection_over_deadline_breach() {
    let objective = RealTimeObjective::hard_real_time(10_000_000, 1_000_000);
    let device = DeviceFacts::unknown();

    // Schedule A: total 8ms (8,000,000ns) <= 10ms budget -> Satisfied
    let mut cost_ok = CostBreakdown::default();
    cost_ok.total = 8_000_000;
    assert!(objective.satisfies_constraints(&cost_ok, device).is_ok());

    // Schedule B: total 12ms (12,000,000ns) > 10ms budget -> Rejected with DeadlineExceeded
    let mut cost_slow = CostBreakdown::default();
    cost_slow.total = 12_000_000;
    let err = objective
        .satisfies_constraints(&cost_slow, device)
        .expect_err("slow schedule must be rejected");

    assert!(matches!(
        err,
        RealTimeViolation::DeadlineExceeded {
            limit_ns: 10_000_000,
            achieved_ns: 12_000_000
        }
    ));
}

#[test]
fn hard_constraint_rejection_over_jitter_and_memory_ceiling() {
    let objective = RealTimeObjective::hard_real_time(10_000_000, 500_000)
        .with_memory_ceiling(32 * 1024 * 1024);
    let device = DeviceFacts::unknown();

    // Schedule with excessive synchronization overhead (jitter > 500us)
    let mut cost_jitter = CostBreakdown::default();
    cost_jitter.total = 4_000_000;
    cost_jitter.barriers = 400;
    cost_jitter.grid_syncs = 50; // 400*1000 + 50*10000 = 900,000ns > 500,000ns
    let err = objective
        .satisfies_constraints(&cost_jitter, device)
        .expect_err("high-jitter schedule must be rejected");

    assert!(matches!(
        err,
        RealTimeViolation::JitterExceeded {
            limit_ns: 500_000,
            achieved_ns: 900_000
        }
    ));

    // Schedule with excessive memory footprint (> 32MB)
    let mut cost_mem = CostBreakdown::default();
    cost_mem.total = 4_000_000;
    cost_mem.planned_peak_bytes = 64 * 1024 * 1024;
    let err = objective
        .satisfies_constraints(&cost_mem, device)
        .expect_err("memory-heavy schedule must be rejected");

    assert!(matches!(
        err,
        RealTimeViolation::MemoryCeilingExceeded {
            limit_bytes: 33_554_432,
            achieved_bytes: 67_108_864
        }
    ));
}

#[test]
fn input_to_visible_measurement_boundary_and_statistics() {
    let raw_samples = vec![
        7_500_000, 7_600_000, 7_550_000, 7_700_000, 8_200_000, 7_450_000, 7_500_000, 9_100_000,
        7_520_000, 7_580_000,
    ];

    let measurement = InputToVisibleMeasurement::new(
        1_000_000_000, // Admitted at 1s
        200_000,       // 200us command encoding
        300_000,       // 300us queue wait
        100_000,       // 100us pipeline creation
        6_500_000,     // 6.5ms device kernel
        400_000,       // 400us synchronization
        100_000,       // 100us presentation handoff
        raw_samples,
        2,               // Queue depth 2
        Some(8_500_000), // 8.5ms deadline
    );

    assert_eq!(measurement.total_latency_ns, 7_600_000);
    assert_eq!(measurement.queue_depth_at_admission, 2);
    assert_eq!(measurement.missed_deadlines_count, 1); // 9.1ms sample missed 8.5ms deadline
    assert_eq!(measurement.worst_case_latency_ns(), 9_100_000);
    assert_eq!(measurement.jitter_ns(), 9_100_000 - 7_450_000);
    assert!(measurement.p99_latency_ns() >= 8_200_000);
}
