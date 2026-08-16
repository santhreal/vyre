//! Contract tests for target emitted metrics, PMU observation, and candidate ranking.
//!
//! Verifies Section 185.5:
//! - Emitted instructions, register and local-memory use, shared-memory traffic,
//!   conflict and async-copy counters, output parity, and device duration.
//! - Automatic disqualification of candidates failing output parity.
//! - Empirical candidate ranking on specific target workloads (not portable semantic guarantees).

use vyre_lower::analyses::{
    rank_measured_candidates, CandidateRanking, TargetEmittedMetrics,
};

#[test]
fn parity_failure_disqualifies_candidate_immediately() {
    let fast_but_corrupted = CandidateRanking {
        candidate_id: "corrupted_fast_tile".to_string(),
        target_device: "cuda_rtx5090".to_string(),
        metrics: TargetEmittedMetrics {
            instruction_count: 50,
            register_count: 32,
            local_memory_spill_bytes: 0,
            shared_memory_bytes: 8192,
            bank_conflict_counter: 0,
            async_copy_counter: 16,
            output_parity_verified: false, // Parity failure!
            device_duration_nanos: 10_000,
        },
        score: 0.0,
        rank: 0,
        is_disqualified: false,
        disqualification_reason: None,
    };

    let slower_correct = CandidateRanking {
        candidate_id: "correct_baseline_tile".to_string(),
        target_device: "cuda_rtx5090".to_string(),
        metrics: TargetEmittedMetrics {
            instruction_count: 100,
            register_count: 32,
            local_memory_spill_bytes: 0,
            shared_memory_bytes: 8192,
            bank_conflict_counter: 0,
            async_copy_counter: 0,
            output_parity_verified: true, // Correct parity!
            device_duration_nanos: 25_000,
        },
        score: 0.0,
        rank: 0,
        is_disqualified: false,
        disqualification_reason: None,
    };

    let ranked = rank_measured_candidates(vec![fast_but_corrupted, slower_correct]);

    // The correct candidate is ranked #1; corrupted candidate is disqualified
    assert_eq!(ranked[0].candidate_id, "correct_baseline_tile");
    assert_eq!(ranked[0].rank, 1);
    assert!(!ranked[0].is_disqualified);

    assert_eq!(ranked[1].candidate_id, "corrupted_fast_tile");
    assert!(ranked[1].is_disqualified);
    assert_eq!(
        ranked[1].disqualification_reason.as_deref(),
        Some("output parity verification failed")
    );
}

#[test]
fn candidates_ranked_by_device_time_and_spill_penalty() {
    let fast_candidate = CandidateRanking {
        candidate_id: "swizzled_async_tile".to_string(),
        target_device: "cuda_rtx5090".to_string(),
        metrics: TargetEmittedMetrics {
            instruction_count: 80,
            register_count: 48,
            local_memory_spill_bytes: 0,
            shared_memory_bytes: 16384,
            bank_conflict_counter: 0,
            async_copy_counter: 32,
            output_parity_verified: true,
            device_duration_nanos: 15_000,
        },
        score: 0.0,
        rank: 0,
        is_disqualified: false,
        disqualification_reason: None,
    };

    let spilled_candidate = CandidateRanking {
        candidate_id: "over_unrolled_tile".to_string(),
        target_device: "cuda_rtx5090".to_string(),
        metrics: TargetEmittedMetrics {
            instruction_count: 200,
            register_count: 140,
            local_memory_spill_bytes: 1024, // Spills to local memory
            shared_memory_bytes: 16384,
            bank_conflict_counter: 128,
            async_copy_counter: 32,
            output_parity_verified: true,
            device_duration_nanos: 30_000,
        },
        score: 0.0,
        rank: 0,
        is_disqualified: false,
        disqualification_reason: None,
    };

    let ranked = rank_measured_candidates(vec![spilled_candidate, fast_candidate]);

    assert_eq!(ranked[0].candidate_id, "swizzled_async_tile");
    assert_eq!(ranked[0].rank, 1);
    assert_eq!(ranked[1].candidate_id, "over_unrolled_tile");
    assert_eq!(ranked[1].rank, 2);
    assert!(ranked[0].score < ranked[1].score);
}
