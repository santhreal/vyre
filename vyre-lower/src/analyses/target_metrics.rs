//! Emitted target instruction metrics, PMU observations, and candidate ranking.
//!
//! Records emitted instructions, register and local-memory use, shared-memory
//! traffic, conflict and async-copy counters, output parity verification, and
//! measured device execution duration.
//!
//! PMU observations rank candidates for the specific measured workload; they are
//! empirical observations rather than portable semantic guarantees.

use serde::{Deserialize, Serialize};

/// Emitted instruction, register, and hardware PMU metrics for a lowered candidate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TargetEmittedMetrics {
    /// Number of emitted target assembly / intermediate instructions.
    pub instruction_count: usize,
    /// Number of physical registers allocated per thread.
    pub register_count: usize,
    /// Local-memory spill in bytes (0 if all values stay in registers).
    pub local_memory_spill_bytes: usize,
    /// Shared-memory allocation in bytes.
    pub shared_memory_bytes: usize,
    /// Hardware bank-conflict counter (from PMU or precise cycle simulation).
    pub bank_conflict_counter: u64,
    /// Asynchronous-copy operations issued.
    pub async_copy_counter: u64,
    /// Whether numerical output parity was verified against reference execution.
    pub output_parity_verified: bool,
    /// Measured execution duration in nanoseconds on physical device.
    pub device_duration_nanos: u64,
}

impl Default for TargetEmittedMetrics {
    fn default() -> Self {
        Self {
            instruction_count: 0,
            register_count: 32,
            local_memory_spill_bytes: 0,
            shared_memory_bytes: 0,
            bank_conflict_counter: 0,
            async_copy_counter: 0,
            output_parity_verified: true,
            device_duration_nanos: 0,
        }
    }
}

/// Ranked optimization candidate evaluated on the owning target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateRanking {
    /// Unique candidate identifier.
    pub candidate_id: String,
    /// Target device identifier.
    pub target_device: String,
    /// Emitted target metrics and PMU counters.
    pub metrics: TargetEmittedMetrics,
    /// Computed composite performance score (lower is faster/better).
    pub score: f64,
    /// Ordinal ranking among evaluated candidates (1 = best).
    pub rank: usize,
    /// Whether this candidate is disqualified (e.g. parity failure or fatal spill).
    pub is_disqualified: bool,
    /// Reason for disqualification if any.
    pub disqualification_reason: Option<String>,
}

/// Rank measured candidates for a specific workload on an owning target.
///
/// Output parity is a mandatory requirement: candidates failing parity verification
/// are automatically disqualified regardless of raw speed.
///
/// NOTE: Resulting rankings represent empirical PMU observations for the measured
/// workload and device, not portable cross-architecture guarantees.
#[must_use]
pub fn rank_measured_candidates(
    mut candidates: Vec<CandidateRanking>,
) -> Vec<CandidateRanking> {
    for candidate in &mut candidates {
        if !candidate.metrics.output_parity_verified {
            candidate.is_disqualified = true;
            candidate.disqualification_reason = Some("output parity verification failed".to_string());
            candidate.score = f64::INFINITY;
            continue;
        }

        if candidate.metrics.local_memory_spill_bytes > 0 {
            // Heavy penalty for register spilling to local memory
            let spill_penalty = (candidate.metrics.local_memory_spill_bytes as f64) * 100.0;
            candidate.score = (candidate.metrics.device_duration_nanos as f64) + spill_penalty;
        } else {
            // Score primarily by device duration, with minor weighting on instruction & conflict counters
            let conflict_penalty = (candidate.metrics.bank_conflict_counter as f64) * 10.0;
            candidate.score = (candidate.metrics.device_duration_nanos as f64) + conflict_penalty;
        }
    }

    // Sort valid candidates by score ascending, disqualified candidates to the end
    candidates.sort_by(|a, b| {
        a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal)
    });

    for (index, candidate) in candidates.iter_mut().enumerate() {
        candidate.rank = index + 1;
    }

    candidates
}
