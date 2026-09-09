//! Workload envelopes and execution phase definitions.
//!
//! Owns representative runtime execution envelopes (Prefill vs. Decode),
//! context window boundaries, search budgets, and optimization objectives.

use serde::{Deserialize, Serialize};
use vyre::compiler::{CompileObjective, ObjectiveMetric, SearchBudget};

/// Autoregressive execution phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExecutionPhase {
    /// Initial prompt ingestion / prefill over prompt tokens.
    Prefill,
    /// Single-token or multi-token autoregressive generation step.
    Decode,
}

/// Workload envelope defining execution dimensions and compilation objectives.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkloadEnvelope {
    /// Execution phase (Prefill or Decode).
    pub phase: ExecutionPhase,
    /// Batch count (concurrent sequences).
    pub batch_size: u32,
    /// Active sequence length for this kernel launch (prompt length for prefill, 1 for single-token decode).
    pub sequence_len: u32,
    /// Prior context length cached in the Key/Value history (0 for prefill).
    pub context_len: u32,
    /// Maximum context capacity.
    pub max_seq_len: u32,
    /// Expected repeat launch count across serving lifetime.
    pub expected_launch_count: u32,
    /// Search budget allocated to schedule exploration.
    pub search_budget: SearchBudget,
    /// Optimization objective submitted to the compiler.
    pub objective: CompileObjective,
}

impl WorkloadEnvelope {
    /// Standard prompt prefill envelope.
    #[must_use]
    pub fn prefill(batch_size: u32, prompt_len: u32, max_seq_len: u32) -> Self {
        Self {
            phase: ExecutionPhase::Prefill,
            batch_size,
            sequence_len: prompt_len,
            context_len: 0,
            max_seq_len,
            expected_launch_count: 1_000,
            search_budget: SearchBudget::new(16, 1_000, 1, 0, 10_000_000),
            objective: CompileObjective::minimize_latency()
                .with_bound(ObjectiveMetric::ArtifactBytes, 128 * 1024 * 1024),
        }
    }

    /// Single-token autoregressive decode envelope.
    #[must_use]
    pub fn decode(batch_size: u32, context_len: u32, max_seq_len: u32) -> Self {
        Self {
            phase: ExecutionPhase::Decode,
            batch_size,
            sequence_len: 1,
            context_len,
            max_seq_len,
            expected_launch_count: 10_000,
            search_budget: SearchBudget::new(16, 1_000, 1, 0, 10_000_000),
            objective: CompileObjective::minimize_latency()
                .with_bound(ObjectiveMetric::ArtifactBytes, 128 * 1024 * 1024),
        }
    }

    /// Latency-critical serving envelope.
    #[must_use]
    pub fn latency_critical(phase: ExecutionPhase, batch_size: u32, seq_len: u32, max_seq_len: u32) -> Self {
        let (sequence_len, context_len) = match phase {
            ExecutionPhase::Prefill => (seq_len, 0),
            ExecutionPhase::Decode => (1, seq_len),
        };
        Self {
            phase,
            batch_size,
            sequence_len,
            context_len,
            max_seq_len,
            expected_launch_count: 50_000,
            search_budget: SearchBudget::new(64, 10_000, 4, 1, 100_000_000),
            objective: CompileObjective::minimize_latency()
                .with_bound(ObjectiveMetric::ArtifactBytes, 128 * 1024 * 1024),
        }
    }

    /// High-throughput batched serving envelope.
    #[must_use]
    pub fn throughput(phase: ExecutionPhase, batch_size: u32, seq_len: u32, max_seq_len: u32) -> Self {
        let (sequence_len, context_len) = match phase {
            ExecutionPhase::Prefill => (seq_len, 0),
            ExecutionPhase::Decode => (1, seq_len),
        };
        Self {
            phase,
            batch_size,
            sequence_len,
            context_len,
            max_seq_len,
            expected_launch_count: 100_000,
            search_budget: SearchBudget::new(32, 2_000, 2, 0, 20_000_000),
            objective: CompileObjective::maximize_throughput(100_000)
                .with_bound(ObjectiveMetric::ArtifactBytes, 128 * 1024 * 1024),
        }
    }
}
