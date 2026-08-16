//! PERF B9: PTX-level instruction scheduling hints.
//!
//! Modern NVIDIA GPUs reorder PTX instructions based on dependency
//! latencies. The driver does most of the work, but PTX exposes
//! `.pragma "nounroll"`, `__pipeline_depth`, and similar hints that
//! pin behavior when the compiler's reordering is suboptimal.
//!
//! This module computes a `SchedulingHints` for a kernel: detects
//! long dependency chains (where back-to-back instructions read what
//! the previous one wrote), reports them as latency-sensitive
//! sequences worth scheduling around.

use serde::{Deserialize, Serialize};
use vyre_lower::{KernelBody, KernelDescriptor};

/// One latency-sensitive operation dependency chain.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DependencyChain {
    /// Op-index where the chain starts.
    pub start_op_index: usize,
    /// Length of the chain (number of dependent ops).
    pub length: u32,
}

/// Instruction-scheduling findings for one kernel.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SchedulingHints {
    /// Stable kernel identifier.
    pub kernel_id: String,
    /// Long dependency chains. Each has `length ≥ 4`.
    pub long_chains: Vec<DependencyChain>,
    /// Total ops in the body (for context).
    pub total_op_count: u32,
}

impl SchedulingHints {
    /// Return the number of long dependency chains.
    #[must_use]
    pub fn long_chain_count(&self) -> usize {
        self.long_chains.len()
    }

    /// Return the longest dependency-chain length.
    #[must_use]
    pub fn longest_chain(&self) -> u32 {
        self.long_chains.iter().map(|c| c.length).max().unwrap_or(0)
    }

    /// Return a combined latency-pressure score.
    #[must_use]
    pub fn schedule_latency_pressure(&self) -> u32 {
        self.longest_chain()
            .saturating_mul(self.long_chain_count().min(u32::MAX as usize) as u32)
    }
}

/// Minimum operation count classified as a long dependency chain.
pub const LONG_CHAIN_THRESHOLD: u32 = 4;

/// Analyze descriptor dependencies for instruction-scheduling pressure.
#[must_use]
pub fn analyze(desc: &KernelDescriptor) -> SchedulingHints {
    let mut long_chains = Vec::new();
    detect_chains(&desc.body, &mut long_chains, 0);
    SchedulingHints {
        kernel_id: desc.id.clone(),
        long_chains,
        total_op_count: count_ops(&desc.body),
    }
}

fn count_ops(body: &KernelBody) -> u32 {
    let mut total: u32 = body.ops.len() as u32;
    for child in &body.child_bodies {
        total = total.saturating_add(count_ops(child));
    }
    total
}

fn detect_chains(body: &KernelBody, chains: &mut Vec<DependencyChain>, op_index_offset: usize) {
    for start in 0..body.ops.len() {
        let mut len: u32 = 1;
        let mut current_index = start;
        let mut prev_result = body.ops[start].result;
        while let Some(result) = prev_result {
            let Some(next_index) = first_later_consumer(body, result, current_index + 1) else {
                break;
            };
            len = len.saturating_add(1);
            current_index = next_index;
            prev_result = body.ops[next_index].result;
        }
        if len >= LONG_CHAIN_THRESHOLD {
            chains.push(DependencyChain {
                start_op_index: op_index_offset + start,
                length: len,
            });
        }
    }
    for child in &body.child_bodies {
        detect_chains(child, chains, op_index_offset + body.ops.len());
    }
}

fn first_later_consumer(body: &KernelBody, value: u32, start: usize) -> Option<usize> {
    body.ops
        .iter()
        .enumerate()
        .skip(start)
        .find_map(|(index, op)| op.operands.contains(&value).then_some(index))
}
