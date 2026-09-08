//! Cost modeling and multi-objective performance estimation for schedules.

use serde::{Deserialize, Serialize};
use super::tree::{ScheduleOp, SchedulePlan, ScheduleTree};

/// Detailed multi-dimensional cost record for a schedule plan.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ScheduleCostRecord {
    /// Estimated arithmetic intensity (FLOP / Byte).
    pub arithmetic_intensity: f64,
    /// Estimated global memory bandwidth pressure (Bytes/sec relative score).
    pub memory_traffic_bytes: u64,
    /// Estimated shared memory access latency penalty.
    pub shared_memory_latency_penalty: f64,
    /// Estimated register pressure score (0.0 = low, 1.0 = spill risk).
    pub register_pressure_score: f64,
    /// Estimated total latency in execution cycles.
    pub estimated_cycles: u64,
}

impl ScheduleCostRecord {
    /// Return true if this cost record Pareto-dominates `other` (i.e. is <= on all metrics and < on at least one).
    #[must_use]
    pub fn pareto_dominates(&self, other: &Self) -> bool {
        let le = self.memory_traffic_bytes <= other.memory_traffic_bytes
            && self.register_pressure_score <= other.register_pressure_score
            && self.estimated_cycles <= other.estimated_cycles;

        let lt = self.memory_traffic_bytes < other.memory_traffic_bytes
            || self.register_pressure_score < other.register_pressure_score
            || self.estimated_cycles < other.estimated_cycles;

        le && lt
    }
}

/// Cost evaluator for schedule plans.
#[derive(Clone, Debug, Default)]
pub struct ScheduleCostModel {
    /// Target warp size.
    pub warp_size: u32,
    /// Target max shared memory per workgroup.
    pub max_shared_memory: u64,
    /// Target max registers per thread.
    pub max_registers_per_thread: u32,
}

impl ScheduleCostModel {
    /// Create a new cost model with hardware parameters.
    #[must_use]
    pub fn new(warp_size: u32, max_shared_memory: u64, max_registers_per_thread: u32) -> Self {
        Self {
            warp_size,
            max_shared_memory,
            max_registers_per_thread,
        }
    }

    /// Evaluate a schedule plan and produce a multi-dimensional cost record.
    #[must_use]
    pub fn evaluate(&self, plan: &SchedulePlan) -> ScheduleCostRecord {
        let mut traffic = plan.resource_bounds.logical_points * 4; // base traffic estimate
        let mut shared_penalty = 0.0;
        let mut estimated_cycles = plan.resource_bounds.logical_points / (self.warp_size as u64).max(1);

        // Traverse tree to refine estimates
        self.evaluate_tree(&plan.root, &mut traffic, &mut shared_penalty, &mut estimated_cycles);

        let reg_score = if self.max_registers_per_thread > 0 {
            (plan.resource_bounds.registers_per_invocation as f64)
                / (self.max_registers_per_thread as f64)
        } else {
            0.5
        };

        ScheduleCostRecord {
            arithmetic_intensity: 2.0,
            memory_traffic_bytes: traffic,
            shared_memory_latency_penalty: shared_penalty,
            register_pressure_score: reg_score.clamp(0.0, 2.0),
            estimated_cycles: estimated_cycles.max(1),
        }
    }

    fn evaluate_tree(
        &self,
        tree: &ScheduleTree,
        traffic: &mut u64,
        shared_penalty: &mut f64,
        cycles: &mut u64,
    ) {
        match tree {
            ScheduleTree::Leaf(op) => self.evaluate_op(op, traffic, shared_penalty, cycles),
            ScheduleTree::Node { op, child, .. } => {
                self.evaluate_op(op, traffic, shared_penalty, cycles);
                self.evaluate_tree(child, traffic, shared_penalty, cycles);
            }
            ScheduleTree::Sequence(children) => {
                for c in children {
                    self.evaluate_tree(c, traffic, shared_penalty, cycles);
                }
            }
            ScheduleTree::Parallel(children) => {
                let mut max_c = 0;
                for c in children {
                    let mut sub_c = *cycles;
                    self.evaluate_tree(c, traffic, shared_penalty, &mut sub_c);
                    max_c = max_c.max(sub_c);
                }
                *cycles = max_c;
            }
        }
    }

    fn evaluate_op(
        &self,
        op: &ScheduleOp,
        traffic: &mut u64,
        shared_penalty: &mut f64,
        cycles: &mut u64,
    ) {
        match op {
            ScheduleOp::Tile { tile_size, .. } => {
                // Tiling improves reuse and reduces external memory traffic
                *traffic = (*traffic / (*tile_size).max(1)).max(1);
            }
            ScheduleOp::Unroll { factor, .. } => {
                // Unrolling reduces loop overhead cycles
                *cycles = (*cycles / (*factor as u64).max(1)).max(1);
            }
            ScheduleOp::Vectorize { vector_width, .. } => {
                // Vectorization reduces cycle count
                *cycles = (*cycles / (*vector_width as u64).max(1)).max(1);
            }
            ScheduleOp::SharedMemoryStage { staging_bytes, .. } => {
                if *staging_bytes > self.max_shared_memory {
                    *shared_penalty += 10.0;
                }
            }
            ScheduleOp::Synchronize { .. } => {
                // Synchronization adds a fixed cycle penalty
                *cycles += 16;
            }
            _ => {}
        }
    }
}
