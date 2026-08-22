//! Concrete asynchronous copy and ping-pong double-buffer scheduling for PTX targets.
//!
//! Extends ldmatrix and cp.async detection with target-local double buffering,
//! accumulator liveness tracking, wait-group depth control, and epilogue overlap.
//! Keeps register layout and instruction scheduling strictly concrete inside PTX emitter.

use super::AsyncCopyPlan;
use crate::ComputeCapability;
use serde::{Deserialize, Serialize};
use vyre_lower::KernelDescriptor;

/// Double-buffer / multi-stage pipeline configuration for PTX async copies.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DoubleBufferSchedule {
    /// Number of buffer stages (2 for ping-pong double buffering, 3+ for deep software pipelining).
    pub stages: u32,
    /// Shared-memory buffer slots assigned to the rotating stages.
    pub buffer_slots: Vec<u32>,
    /// Accumulator liveness bounds for matrix operations.
    pub accumulator_liveness: AccumulatorLiveness,
    /// Wait-group depth policy for overlapping async copies with MMA computation.
    pub wait_group_policy: WaitGroupPolicy,
    /// Epilogue store overlap configuration.
    pub epilogue_overlap: EpilogueOverlap,
}

impl Default for DoubleBufferSchedule {
    fn default() -> Self {
        Self {
            stages: 2,
            buffer_slots: vec![0, 1],
            accumulator_liveness: AccumulatorLiveness::default(),
            wait_group_policy: WaitGroupPolicy::default(),
            epilogue_overlap: EpilogueOverlap::default(),
        }
    }
}

/// Accumulator register liveness and pressure profile.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AccumulatorLiveness {
    /// Estimated live accumulator registers per thread.
    pub live_accumulator_registers: usize,
    /// Available register budget per thread before spilling.
    pub register_file_budget: usize,
    /// Whether register pressure causes local-memory spill under this schedule.
    pub causes_register_spill: bool,
}

impl Default for AccumulatorLiveness {
    fn default() -> Self {
        Self {
            live_accumulator_registers: 32,
            register_file_budget: 128,
            causes_register_spill: false,
        }
    }
}

/// Wait group depth policy for cp.async.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WaitGroupPolicy {
    /// Maximum number of in-flight commit groups allowed before blocking.
    pub max_in_flight_groups: u32,
    /// Wait group depth passed to `cp.async.wait_group N` inside the main loop.
    pub loop_wait_group_depth: u32,
    /// Drain policy at epilogue (`cp.async.wait_group 0` + `membar.cta`).
    pub drain_at_epilogue: bool,
}

impl Default for WaitGroupPolicy {
    fn default() -> Self {
        Self {
            max_in_flight_groups: 2,
            loop_wait_group_depth: 1,
            drain_at_epilogue: true,
        }
    }
}

/// Configuration for overlapping epilogue write-out with async transfers.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EpilogueOverlap {
    /// Allow epilogue global stores to overlap with next tile async prefetching.
    pub overlap_with_next_tile: bool,
    /// Number of epilogue stages decoupled from main loop barrier.
    pub epilogue_wait_depth: u32,
}

impl Default for EpilogueOverlap {
    fn default() -> Self {
        Self {
            overlap_with_next_tile: true,
            epilogue_wait_depth: 0,
        }
    }
}

/// Pipeline operation within the steady-state loop of a ping-pong schedule.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PipelineStageOp {
    /// Issue async global-to-shared copy into stage index.
    AsyncCopyIssue {
        /// Target buffer stage index.
        stage: u32,
        /// Destination shared memory slot.
        shared_slot: u32,
        /// Source global memory slot.
        global_slot: u32,
    },
    /// Commit currently staged async copies into a wait group.
    CommitGroup,
    /// Wait until `depth` older groups remain in flight.
    WaitGroup {
        /// Target in-flight wait-group depth.
        depth: u32,
    },
    /// Shared-memory block synchronization fence (`membar.cta` or `bar.sync`).
    BlockBarrier,
    /// Matrix load from shared memory to fragment registers (`ldmatrix`).
    LdMatrix {
        /// Source buffer stage index.
        stage: u32,
        /// Source shared memory slot.
        shared_slot: u32,
        /// Number of matrix fragment registers loaded.
        matrix_reg_count: u32,
    },
    /// Core compute MMA / FMA instruction.
    MmaCompute {
        /// Active compute stage index.
        stage: u32,
    },
    /// Advance stage pointer (ping-pong toggle).
    RotateStage,
}

/// Complete planned double-buffer ping-pong schedule for a kernel.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PingPongPlan {
    /// Kernel identifier.
    pub kernel_id: String,
    /// Target compute capability.
    pub target: ComputeCapability,
    /// Concrete schedule parameters.
    pub schedule: DoubleBufferSchedule,
    /// Prologue operations (initial prefetch).
    pub prologue: Vec<PipelineStageOp>,
    /// Steady-state loop body operations.
    pub steady_state: Vec<PipelineStageOp>,
    /// Epilogue drain and final write-out operations.
    pub epilogue: Vec<PipelineStageOp>,
}

/// Errors returned during schedule construction.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScheduleError {
    /// Target lacks hardware support for async copy (`cp.async` requires sm_80+).
    UnsupportedTarget {
        /// Target compute capability.
        target: ComputeCapability,
        /// Reason why target is unsupported.
        reason: &'static str,
    },
    /// Kernel has no eligible async copy candidates.
    NoEligibleCandidates,
    /// Insufficient shared-memory slots allocated for double-buffering.
    InsufficientBufferSlots {
        /// Required buffer slot count.
        required: usize,
        /// Actual buffer slot count available.
        actual: usize,
    },
    /// Excessive register pressure induces unacceptable spill.
    ExcessiveRegisterPressure {
        /// Required register count.
        required: usize,
        /// Register file limit.
        limit: usize,
    },
}

impl std::fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedTarget { target, reason } => {
                write!(
                    f,
                    "PTX target sm_{}{} does not support async copy: {reason}",
                    target.major, target.minor
                )
            }
            Self::NoEligibleCandidates => write!(f, "no eligible load/store pairs for async copy"),
            Self::InsufficientBufferSlots { required, actual } => {
                write!(
                    f,
                    "insufficient buffer slots: required {required}, available {actual}"
                )
            }
            Self::ExcessiveRegisterPressure { required, limit } => {
                write!(
                    f,
                    "register pressure ({required} registers) exceeds limit ({limit})"
                )
            }
        }
    }
}

impl std::error::Error for ScheduleError {}

/// Plan a concrete double-buffer ping-pong schedule for a kernel descriptor.
pub fn plan_double_buffer_schedule(
    desc: &KernelDescriptor,
    target: ComputeCapability,
    plan: &AsyncCopyPlan,
    buffer_slots: &[u32],
    register_budget: usize,
) -> Result<PingPongPlan, ScheduleError> {
    if !target.supports_async_copy() {
        return Err(ScheduleError::UnsupportedTarget {
            target,
            reason: "cp.async requires Ampere (sm_80+) or higher architecture",
        });
    }

    if plan.candidates.is_empty() {
        return Err(ScheduleError::NoEligibleCandidates);
    }

    if buffer_slots.len() < 2 {
        return Err(ScheduleError::InsufficientBufferSlots {
            required: 2,
            actual: buffer_slots.len(),
        });
    }

    // Accumulator liveness: matrix fragment tile accumulator requires ~32-64 registers
    let required_registers = 48;
    if required_registers > register_budget {
        return Err(ScheduleError::ExcessiveRegisterPressure {
            required: required_registers,
            limit: register_budget,
        });
    }

    let schedule = DoubleBufferSchedule {
        stages: 2,
        buffer_slots: buffer_slots.to_vec(),
        accumulator_liveness: AccumulatorLiveness {
            live_accumulator_registers: required_registers,
            register_file_budget: register_budget,
            causes_register_spill: false,
        },
        wait_group_policy: WaitGroupPolicy {
            max_in_flight_groups: 2,
            loop_wait_group_depth: 1,
            drain_at_epilogue: true,
        },
        epilogue_overlap: EpilogueOverlap {
            overlap_with_next_tile: true,
            epilogue_wait_depth: 0,
        },
    };

    let first_cand = &plan.candidates[0];

    // Prologue: Prefetch stage 0
    let prologue = vec![
        PipelineStageOp::AsyncCopyIssue {
            stage: 0,
            shared_slot: buffer_slots[0],
            global_slot: first_cand.global_binding_slot,
        },
        PipelineStageOp::CommitGroup,
    ];

    // Steady state: issue stage 1, wait for stage 0, compute stage 0, rotate
    let steady_state = vec![
        PipelineStageOp::AsyncCopyIssue {
            stage: 1,
            shared_slot: buffer_slots[1],
            global_slot: first_cand.global_binding_slot,
        },
        PipelineStageOp::CommitGroup,
        PipelineStageOp::WaitGroup { depth: 1 },
        PipelineStageOp::BlockBarrier,
        PipelineStageOp::LdMatrix {
            stage: 0,
            shared_slot: buffer_slots[0],
            matrix_reg_count: 16,
        },
        PipelineStageOp::MmaCompute { stage: 0 },
        PipelineStageOp::RotateStage,
    ];

    // Epilogue: drain remaining wait groups, finish computation
    let epilogue = vec![
        PipelineStageOp::WaitGroup { depth: 0 },
        PipelineStageOp::BlockBarrier,
        PipelineStageOp::LdMatrix {
            stage: 1,
            shared_slot: buffer_slots[1],
            matrix_reg_count: 16,
        },
        PipelineStageOp::MmaCompute { stage: 1 },
    ];

    Ok(PingPongPlan {
        kernel_id: desc.id.clone(),
        target,
        schedule,
        prologue,
        steady_state,
        epilogue,
    })
}
