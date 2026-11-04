//! D4 substrate: pre-recorded command reuse policy.
//!
//! When the same dispatch shape repeats (same Program, same binding
//! handles, same workgroup, same workload count), backends can record
//! the launch sequence once and replay it through their native command
//! reuse primitive. This eliminates per-launch driver API overhead.
//!
//! Pure decision: given a dispatch repetition count and the measured
//! per-launch overhead vs command-record overhead, should the
//! dispatcher record-and-replay or just launch normally?
//!
//! This sits next to D1 (persistent kernels). Persistent mode wins
//! for unpredictable batches of small kernels; command reuse wins for
//! REPEATED dispatches of the same shape.

/// Inputs to the command-reuse decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommandReuseInputs {
    /// Number of times this exact dispatch shape will be repeated
    /// (the same Program + bindings + workload count).
    pub repeat_count: u32,
    /// Per-launch driver API overhead in nanoseconds. Same number
    /// the persistent-kernel policy uses.
    pub per_launch_overhead_ns: u64,
    /// One-time cost of recording the native command sequence.
    pub record_overhead_ns: u64,
    /// Per-replay cost of the native command-reuse primitive.
    pub replay_overhead_ns: u64,
}

/// Verdict from [`decide_command_reuse`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandReuseDecision {
    /// Use plain dispatch  -  repeat count too low to amortise the
    /// command-record cost.
    PlainLaunches,
    /// Record once, replay `repeat_count - 1` more times. Includes
    /// the predicted savings vs plain launches for telemetry.
    RecordAndReplay {
        /// Predicted total time saved (in nanoseconds) vs plain
        /// launches. Positive by construction.
        savings_ns: u128,
    },
}

/// Decide whether to record a command sequence once and replay it for
/// the remaining `repeat_count - 1` dispatches.
///
/// Plain cost:    `repeat * per_launch_ovh`
/// Reuse cost:    `record_ovh + repeat * replay_ovh`
/// Reuse wins iff `repeat * (per_launch_ovh - replay_ovh) > record_ovh`.
#[must_use]
pub fn decide_command_reuse(inputs: CommandReuseInputs) -> CommandReuseDecision {
    if inputs.repeat_count <= 1 {
        return CommandReuseDecision::PlainLaunches;
    }
    if inputs.per_launch_overhead_ns <= inputs.replay_overhead_ns {
        // Replay is not actually cheaper than plain launch.
        // recording costs us bytes for nothing.
        return CommandReuseDecision::PlainLaunches;
    }
    let per_call_savings =
        u128::from(inputs.per_launch_overhead_ns) - u128::from(inputs.replay_overhead_ns);
    let total_call_savings = u128::from(inputs.repeat_count) * per_call_savings;
    let record_overhead_ns = u128::from(inputs.record_overhead_ns);
    if total_call_savings <= record_overhead_ns {
        return CommandReuseDecision::PlainLaunches;
    }
    let savings_ns = total_call_savings - record_overhead_ns;
    CommandReuseDecision::RecordAndReplay { savings_ns }
}
