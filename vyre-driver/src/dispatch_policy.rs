//! One-shot evaluation of every dispatch-side launch policy.
//!
//! A dispatcher needs five decisions for every batch: arm independence, async
//! copy overlap, command reuse, bindless binding, and trace-JIT speculation.
//! Calling five functions and threading five verdicts through the dispatcher is
//! boilerplate, so this module owns the bundle: pass `DispatchPolicyInputs`, get
//! back a `DispatchPolicyVerdict` with every sub-decision already made.
//!
//! Residency is not one of them. The compiler decides whether a program runs as
//! a persistent kernel while it can still see the whole graph, the device facts
//! and the measured launch costs, and records that decision in the artifact as
//! `vyre_megakernel::ExecutionMode`. The dispatcher carries the decision it was
//! given: it may decline a persistent artifact in favour of command reuse, and it
//! can never promote a `Static` artifact to persistent residency.
//!
//! The bundle is pure composition. It adds no policy of its own beyond
//! resolving the one conflict two profitable strategies can create, which
//! `select_primary_execution_mode` owns. `bundle_equals_the_policies_it_composes`
//! holds it to that.

use crate::arm_independence::{
    can_dispatch_concurrently, ArmBindingSummary, ArmIndependenceVerdict,
};
use crate::async_copy_overlap::{can_overlap_copy_with_kernel, CopyOverlapDecision};
use crate::bindless_policy::{decide_bindless, BindlessDecision, BindlessInputs};
use crate::command_reuse_policy::{decide_command_reuse, CommandReuseDecision, CommandReuseInputs};
use crate::observability::{record_substrate_audit_event, SubstrateAuditEvent};
use crate::trace_jit_policy::{decide_trace_jit_speculation, TraceJitDecision, TraceJitInputs};
use vyre_megakernel::ExecutionMode;

/// Input bundle for a single dispatch-policy invocation.
///
/// Two arms (`arm_a`, `arm_b`) are needed for independence and overlap even
/// when only one is real: pass an empty `ArmBindingSummary::default()` for the
/// absent slot. `copy_dst_slot` is `None` when no host-to-device copy is queued
/// for this batch.
#[derive(Debug, Clone)]
pub struct DispatchPolicyInputs {
    /// Execution mode the compiled artifact selected for this program.
    pub execution: ExecutionMode,
    /// First arm of the independence pair, and the kernel side of the copy.
    pub arm_a: ArmBindingSummary,
    /// Second arm of the independence pair.
    pub arm_b: ArmBindingSummary,
    /// Copy destination slot, or `None` when no host-to-device copy is queued.
    pub copy_dst_slot: Option<u32>,
    /// Command-reuse inputs.
    pub graph: CommandReuseInputs,
    /// Bindless-binding inputs.
    pub bindless: BindlessInputs,
    /// Trace-JIT speculation inputs.
    pub trace_jit: TraceJitInputs,
}

/// Result bundle from a single dispatch-policy invocation. Every
/// sub-substrate verdict appears in its typed form.
#[derive(Debug, Clone)]
pub struct DispatchPolicyVerdict {
    /// Execution mode the compiled artifact selected, carried through unchanged.
    pub execution: ExecutionMode,
    /// Independence verdict for the (`arm_a`, `arm_b`) pair.
    pub arm_independence: ArmIndependenceVerdict,
    /// `None` when the inputs had no `copy_dst_slot`; otherwise the overlap
    /// verdict for that copy.
    pub copy_overlap: Option<CopyOverlapDecision>,
    /// Command-reuse verdict.
    pub command_reuse: CommandReuseDecision,
    /// Bindless-binding verdict.
    pub bindless: BindlessDecision,
    /// Trace-JIT speculation verdict.
    pub trace_jit: TraceJitDecision,
}

/// Mutually exclusive launch strategy selected from the dispatch-policy bundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchExecutionMode {
    /// Plain launches remain cheapest for this batch.
    PlainLaunches,
    /// Use persistent kernel mode.
    PersistentKernel {
        /// Predicted saved nanoseconds versus plain launches.
        savings_ns: u128,
    },
    /// Use native command record/replay.
    CommandReuse {
        /// Predicted saved nanoseconds versus plain launches.
        savings_ns: u128,
    },
}

impl DispatchPolicyVerdict {
    /// Return the mutually exclusive primary launch strategy.
    ///
    /// A persistent artifact and D4 command reuse can both be profitable on
    /// paper. A concrete dispatcher cannot run both for the same launch group,
    /// so this resolver chooses the higher predicted saving. Equal savings prefer
    /// command reuse because it avoids persistent queue residency.
    #[must_use]
    pub fn primary_execution_mode(&self) -> DispatchExecutionMode {
        select_primary_execution_mode(self.execution, self.command_reuse)
    }
}

/// One-shot evaluation of every dispatch-side policy substrate.
#[must_use]
pub fn evaluate_dispatch_policy(inputs: &DispatchPolicyInputs) -> DispatchPolicyVerdict {
    let execution = inputs.execution;
    let arm_independence = can_dispatch_concurrently(&inputs.arm_a, &inputs.arm_b);
    let copy_overlap = inputs
        .copy_dst_slot
        .map(|slot| can_overlap_copy_with_kernel(slot, &inputs.arm_a));
    let command_reuse = decide_command_reuse(inputs.graph);
    let bindless = decide_bindless(inputs.bindless);
    let trace_jit = decide_trace_jit_speculation(inputs.trace_jit);
    record_policy_audit_events(execution, command_reuse, bindless, trace_jit);
    DispatchPolicyVerdict {
        execution,
        arm_independence,
        copy_overlap,
        command_reuse,
        bindless,
        trace_jit,
    }
}

/// Select a single primary launch strategy from the artifact's execution mode
/// and the D4 command-reuse decision.
///
/// `ExecutionMode::Static` never yields `DispatchExecutionMode::PersistentKernel`.
/// Residency needs a cooperative launch and a program the compiler shaped for it,
/// so the dispatcher cannot decide from a batch shape what the compiler declined.
#[must_use]
pub fn select_primary_execution_mode(
    execution: ExecutionMode,
    command_reuse: CommandReuseDecision,
) -> DispatchExecutionMode {
    match (execution, command_reuse) {
        (
            ExecutionMode::Persistent {
                saved_ns: persistent_savings,
            },
            CommandReuseDecision::RecordAndReplay {
                savings_ns: command_savings,
            },
        ) => {
            if u128::from(persistent_savings) > command_savings {
                DispatchExecutionMode::PersistentKernel {
                    savings_ns: u128::from(persistent_savings),
                }
            } else {
                DispatchExecutionMode::CommandReuse {
                    savings_ns: command_savings,
                }
            }
        }
        (ExecutionMode::Persistent { saved_ns }, CommandReuseDecision::PlainLaunches) => {
            DispatchExecutionMode::PersistentKernel {
                savings_ns: u128::from(saved_ns),
            }
        }
        (ExecutionMode::Static, CommandReuseDecision::RecordAndReplay { savings_ns }) => {
            DispatchExecutionMode::CommandReuse { savings_ns }
        }
        (ExecutionMode::Static, CommandReuseDecision::PlainLaunches) => {
            DispatchExecutionMode::PlainLaunches
        }
    }
}

fn record_policy_audit_events(
    execution: ExecutionMode,
    command_reuse: CommandReuseDecision,
    bindless: BindlessDecision,
    trace_jit: TraceJitDecision,
) {
    record_policy_audit_events_with(
        execution,
        command_reuse,
        bindless,
        trace_jit,
        record_substrate_audit_event,
    );
}

fn record_policy_audit_events_with(
    execution: ExecutionMode,
    command_reuse: CommandReuseDecision,
    bindless: BindlessDecision,
    trace_jit: TraceJitDecision,
    mut record: impl FnMut(SubstrateAuditEvent),
) {
    if let ExecutionMode::Persistent { saved_ns } = execution {
        record(SubstrateAuditEvent {
            substrate: "persistent_kernel",
            action: "queue_batch",
            saved_ns: u128::from(saved_ns),
            detail: "launch_overhead",
        });
    }
    if let CommandReuseDecision::RecordAndReplay { savings_ns } = command_reuse {
        record(SubstrateAuditEvent {
            substrate: "command_reuse",
            action: "record_and_replay",
            saved_ns: savings_ns,
            detail: "repeat_shape",
        });
    }
    if bindless == BindlessDecision::Bindless {
        record(SubstrateAuditEvent {
            substrate: "bindless",
            action: "descriptor_array",
            saved_ns: 0,
            detail: "resource_count_threshold",
        });
    }
    if let TraceJitDecision::Speculate {
        expected_savings_ns,
    } = trace_jit
    {
        record(SubstrateAuditEvent {
            substrate: "trace_jit",
            action: "speculate",
            saved_ns: expected_savings_ns,
            detail: "predicted_shape",
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bindless_policy::BindlessSupport;

    fn arm(reads: &[u32], writes: &[u32]) -> ArmBindingSummary {
        ArmBindingSummary {
            reads: reads.iter().copied().collect(),
            writes: writes.iter().copied().collect(),
        }
    }

    /// Large batch of small kernels, disjoint arms, a copy onto a slot neither
    /// arm touches, a repeated pipeline, many resources, and a hot shape. Every
    /// policy has a profitable answer for this workload.
    fn aggressive_inputs() -> DispatchPolicyInputs {
        DispatchPolicyInputs {
            execution: ExecutionMode::Persistent {
                saved_ns: 2_450_000,
            },
            arm_a: arm(&[0, 1], &[2]),
            arm_b: arm(&[3, 4], &[5]),
            copy_dst_slot: Some(7),
            graph: CommandReuseInputs {
                repeat_count: 500,
                per_launch_overhead_ns: 5_000,
                record_overhead_ns: 25_000,
                replay_overhead_ns: 500,
            },
            bindless: BindlessInputs {
                resource_count: 40,
                support: BindlessSupport::Full,
                dynamic_indexing: true,
            },
            trace_jit: TraceJitInputs {
                shader_hit_count: 100,
                prediction_confidence_bps: 9_000,
                speculative_spec_cost_ns: 10_000,
                miss_cost_ns: 100_000,
            },
        }
    }

    /// A single dispatch, arms that conflict on slot 5, a copy onto a slot
    /// `arm_a` reads, no repetition, few resources, and a cold shape. Every
    /// policy has to decline.
    fn conservative_inputs() -> DispatchPolicyInputs {
        DispatchPolicyInputs {
            execution: ExecutionMode::Static,
            arm_a: arm(&[5], &[1]),
            arm_b: arm(&[0], &[5]),
            copy_dst_slot: Some(5),
            graph: CommandReuseInputs {
                repeat_count: 1,
                per_launch_overhead_ns: 5_000,
                record_overhead_ns: 25_000,
                replay_overhead_ns: 500,
            },
            bindless: BindlessInputs {
                resource_count: 4,
                support: BindlessSupport::Full,
                dynamic_indexing: false,
            },
            trace_jit: TraceJitInputs {
                shader_hit_count: 2,
                prediction_confidence_bps: 9_000,
                speculative_spec_cost_ns: 10_000,
                miss_cost_ns: 100_000,
            },
        }
    }

    /// Every verdict field equals the policy called on its own.
    ///
    /// The bundle claims to add no logic. A second implementation inside it
    /// would satisfy the per-workload expectations below while disagreeing with
    /// the policy a caller reaches directly, which is the failure nothing else
    /// in this crate observes.
    fn assert_bundle_equals_parts(inputs: &DispatchPolicyInputs) {
        let verdict = evaluate_dispatch_policy(inputs);
        assert_eq!(
            verdict.execution,
            inputs.execution,
            "Fix: the bundle must carry the artifact's execution mode unchanged."
        );
        assert_eq!(
            verdict.arm_independence,
            can_dispatch_concurrently(&inputs.arm_a, &inputs.arm_b),
            "Fix: the bundle must report the arm-independence policy's own verdict."
        );
        assert_eq!(
            verdict.copy_overlap,
            inputs
                .copy_dst_slot
                .map(|slot| can_overlap_copy_with_kernel(slot, &inputs.arm_a)),
            "Fix: the bundle must report the copy-overlap policy's own verdict, and None when no copy is queued."
        );
        assert_eq!(
            verdict.command_reuse,
            decide_command_reuse(inputs.graph),
            "Fix: the bundle must report the command-reuse policy's own verdict."
        );
        assert_eq!(
            verdict.bindless,
            decide_bindless(inputs.bindless),
            "Fix: the bundle must report the bindless policy's own verdict."
        );
        assert_eq!(
            verdict.trace_jit,
            decide_trace_jit_speculation(inputs.trace_jit),
            "Fix: the bundle must report the trace-JIT policy's own verdict."
        );
    }

    #[test]
    fn bundle_equals_the_policies_it_composes() {
        assert_bundle_equals_parts(&aggressive_inputs());
        assert_bundle_equals_parts(&conservative_inputs());
        let mut without_copy = aggressive_inputs();
        without_copy.copy_dst_slot = None;
        assert_bundle_equals_parts(&without_copy);
    }

    #[test]
    fn aggressive_workload_routes_through_every_aggressive_path() {
        let _guard = crate::observability::audit_events_test_lock();
        crate::observability::clear_substrate_audit_events_for_test();
        let v = evaluate_dispatch_policy(&aggressive_inputs());
        assert!(matches!(v.execution, ExecutionMode::Persistent { .. }));
        assert_eq!(v.arm_independence, ArmIndependenceVerdict::Independent);
        assert_eq!(v.copy_overlap, Some(CopyOverlapDecision::Overlap));
        assert!(matches!(
            v.command_reuse,
            CommandReuseDecision::RecordAndReplay { .. }
        ));
        assert_eq!(v.bindless, BindlessDecision::Bindless);
        assert!(matches!(v.trace_jit, TraceJitDecision::Speculate { .. }));
        assert_eq!(
            v.primary_execution_mode(),
            DispatchExecutionMode::PersistentKernel {
                savings_ns: 2_450_000
            }
        );
        record_policy_audit_events_with(
            v.execution,
            v.command_reuse,
            v.bindless,
            v.trace_jit,
            crate::observability::record_substrate_audit_event_for_test,
        );
        let log = crate::observability::snapshot_for_test().to_audit_log();
        assert!(log.contains("persistent_kernel queue_batch"));
        assert!(log.contains("command_reuse record_and_replay"));
        assert!(log.contains("bindless descriptor_array"));
        assert!(log.contains("trace_jit speculate"));
        crate::observability::clear_substrate_audit_events_for_test();
    }

    #[test]
    fn conservative_workload_routes_through_every_conservative_path() {
        let v = evaluate_dispatch_policy(&conservative_inputs());
        assert_eq!(v.execution, ExecutionMode::Static);
        assert!(matches!(
            v.arm_independence,
            ArmIndependenceVerdict::SerializeRequired { .. }
        ));
        assert_eq!(v.copy_overlap, Some(CopyOverlapDecision::Serialize));
        assert_eq!(v.command_reuse, CommandReuseDecision::PlainLaunches);
        assert_eq!(v.bindless, BindlessDecision::TraditionalBindings);
        assert_eq!(v.trace_jit, TraceJitDecision::HoldSteady);
        assert_eq!(
            v.primary_execution_mode(),
            DispatchExecutionMode::PlainLaunches
        );
    }

    #[test]
    fn missing_copy_slot_reports_none_for_overlap() {
        let mut inputs = aggressive_inputs();
        inputs.copy_dst_slot = None;
        assert_eq!(
            evaluate_dispatch_policy(&inputs).copy_overlap,
            None,
            "Fix: with no copy queued the bundle must report None instead of fabricating a verdict."
        );
    }

    #[test]
    fn primary_execution_mode_prefers_command_reuse_on_equal_savings() {
        let mode = select_primary_execution_mode(
            ExecutionMode::Persistent { saved_ns: 100 },
            CommandReuseDecision::RecordAndReplay { savings_ns: 100 },
        );
        assert_eq!(
            mode,
            DispatchExecutionMode::CommandReuse { savings_ns: 100 }
        );
    }

    #[test]
    fn primary_execution_mode_selects_only_profitable_substrate() {
        assert_eq!(
            select_primary_execution_mode(
                ExecutionMode::Persistent { saved_ns: 500 },
                CommandReuseDecision::PlainLaunches,
            ),
            DispatchExecutionMode::PersistentKernel { savings_ns: 500 }
        );
        assert_eq!(
            select_primary_execution_mode(
                ExecutionMode::Static,
                CommandReuseDecision::RecordAndReplay { savings_ns: 700 },
            ),
            DispatchExecutionMode::CommandReuse { savings_ns: 700 }
        );
    }

    /// WHY: 150.14. Residency is the compiler's decision, recorded in the
    /// artifact. The dispatcher used to decide it from a batch shape, so a batch
    /// of many cheap launches queued a persistent kernel for a program the
    /// compiler never shaped for one. No batch shape and no command-reuse verdict
    /// may promote a `Static` artifact.
    ///
    /// The variant space is the `ExecutionMode` union crossed with the
    /// `CommandReuseDecision` union, both enumerated here rather than sampled: a
    /// new `ExecutionMode` variant fails to compile this match until someone
    /// records what the dispatcher does with it.
    #[test]
    fn a_static_artifact_is_never_promoted_to_persistent_residency() {
        for command_reuse in [
            CommandReuseDecision::PlainLaunches,
            CommandReuseDecision::RecordAndReplay { savings_ns: 1 },
            CommandReuseDecision::RecordAndReplay {
                savings_ns: u128::MAX,
            },
        ] {
            let mode = select_primary_execution_mode(ExecutionMode::Static, command_reuse);
            assert!(
                !matches!(mode, DispatchExecutionMode::PersistentKernel { .. }),
                "Fix: a Static artifact must never dispatch as a persistent kernel, got {mode:?} for {command_reuse:?}."
            );
            let expected = match command_reuse {
                CommandReuseDecision::PlainLaunches => DispatchExecutionMode::PlainLaunches,
                CommandReuseDecision::RecordAndReplay { savings_ns } => {
                    DispatchExecutionMode::CommandReuse { savings_ns }
                }
            };
            assert_eq!(mode, expected);
        }
        let mut inputs = aggressive_inputs();
        inputs.execution = ExecutionMode::Static;
        let verdict = evaluate_dispatch_policy(&inputs);
        assert_eq!(
            verdict.execution,
            ExecutionMode::Static,
            "Fix: the bundle must not rewrite the artifact's execution mode."
        );
        assert!(
            matches!(
                verdict.primary_execution_mode(),
                DispatchExecutionMode::CommandReuse { .. }
            ),
            "Fix: the aggressive batch shape must reach command reuse, never residency, for a Static artifact."
        );
    }

    /// WHY: 150.14. A persistent artifact keeps its own predicted saving, and the
    /// audit event reports that number rather than a batch-shape estimate.
    #[test]
    fn a_persistent_artifact_reports_its_own_saving() {
        let mut recorded = Vec::new();
        record_policy_audit_events_with(
            ExecutionMode::Persistent { saved_ns: 4_096 },
            CommandReuseDecision::PlainLaunches,
            BindlessDecision::TraditionalBindings,
            TraceJitDecision::HoldSteady,
            |event| recorded.push(event),
        );
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0].substrate, "persistent_kernel");
        assert_eq!(recorded[0].saved_ns, 4_096);
        let mut none = Vec::new();
        record_policy_audit_events_with(
            ExecutionMode::Static,
            CommandReuseDecision::PlainLaunches,
            BindlessDecision::TraditionalBindings,
            TraceJitDecision::HoldSteady,
            |event| none.push(event),
        );
        assert!(
            none.is_empty(),
            "Fix: a Static artifact must record no residency event."
        );
    }

    /// Saturating arithmetic in every policy, proved by the verdict it reaches.
    ///
    /// Extreme inputs used to be discarded with `let _`, which caught a panic
    /// and nothing else: a policy that overflowed into a small number passed.
    /// These assert the saturated answer instead.
    #[test]
    fn extreme_inputs_saturate_instead_of_overflowing() {
        assert_eq!(
            select_primary_execution_mode(
                ExecutionMode::Persistent { saved_ns: u64::MAX },
                CommandReuseDecision::RecordAndReplay {
                    savings_ns: u128::from(u64::MAX),
                },
            ),
            DispatchExecutionMode::CommandReuse {
                savings_ns: u128::from(u64::MAX)
            },
            "Fix: a saturated artifact saving must not widen into a larger number than command reuse reports."
        );
        assert!(
            matches!(
                decide_command_reuse(CommandReuseInputs {
                    repeat_count: u32::MAX,
                    per_launch_overhead_ns: u64::MAX / 2,
                    record_overhead_ns: 1,
                    replay_overhead_ns: 1,
                }),
                CommandReuseDecision::RecordAndReplay { .. }
            ),
            "Fix: a saturated repeat total must still favour record and replay."
        );
        assert!(
            matches!(
                decide_trace_jit_speculation(TraceJitInputs {
                    shader_hit_count: u32::MAX,
                    prediction_confidence_bps: 10_000,
                    speculative_spec_cost_ns: 1,
                    miss_cost_ns: u64::MAX,
                }),
                TraceJitDecision::Speculate { .. }
            ),
            "Fix: a certain prediction against a saturated miss cost must still speculate."
        );
        assert_eq!(
            decide_bindless(BindlessInputs {
                resource_count: u32::MAX,
                support: BindlessSupport::Full,
                dynamic_indexing: true,
            }),
            BindlessDecision::Bindless,
            "Fix: a saturated resource count must still clear the bindless threshold."
        );

        let wide = arm(
            &(0..1000).collect::<Vec<u32>>(),
            &(1000..2000).collect::<Vec<u32>>(),
        );
        let disjoint = arm(
            &(2000..3000).collect::<Vec<u32>>(),
            &(3000..4000).collect::<Vec<u32>>(),
        );
        assert_eq!(
            can_dispatch_concurrently(&wide, &disjoint),
            ArmIndependenceVerdict::Independent,
            "Fix: a thousand disjoint slots per arm must still read as independent."
        );
        assert_eq!(
            can_overlap_copy_with_kernel(u32::MAX, &wide),
            CopyOverlapDecision::Overlap,
            "Fix: a copy onto a slot outside every declared range must still overlap."
        );
    }
}
