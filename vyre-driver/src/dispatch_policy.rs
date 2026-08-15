//! One-shot evaluation of every dispatch-side launch policy.
//!
//! A dispatcher needs all six decisions for every batch: persistent kernel
//! residency, arm independence, async copy overlap, command reuse, bindless
//! binding, and trace-JIT speculation. Calling six functions and threading six
//! verdicts through the dispatcher is boilerplate, so this module owns the
//! bundle: pass `DispatchPolicyInputs`, get back a `DispatchPolicyVerdict` with
//! every sub-decision already made.
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
use crate::persistent_kernel_policy::{
    decide_persistent_kernel, PersistentKernelDecision, PersistentKernelInputs,
};
use crate::trace_jit_policy::{decide_trace_jit_speculation, TraceJitDecision, TraceJitInputs};

/// Input bundle for a single dispatch-policy invocation.
///
/// Two arms (`arm_a`, `arm_b`) are needed for independence and overlap even
/// when only one is real: pass an empty `ArmBindingSummary::default()` for the
/// absent slot. `copy_dst_slot` is `None` when no host-to-device copy is queued
/// for this batch.
#[derive(Debug, Clone)]
pub struct DispatchPolicyInputs {
    /// Persistent-kernel residency inputs.
    pub persistent: PersistentKernelInputs,
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
    /// Persistent-kernel residency verdict.
    pub persistent: PersistentKernelDecision,
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
    /// D1 persistent kernels and D4 command reuse can both be profitable on
    /// paper. A concrete dispatcher cannot run both for the same launch group,
    /// so this resolver chooses the higher predicted savings. Equal savings
    /// prefer command reuse because it avoids persistent queue residency.
    #[must_use]
    pub fn primary_execution_mode(&self) -> DispatchExecutionMode {
        select_primary_execution_mode(self.persistent, self.command_reuse)
    }
}

/// One-shot evaluation of every dispatch-side policy substrate.
#[must_use]
pub fn evaluate_dispatch_policy(inputs: &DispatchPolicyInputs) -> DispatchPolicyVerdict {
    let persistent = decide_persistent_kernel(inputs.persistent);
    let arm_independence = can_dispatch_concurrently(&inputs.arm_a, &inputs.arm_b);
    let copy_overlap = inputs
        .copy_dst_slot
        .map(|slot| can_overlap_copy_with_kernel(slot, &inputs.arm_a));
    let command_reuse = decide_command_reuse(inputs.graph);
    let bindless = decide_bindless(inputs.bindless);
    let trace_jit = decide_trace_jit_speculation(inputs.trace_jit);
    record_policy_audit_events(persistent, command_reuse, bindless, trace_jit);
    DispatchPolicyVerdict {
        persistent,
        arm_independence,
        copy_overlap,
        command_reuse,
        bindless,
        trace_jit,
    }
}

/// Select a single primary launch strategy from D1 and D4 decisions.
#[must_use]
pub fn select_primary_execution_mode(
    persistent: PersistentKernelDecision,
    command_reuse: CommandReuseDecision,
) -> DispatchExecutionMode {
    match (persistent, command_reuse) {
        (
            PersistentKernelDecision::PersistentKernel {
                savings_ns: persistent_savings,
            },
            CommandReuseDecision::RecordAndReplay {
                savings_ns: command_savings,
            },
        ) => {
            if persistent_savings > command_savings {
                DispatchExecutionMode::PersistentKernel {
                    savings_ns: persistent_savings,
                }
            } else {
                DispatchExecutionMode::CommandReuse {
                    savings_ns: command_savings,
                }
            }
        }
        (
            PersistentKernelDecision::PersistentKernel { savings_ns },
            CommandReuseDecision::PlainLaunches,
        ) => DispatchExecutionMode::PersistentKernel { savings_ns },
        (
            PersistentKernelDecision::StandardLaunches,
            CommandReuseDecision::RecordAndReplay { savings_ns },
        ) => DispatchExecutionMode::CommandReuse { savings_ns },
        (PersistentKernelDecision::StandardLaunches, CommandReuseDecision::PlainLaunches) => {
            DispatchExecutionMode::PlainLaunches
        }
    }
}

fn record_policy_audit_events(
    persistent: PersistentKernelDecision,
    command_reuse: CommandReuseDecision,
    bindless: BindlessDecision,
    trace_jit: TraceJitDecision,
) {
    record_policy_audit_events_with(
        persistent,
        command_reuse,
        bindless,
        trace_jit,
        record_substrate_audit_event,
    );
}

fn record_policy_audit_events_with(
    persistent: PersistentKernelDecision,
    command_reuse: CommandReuseDecision,
    bindless: BindlessDecision,
    trace_jit: TraceJitDecision,
    mut record: impl FnMut(SubstrateAuditEvent),
) {
    if let PersistentKernelDecision::PersistentKernel { savings_ns } = persistent {
        record(SubstrateAuditEvent {
            substrate: "persistent_kernel",
            action: "queue_batch",
            saved_ns: savings_ns,
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
            persistent: PersistentKernelInputs {
                batch_size: 500,
                per_launch_overhead_ns: 5_000,
                per_item_kernel_ns: 1_000,
                persistent_setup_overhead_ns: 50_000,
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
            persistent: PersistentKernelInputs {
                batch_size: 1,
                per_launch_overhead_ns: 5_000,
                per_item_kernel_ns: 1_000,
                persistent_setup_overhead_ns: 50_000,
            },
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
            verdict.persistent,
            decide_persistent_kernel(inputs.persistent),
            "Fix: the bundle must report the persistent-kernel policy's own verdict."
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
        assert!(matches!(
            v.persistent,
            PersistentKernelDecision::PersistentKernel { .. }
        ));
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
            v.persistent,
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
        assert_eq!(v.persistent, PersistentKernelDecision::StandardLaunches);
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
            PersistentKernelDecision::PersistentKernel { savings_ns: 100 },
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
                PersistentKernelDecision::PersistentKernel { savings_ns: 500 },
                CommandReuseDecision::PlainLaunches,
            ),
            DispatchExecutionMode::PersistentKernel { savings_ns: 500 }
        );
        assert_eq!(
            select_primary_execution_mode(
                PersistentKernelDecision::StandardLaunches,
                CommandReuseDecision::RecordAndReplay { savings_ns: 700 },
            ),
            DispatchExecutionMode::CommandReuse { savings_ns: 700 }
        );
    }

    /// Saturating arithmetic in every policy, proved by the verdict it reaches.
    ///
    /// Extreme inputs used to be discarded with `let _`, which caught a panic
    /// and nothing else: a policy that overflowed into a small number passed.
    /// These assert the saturated answer instead.
    #[test]
    fn extreme_inputs_saturate_instead_of_overflowing() {
        assert!(
            matches!(
                decide_persistent_kernel(PersistentKernelInputs {
                    batch_size: u32::MAX,
                    per_launch_overhead_ns: u64::MAX / 2,
                    per_item_kernel_ns: u64::MAX / 2,
                    persistent_setup_overhead_ns: u64::MAX / 4,
                }),
                PersistentKernelDecision::PersistentKernel { .. }
            ),
            "Fix: a saturated launch-overhead total must still favour persistent residency."
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
