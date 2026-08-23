//! Canonical physical-kernel lowering boundary.
//!
//! This is the only production boundary from high-level `Program` IR to a
//! validated `PhysicalKernel`: expand registered compositions, run the
//! registered fallible semantic optimizer once, reject unresolved calls, lower
//! to a `KernelDescriptor`, canonicalize representation order, and verify.

use crate::descriptor::KernelDescriptor;
use crate::lower::lower;
use crate::{verify_descriptor, VerifyFailure};
use std::fmt;
use vyre_foundation::{
    ir::Program,
    schedule::{SchedulePhaseId, SelectedSchedule},
};

/// Verified physical kernel IR. Construction is restricted to
/// [`lower_physical`], so an unverified descriptor cannot enter target
/// compilation.
#[derive(Debug, Clone)]
pub struct PhysicalKernel {
    descriptor: KernelDescriptor,
}

impl PhysicalKernel {
    /// Borrow the verified backend-neutral descriptor.
    #[must_use]
    pub const fn descriptor(&self) -> &KernelDescriptor {
        &self.descriptor
    }

    /// Consume the physical stage and return its verified descriptor.
    #[must_use]
    pub fn into_descriptor(self) -> KernelDescriptor {
        self.descriptor
    }
}

/// Canonical semantic program and its verified physical kernel.
#[derive(Debug, Clone)]
pub struct PhysicalLowering {
    /// Program after semantic optimization.
    pub program: Program,
    /// Verified physical kernel stage.
    pub kernel: PhysicalKernel,
}

impl PhysicalLowering {
    /// Borrow the verified physical descriptor.
    #[must_use]
    pub const fn descriptor(&self) -> &KernelDescriptor {
        self.kernel.descriptor()
    }

    /// Consume the lowering result and return its verified descriptor.
    #[must_use]
    pub fn into_descriptor(self) -> KernelDescriptor {
        self.kernel.into_descriptor()
    }
}

/// Error raised while constructing verified physical kernel IR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalLoweringError {
    message: String,
}

impl PhysicalLoweringError {
    fn new(message: impl Into<String>) -> Self {
        let message = message.into();
        debug_assert!(message.contains("Fix:"));
        Self { message }
    }

    /// Return the actionable diagnostic.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for PhysicalLoweringError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for PhysicalLoweringError {}

fn prepare_physical_program(program: &Program) -> Result<Program, PhysicalLoweringError> {
    let expanded = vyre_foundation::transform::inline::inline_composite_calls(program).map_err(
        |error| {
            PhysicalLoweringError::new(format!(
                "composition expansion failed before semantic optimization: {error}. Fix: repair the registered composition body or its call graph."
            ))
        },
    )?;
    let expanded = lower_single_rank_collectives_for_emit(expanded)?;
    let optimized = vyre_foundation::optimizer::optimize(expanded).map_err(|error| {
        PhysicalLoweringError::new(format!(
            "registered semantic optimization failed before descriptor lowering: {error}. Fix: repair pass registration, legality, or convergence instead of emitting unoptimized IR."
        ))
    })?;
    vyre_foundation::transform::inline::inline_calls(&optimized).map_err(|error| {
        PhysicalLoweringError::new(format!(
            "unresolved call remained after semantic optimization: {error}. Fix: register its composition body or eliminate the dead call before backend emission."
        ))
    })
}

fn lower_single_rank_collectives_for_emit(
    program: Program,
) -> Result<Program, PhysicalLoweringError> {
    match vyre_foundation::transform::collectives::lower_single_rank_collectives(&program) {
        Ok(Some(lowered)) => Ok(lowered),
        Ok(None) => Ok(program),
        Err(error) => Err(PhysicalLoweringError::new(format!(
            "single-rank collective lowering failed before descriptor lowering: {error}. Fix: route true multi-rank collectives through a backend transport path or lower them before physical lowering."
        ))),
    }
}

/// Apply one validated selected-schedule phase and construct physical kernel IR.
///
/// Schedule lowering freezes the phase's exact workgroup shape on a cloned
/// semantic program before the canonical physical lowering boundary. The
/// selected schedule remains the authority; target emitters cannot substitute
/// a different shape through this API.
///
/// # Errors
///
/// Returns [`PhysicalLoweringError`] when the schedule is malformed, the phase
/// does not exist, or canonical physical lowering fails.
pub fn lower_scheduled(
    program: &Program,
    schedule: &SelectedSchedule,
    phase: SchedulePhaseId,
) -> Result<PhysicalLowering, PhysicalLoweringError> {
    schedule.validate().map_err(|error| {
        PhysicalLoweringError::new(format!(
            "selected schedule validation failed before physical lowering: {error}. Fix: repair bounded schedule search and persist only a validated neutral schedule."
        ))
    })?;
    let selected = schedule
        .phases
        .iter()
        .find(|selected| selected.id == phase)
        .ok_or_else(|| {
            PhysicalLoweringError::new(format!(
                "selected schedule phase {} is absent before physical lowering. Fix: preserve fusion-group to schedule-phase identity through artifact construction.",
                phase.0
            ))
        })?;
    let mut scheduled = program.clone();
    scheduled.set_workgroup_size(selected.workgroup);
    lower_physical(&scheduled)
}

/// Construct verified physical kernel IR from a semantic [`Program`].
///
/// The constructor expands compositions, optimizes once, lowers into the
/// backend-neutral physical descriptor, canonicalizes representation order,
/// and verifies the result. [`PhysicalKernel`] has no public raw-descriptor
/// constructor, so every target receives a value that crossed this validator.
///
/// # Errors
///
/// Returns [`PhysicalLoweringError`] when composition expansion, semantic
/// optimization, call resolution, descriptor lowering, canonicalization, or
/// verification fails.
pub fn lower_physical(program: &Program) -> Result<PhysicalLowering, PhysicalLoweringError> {
    let program = prepare_physical_program(program)?;
    let descriptor = lower(&program).map_err(|error| {
        PhysicalLoweringError::new(format!(
            "KernelDescriptor lowering failed after semantic Program optimization: {error}. Fix: add the missing neutral descriptor mapping before any concrete backend emits this Program."
        ))
    })?;
    let descriptor = verify_descriptor(&descriptor).map_err(|failure| {
        let (stage, fix) = match failure {
            VerifyFailure::Input(_) => (
                "after semantic Program optimization",
                "Fix: repair the neutral lowering mapping before descriptor canonicalization.",
            ),
            VerifyFailure::Output(_) => (
                "after bounded representation canonicalization",
                "Fix: repair vyre-lower canonicalization so every emitter receives valid neutral lower IR.",
            ),
        };
        PhysicalLoweringError::new(format!(
            "KernelDescriptor verification failed {stage}: {}. {fix}",
            format_verify_failure(&failure)
        ))
    })?;
    Ok(PhysicalLowering {
        program,
        kernel: PhysicalKernel { descriptor },
    })
}

fn format_verify_failure(error: &VerifyFailure) -> String {
    use std::fmt::Write as _;

    let stage = match error {
        VerifyFailure::Input(_) => "input",
        VerifyFailure::Output(_) => "output",
    };
    let mut out = String::with_capacity(64);
    out.push_str(stage);
    out.push_str(" descriptor invalid");
    for (index, err) in error.errors().iter().take(4).enumerate() {
        out.push_str(if index == 0 { ": " } else { "; " });
        let _ = write!(out, "{err:?}");
    }
    if error.errors().len() > 4 {
        out.push_str("; ...");
    }
    out
}

// Inline: covers the crate-private `message` and `new`, which no integration test can reach.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{KernelBody, KernelOpKind};
    use vyre_foundation::ir::{
        BufferAccess, BufferDecl, CollectiveOp, CommGroup, DataType, Expr, Ident, Node,
    };
    use vyre_foundation::schedule::{SchedulePhaseId, ScheduleTransform, SelectedSchedule};

    #[test]
    fn lower_physical_runs_program_and_descriptor_pipeline() {
        let buffer =
            BufferDecl::storage("out", 0, BufferAccess::ReadWrite, DataType::U32).with_count(16);
        let program = Program::wrapped(
            vec![buffer],
            [64, 1, 1],
            vec![Node::Store {
                buffer: Ident::from("out"),
                index: Expr::InvocationId { axis: 0 },
                value: Expr::LitU32(7),
            }],
        );

        let lowered = lower_physical(&program).expect("Fix: pre-emit lowering must pass");

        assert_eq!(lowered.program.workgroup_size(), [64, 1, 1]);
        assert_eq!(lowered.descriptor().dispatch.workgroup_size, [64, 1, 1]);
        assert_eq!(lowered.descriptor().bindings.slots.len(), 1);
        assert!(crate::verify::verify(lowered.descriptor()).is_ok());
        assert_eq!(
            crate::canonicalize::canonicalize_for_emit(lowered.descriptor()),
            *lowered.descriptor()
        );
    }

    #[test]
    fn lower_scheduled_freezes_the_selected_phase_shape_before_physical_lowering() {
        let program = Program::wrapped(Vec::new(), [64, 1, 1], Vec::new());
        let mut schedule = SelectedSchedule::synthetic(1);
        schedule
            .apply(ScheduleTransform::SetWorkgroup {
                phase: SchedulePhaseId(0),
                shape: [32, 2, 1],
            })
            .unwrap();

        let lowered = lower_scheduled(&program, &schedule, SchedulePhaseId(0)).unwrap();
        assert_eq!(lowered.program.workgroup_size(), [32, 2, 1]);
        assert_eq!(lowered.descriptor().dispatch.workgroup_size, [32, 2, 1]);

        let error = lower_scheduled(&program, &schedule, SchedulePhaseId(9)).unwrap_err();
        assert!(error.message().contains("phase 9 is absent"));
        assert!(error.message().contains("Fix:"));
    }

    #[test]
    fn lower_physical_rejects_invalid_descriptor_before_backend_emit() {
        let program = Program::wrapped(Vec::new(), [0, 1, 1], Vec::new());

        let error = lower_physical(&program).expect_err("zero dispatch must fail");

        assert!(error.message().contains("KernelDescriptor"));
        assert!(error.message().contains("Fix:"));
    }

    #[test]
    fn lower_physical_lowers_world_allgather_before_descriptor_lowering() {
        let program = Program::wrapped(
            vec![
                BufferDecl::read("gather_in", 0, DataType::U32).with_count(8),
                BufferDecl::output("gather_out", 1, DataType::U32).with_count(8),
            ],
            [64, 1, 1],
            vec![Node::AllGather {
                input: "gather_in".into(),
                output: "gather_out".into(),
                group: CommGroup::WORLD,
            }],
        );

        let lowered = lower_physical(&program).expect(
            "Fix: canonical pre-emit must lower WORLD AllGather before descriptor lowering.",
        );

        assert!(!lowered.program.stats().distributed_collectives());
        assert!(crate::verify::verify(lowered.descriptor()).is_ok());
    }

    #[test]
    fn lower_physical_rejects_transport_collectives_before_descriptor_lowering() {
        let program = Program::wrapped(
            vec![
                BufferDecl::read("scatter_in", 0, DataType::U32).with_count(8),
                BufferDecl::output("scatter_out", 1, DataType::U32).with_count(8),
            ],
            [64, 1, 1],
            vec![Node::ReduceScatter {
                input: "scatter_in".into(),
                output: "scatter_out".into(),
                op: CollectiveOp::Sum,
                group: CommGroup(7),
            }],
        );

        let error = lower_physical(&program)
            .expect_err("Fix: canonical pre-emit must reject collectives that need transport.");

        assert!(error.message().contains("Multi-rank collective transport"));
    }

    #[test]
    fn lower_physical_preserves_loop_carrier_swap_snapshot() {
        let program = Program::wrapped(
            vec![BufferDecl::output("results", 0, DataType::U32).with_count(1)],
            [64, 1, 1],
            vec![
                Node::let_bind("s0", Expr::u32(1)),
                Node::let_bind("s1", Expr::u32(2)),
                Node::Loop {
                    var: "pc".into(),
                    from: Expr::u32(0),
                    to: Expr::u32(4),
                    body: vec![
                        Node::let_bind("tmp", Expr::var("s0")),
                        Node::assign("s0", Expr::var("s1")),
                        Node::assign("s1", Expr::var("tmp")),
                    ],
                },
                Node::store("results", Expr::u32(0), Expr::var("s0")),
            ],
        );

        let lowered = lower_physical(&program).expect("Fix: pre-emit lowering must pass");

        assert!(
            body_has_s1_end_from_copy(&lowered.descriptor().body),
            "Fix: lowering must preserve `let tmp = s0` as a Copy snapshot so SWAP writes s1 from old s0 instead of the post-assign s0 carrier"
        );
    }

    fn body_has_s1_end_from_copy(body: &KernelBody) -> bool {
        body.ops.iter().any(|op| {
            let KernelOpKind::LoopCarrierEnd { name } = &op.kind else {
                return false;
            };
            name.as_ref() == "s1"
                && op.operands.first().copied().is_some_and(|operand| {
                    body.ops.iter().any(|producer| {
                        producer.result == Some(operand)
                            && matches!(producer.kind, KernelOpKind::Copy)
                    })
                })
        }) || body.child_bodies.iter().any(body_has_s1_end_from_copy)
    }
}
