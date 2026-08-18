//! Canonical verified lowering boundary.
//!
//! This is the only production boundary from high-level `Program` IR to an
//! emitter-ready `KernelDescriptor`: expand registered compositions, run the
//! registered fallible semantic optimizer once, reject unresolved calls, lower,
//! canonicalize representation order, and verify.

use crate::descriptor::KernelDescriptor;
use crate::lower::lower;
use crate::{verify_descriptor, VerifyFailure};
use std::fmt;
use vyre_foundation::ir::Program;

/// Program and descriptor produced by the canonical verified lower boundary.
#[derive(Debug, Clone)]
pub struct VerifiedLowering {
    /// Program after composition expansion and registered semantic optimization.
    pub program: Program,
    /// Verified descriptor after bounded representation canonicalization.
    pub descriptor: KernelDescriptor,
}

/// Error raised by canonical verified lowering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LowerVerifiedError {
    message: String,
}

impl LowerVerifiedError {
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

impl fmt::Display for LowerVerifiedError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for LowerVerifiedError {}

fn prepare_verified_program(program: &Program) -> Result<Program, LowerVerifiedError> {
    let expanded = vyre_foundation::transform::inline::inline_composite_calls(program).map_err(|error| {
        LowerVerifiedError::new(format!(
            "composition expansion failed before semantic optimization: {error}. Fix: repair the registered composition body or its call graph."
        ))
    })?;
    let expanded = lower_single_rank_collectives_for_emit(expanded)?;
    let optimized = vyre_foundation::optimizer::optimize(expanded).map_err(|error| {
        LowerVerifiedError::new(format!(
            "registered semantic optimization failed before descriptor lowering: {error}. Fix: repair pass registration, legality, or convergence instead of emitting unoptimized IR."
        ))
    })?;
    vyre_foundation::transform::inline::inline_calls(&optimized).map_err(|error| {
        LowerVerifiedError::new(format!(
            "unresolved call remained after semantic optimization: {error}. Fix: register its composition body or eliminate the dead call before backend emission."
        ))
    })
}

fn lower_single_rank_collectives_for_emit(program: Program) -> Result<Program, LowerVerifiedError> {
    match vyre_foundation::transform::collectives::lower_single_rank_collectives(&program) {
        Ok(Some(lowered)) => Ok(lowered),
        Ok(None) => Ok(program),
        Err(error) => Err(LowerVerifiedError::new(format!(
            "single-rank collective lowering failed before descriptor lowering: {error}. Fix: route true multi-rank collectives through a backend transport path or lower them before verified lowering."
        ))),
    }
}

/// Expand compositions, optimize once, and produce verified neutral lower IR.
///
/// # Errors
///
/// Returns [`LowerVerifiedError`] when composition expansion, semantic
/// optimization, call resolution, descriptor lowering, canonicalization, or
/// verification fails.
pub fn lower_verified(program: &Program) -> Result<VerifiedLowering, LowerVerifiedError> {
    let program = prepare_verified_program(program)?;
    let descriptor = lower(&program).map_err(|error| {
        LowerVerifiedError::new(format!(
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
        LowerVerifiedError::new(format!(
            "KernelDescriptor verification failed {stage}: {}. {fix}",
            format_verify_failure(&failure)
        ))
    })?;
    Ok(VerifiedLowering {
        program,
        descriptor,
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

    #[test]
    fn lower_verified_runs_program_and_descriptor_pipeline() {
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

        let lowered = lower_verified(&program).expect("Fix: pre-emit lowering must pass");

        assert_eq!(lowered.program.workgroup_size(), [64, 1, 1]);
        assert_eq!(lowered.descriptor.dispatch.workgroup_size, [64, 1, 1]);
        assert_eq!(lowered.descriptor.bindings.slots.len(), 1);
        assert!(crate::verify::verify(&lowered.descriptor).is_ok());
        assert_eq!(
            crate::canonicalize::canonicalize_for_emit(&lowered.descriptor),
            lowered.descriptor
        );
    }

    #[test]
    fn lower_verified_rejects_invalid_descriptor_before_backend_emit() {
        let program = Program::wrapped(Vec::new(), [0, 1, 1], Vec::new());

        let error = lower_verified(&program).expect_err("zero dispatch must fail");

        assert!(error.message().contains("KernelDescriptor"));
        assert!(error.message().contains("Fix:"));
    }

    #[test]
    fn lower_verified_lowers_world_allgather_before_descriptor_lowering() {
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

        let lowered = lower_verified(&program).expect(
            "Fix: canonical pre-emit must lower WORLD AllGather before descriptor lowering.",
        );

        assert!(!lowered.program.stats().distributed_collectives());
        assert!(crate::verify::verify(&lowered.descriptor).is_ok());
    }

    #[test]
    fn lower_verified_rejects_transport_collectives_before_descriptor_lowering() {
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

        let error = lower_verified(&program)
            .expect_err("Fix: canonical pre-emit must reject collectives that need transport.");

        assert!(error.message().contains("Multi-rank collective transport"));
    }

    #[test]
    fn lower_verified_preserves_loop_carrier_swap_snapshot() {
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

        let lowered = lower_verified(&program).expect("Fix: pre-emit lowering must pass");

        assert!(
        body_has_s1_end_from_copy(&lowered.descriptor.body),
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
