//! Descriptor-level validation and analysis before concrete CUDA PTX emission.

use vyre_foundation::ir::Program;
use vyre_lower::pattern_audit::PatternAudit;

pub(crate) fn validate_and_analyze(
    program: &Program,
    target_sm: u32,
) -> Result<vyre_lower::KernelDescriptor, String> {
    let descriptor = lower_for_cuda_emit(program)?;
    if crate::instrumentation::cuda_descriptor_audit_enabled() {
        let mut facts = vyre_lower::analyses::AnalysisFacts::none();
        if let Some(banks) = std::num::NonZeroU32::new(crate::device::SHARED_MEMORY_BANK_COUNT) {
            facts = facts.with_shared_memory_banks(banks);
        }
        let neutral = vyre_lower::audit(&descriptor, &facts);
        let concrete = vyre_emit_ptx::patterns::audit(&descriptor, compute_capability(target_sm));
        tracing::trace!(
            target: "vyre_driver_cuda::descriptor",
            kernel = %descriptor.id,
            neutral = %neutral.format_short(),
            concrete = %concrete.format_short(),
            "descriptor analysis completed before CUDA PTX emission",
        );
    }
    Ok(descriptor)
}

fn lower_for_cuda_emit(program: &Program) -> Result<vyre_lower::KernelDescriptor, String> {
    let trace = crate::instrumentation::cuda_stage_trace_enabled();
    let start = std::time::Instant::now();
    let descriptor = vyre_lower::lower_physical(program)
        .map_err(|error| {
            format!(
                "physical lowering failed before CUDA PTX emission: {error}. Fix: repair the source Program or add the missing neutral mapping before PTX emission."
            )
        })?
        .into_descriptor();
    if trace {
        tracing::debug!(
            "[cuda-codegen] +{}ms lower ops={} bindings={}",
            start.elapsed().as_millis(),
            descriptor.body.ops.len(),
            descriptor.bindings.slots.len()
        );
    }
    Ok(descriptor)
}

pub(crate) fn compute_capability(target_sm: u32) -> vyre_emit_ptx::ComputeCapability {
    vyre_emit_ptx::ComputeCapability {
        major: target_sm / 10,
        minor: target_sm % 10,
    }
}

// Inline: covers `compute_capability`, `validate_and_analyze`, which no integration test can name.
#[cfg(test)]
mod tests {
    use super::*;
    use vyre_foundation::ir::{BufferDecl, DataType, Expr, Ident, Node, Program};
    use vyre_lower::emit_adversarial_corpus;

    #[test]
    fn validates_simple_store_program() {
        let buffer = BufferDecl::output("out", 0, DataType::U32).with_count(16);
        let program = Program::wrapped(
            vec![buffer],
            [128, 1, 1],
            vec![Node::Store {
                buffer: Ident::from("out"),
                index: Expr::InvocationId { axis: 0 },
                value: Expr::LitU32(9),
            }],
        );

        let descriptor =
            validate_and_analyze(&program, 90).expect("Fix: descriptor gate must pass");

        assert_eq!(descriptor.dispatch.workgroup_size, [128, 1, 1]);
        assert_eq!(descriptor.bindings.slots.len(), 1);
        assert!(vyre_lower::verify(&descriptor).is_ok());
    }

    #[test]
    fn rejects_descriptor_verification_failures() {
        let program = Program::wrapped(Vec::new(), [1, 0, 1], Vec::new());

        let error = validate_and_analyze(&program, 90).expect_err("zero dispatch must fail");

        assert!(error.contains("physical lowering failed"));
        assert!(error.contains("KernelDescriptor"));
        assert!(error.contains("Fix:"));
    }

    #[test]
    fn adversarial_success_corpus_passes_verification_and_ptx_emit() {
        for case in emit_adversarial_corpus::success_cases() {
            let descriptor =
                vyre_lower::verify_descriptor(&case.descriptor).unwrap_or_else(|error| {
                    panic!(
                        "Fix: `{}` ({:?}) must pass shared descriptor verification: {error:?}",
                        case.id, case.family
                    )
                });
            let ptx = vyre_emit_ptx::emit_with_target(&descriptor, compute_capability(90))
                .unwrap_or_else(|error| {
                    panic!(
                        "Fix: `{}` ({:?}) must emit CUDA PTX after shared verification: {error:?}",
                        case.id, case.family
                    )
                });
            assert!(
                ptx.contains(".entry main") && ptx.contains("ret;"),
                "Fix: `{}` CUDA PTX artifact must contain a main entry and return.\n{ptx}",
                case.id
            );
        }
    }

    #[test]
    fn adversarial_rejection_corpus_returns_structured_cuda_errors() {
        for case in emit_adversarial_corpus::rejection_cases() {
            let result = vyre_lower::verify_descriptor(&case.descriptor)
                .map_err(|error| {
                    format!(
                        "CUDA descriptor verification failed for `{}`: {error:?}. Fix: repair the shared lower-IR descriptor.",
                        case.id
                    )
                })
                .and_then(|descriptor| {
                    vyre_emit_ptx::emit_with_target(&descriptor, compute_capability(90))
                        .map(|_| ())
                        .map_err(|error| {
                            format!(
                                "CUDA descriptor PTX emission failed for `{}`: {error}. Fix: add the missing PTX lowering in vyre-emit-ptx.",
                                case.id
                            )
                        })
                });
            let error = result.expect_err(
                "Fix: rejection corpus case must fail descriptor verification or PTX emission",
            );
            assert!(
                error.contains(case.id) && error.contains("Fix:"),
                "Fix: `{}` CUDA rejection must include case id and repair text: {error}",
                case.id
            );
        }
    }

    #[test]
    fn maps_sm_number_to_compute_capability() {
        let cc = compute_capability(89);
        assert_eq!(cc.major, 8);
        assert_eq!(cc.minor, 9);
    }
}
