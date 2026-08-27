//! Descriptor-level validation and analysis before concrete wgpu emission.

use vyre_foundation::ir::Program;
use vyre_foundation::lower::LoweringError;
use vyre_lower::pattern_audit::PatternAudit;

pub(crate) fn validate_and_analyze(
    program: &Program,
) -> Result<vyre_lower::KernelDescriptor, LoweringError> {
    let lowered = vyre_lower::lower_physical(program).map_err(|error| {
        LoweringError::invalid(format!(
            "physical lowering failed before wgpu emission: {error}. Fix: add the missing neutral mapping to vyre-lower instead of concrete-driver lowering."
        ))
    })?;
    let descriptor = lowered.into_descriptor();
    // The portable adapter reports no shared-memory bank geometry here, so the
    // neutral audit runs without a bank-conflict section rather than against an
    // assumed layout.
    let neutral = vyre_lower::audit(&descriptor, &vyre_lower::analyses::AnalysisFacts::none());
    let concrete = vyre_emit_naga::patterns::audit(&descriptor);
    tracing::trace!(
        target: "vyre_driver_wgpu::descriptor",
        kernel = %descriptor.id,
        neutral = %neutral.format_short(),
        concrete = %concrete.format_short(),
        "descriptor analysis completed before wgpu emission",
    );
    Ok(descriptor)
}

// Inline: covers `validate_and_analyze`, which no integration test can name.
#[cfg(test)]
mod tests {
    use super::*;
    use vyre_foundation::ir::{BufferDecl, DataType, Expr, Ident, Node, Program};

    #[test]
    fn validates_simple_store_program() {
        let buffer = BufferDecl::output("out", 0, DataType::U32).with_count(16);
        let program = Program::wrapped(
            vec![buffer],
            [64, 1, 1],
            vec![Node::Store {
                buffer: Ident::from("out"),
                index: Expr::InvocationId { axis: 0 },
                value: Expr::LitU32(7),
            }],
        );

        let descriptor = validate_and_analyze(&program).expect("Fix: descriptor gate must pass");

        assert_eq!(descriptor.dispatch.workgroup_size, [64, 1, 1]);
        assert_eq!(descriptor.bindings.slots.len(), 1);
        assert!(vyre_lower::verify(&descriptor).is_ok());
    }

    #[test]
    fn rejects_descriptor_verification_failures() {
        let program = Program::wrapped(Vec::new(), [0, 1, 1], Vec::new());

        let error = validate_and_analyze(&program).expect_err("zero dispatch must fail");

        assert!(error.message().contains("physical lowering failed"));
        assert!(error.message().contains("KernelDescriptor"));
        assert!(error.message().contains("Fix:"));
    }
}
