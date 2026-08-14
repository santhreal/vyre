//! Descriptor-level validation and analysis before concrete wgpu emission.

use vyre_foundation::ir::Program;
use vyre_foundation::lower::LoweringError;
use vyre_lower::pattern_audit::PatternAudit;

pub(crate) fn validate_and_analyze(
    program: &Program,
) -> Result<vyre_lower::KernelDescriptor, LoweringError> {
    let lowered = vyre_lower::lower_verified(program).map_err(|error| {
        LoweringError::invalid(format!(
            "verified lowering failed before wgpu emission: {error}. Fix: route Programs through vyre_lower::lower_verified and add missing neutral mappings there instead of concrete-driver lowering."
        ))
    })?;
    let descriptor = lowered.descriptor;
    let neutral = vyre_lower::audit::audit(&descriptor);
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
        assert!(vyre_lower::verify::verify(&descriptor).is_ok());
    }

    #[test]
    fn rejects_descriptor_verification_failures() {
        let program = Program::wrapped(Vec::new(), [0, 1, 1], Vec::new());

        let error = validate_and_analyze(&program).expect_err("zero dispatch must fail");

        assert!(error.message().contains("verified lowering failed"));
        assert!(error.message().contains("KernelDescriptor"));
        assert!(error.message().contains("Fix:"));
    }
}
