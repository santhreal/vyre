//! Workspace error diagnostic schema coverage.
//!
//! Asserts that workspace boundary error types project into structured
//! diagnostics adhering to the diagnostic protocol schema (stable code,
//! severity, compiler level, structured cause chain, retry class, corrective action).

use vyre_aot::CompileError;
use vyre_driver::BackendError;
use vyre_foundation::diagnostics::{
    CompilerLevel, DiagnosticStage, RetryClass, Severity, ToDiagnostic,
};
use vyre_foundation::validate::ValidationError;
use vyre_foundation::IrError;
use vyre_lower::LowerError;
use vyre_megakernel::TargetCompileError;

#[test]
fn validation_error_maps_to_diagnostic_schema() {
    let err = ValidationError::unsupported_op("cuda", &"custom.op".into(), 0);
    let diag = err.to_diagnostic();
    assert_eq!(diag.severity, Severity::Error);
    assert_eq!(diag.code.as_str(), "V056");
    assert_eq!(diag.stage, DiagnosticStage::Validate);
    assert_eq!(diag.compiler_level, Some(CompilerLevel::FoundationIr));
    assert_eq!(diag.retry, RetryClass::Never);
    assert!(!diag.cause_chain.is_empty());
    assert!(diag.suggested_fix.is_some());
}

#[test]
fn ir_error_maps_to_diagnostic_schema() {
    let err = IrError::InlineCycle {
        op_id: "test.cycle".to_string(),
    };
    let diag = err.to_diagnostic();
    assert_eq!(diag.severity, Severity::Error);
    assert_eq!(diag.code.as_str(), "IRC001_INLINE_CYCLE");
    assert_eq!(diag.compiler_level, Some(CompilerLevel::Optimizer));
    assert_eq!(diag.retry, RetryClass::Never);
    assert!(diag.suggested_fix.is_some());
}

#[test]
fn backend_error_maps_to_diagnostic_schema() {
    let err = BackendError::new("dispatch queue failed. Fix: ensure CUDA driver is loaded");
    let diag = err.to_diagnostic();
    assert_eq!(diag.severity, Severity::Error);
    assert_eq!(diag.code.as_str(), "BACKEND_OTHER");
    assert_eq!(diag.stage, DiagnosticStage::Submit);
    assert_eq!(diag.compiler_level, Some(CompilerLevel::DriverRuntime));
    assert_eq!(diag.retry, RetryClass::Never);
    assert!(!diag.cause_chain.is_empty());
    assert_eq!(
        diag.suggested_fix.as_deref(),
        Some("ensure CUDA driver is loaded")
    );
}

#[test]
fn lower_error_maps_to_diagnostic_schema() {
    let err = LowerError::UnsupportedConstruct("unsupported dynamic shape in PTX lowering".into());
    let diag = err.to_diagnostic();
    assert_eq!(diag.severity, Severity::Error);
    assert_eq!(diag.code.as_str(), "LWR001_UNSUPPORTED_CONSTRUCT");
    assert_eq!(diag.stage, DiagnosticStage::Lower);
    assert_eq!(diag.compiler_level, Some(CompilerLevel::Lowering));
    assert_eq!(diag.retry, RetryClass::RecompileSource);
    assert!(!diag.cause_chain.is_empty());
    assert!(diag.suggested_fix.is_some());
}

#[test]
fn target_compile_error_maps_to_diagnostic_schema() {
    let err = TargetCompileError::Emission("MSL pipeline compilation failed".into());
    let diag = err.to_diagnostic();
    assert_eq!(diag.severity, Severity::Error);
    assert_eq!(diag.code.as_str(), "MKC_TARGET_EMISSION_FAILED");
    assert_eq!(diag.stage, DiagnosticStage::Emit);
    assert_eq!(diag.compiler_level, Some(CompilerLevel::Emission));
    assert_eq!(diag.retry, RetryClass::RecompileSource);
    assert!(!diag.cause_chain.is_empty());
}

#[test]
fn compile_error_retains_nested_structured_cause_chain() {
    let target_err = TargetCompileError::Emission("MSL pipeline compilation failed".into());
    let compile_err = CompileError::TargetCompilation(target_err);
    let diag = compile_err.to_diagnostic();
    assert_eq!(diag.severity, Severity::Error);
    assert_eq!(diag.stage, DiagnosticStage::Emit);
    assert_eq!(diag.compiler_level, Some(CompilerLevel::Emission));
    assert_eq!(diag.retry, RetryClass::RecompileSource);
    // Preserved the nested cause chain from TargetCompileError
    assert!(!diag.cause_chain.is_empty());
    assert_eq!(diag.cause_chain[0].kind, "emission_failure");
}
