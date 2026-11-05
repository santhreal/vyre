//! The `fma_f32_violations` filter emit backends run before lowering.
//!
//! WHY: this is the focused subset an emit backend asks for, and it is the
//! only thing standing between an integer `Expr::Fma` and a silently
//! contracted `a * b + c` on a target that accepts one. The filter is pinned
//! to `V028` so a message change cannot disable the rejection, and the
//! accepted counterpart proves the filter does not swallow unrelated
//! diagnostics an emit boundary would otherwise report.
//!
//! Both programs come from `vyre_test_support::strict_float_programs`, the
//! same shapes the emit-side rejection contract in `vyre-driver-wgpu` runs.

use vyre_foundation::validate::fma_f32_violations;
use vyre_test_support::strict_float_programs::{
    constant_f32_fma_program, integer_operand_fma_program,
};

#[test]
fn fma_f32_violations_flags_integer_fma_with_actionable_message() {
    let program = integer_operand_fma_program();
    let violations = fma_f32_violations(&program);
    assert_eq!(
        violations.len(),
        3,
        "every non-f32 Fma operand (a, b, c) must be reported, got: {violations:?}"
    );
    for violation in &violations {
        assert!(
            violation.code().as_str() == "V028",
            "fma_f32_violations must only return V028 errors, got: {}",
            violation.message()
        );
        assert!(
            violation
                .message()
                .contains("Fma requires three f32 operands")
                && violation.message().contains("must be `f32`")
                && violation.message().contains("Fix:"),
            "V028 message must name the f32 contract and a fix, got: {}",
            violation.message()
        );
    }
}

#[test]
fn fma_f32_violations_empty_for_all_f32_operands() {
    let program = constant_f32_fma_program();
    assert!(
        fma_f32_violations(&program).is_empty(),
        "f32 Fma is valid and must not be flagged"
    );
}
