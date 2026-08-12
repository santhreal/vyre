//! Structured validation-source preservation at the reference boundary.

use vyre_foundation::ir::{BufferDecl, DataType, Program};
use vyre_reference::reference_eval;

#[test]
fn reference_error_preserves_structured_validation_source() {
    let program = Program::wrapped(
        vec![BufferDecl::output("out", 0, DataType::U32)],
        [0, 1, 1],
        Vec::new(),
    );

    let error =
        reference_eval(&program, &[]).expect_err("zero workgroup axis must fail validation");
    let source = error
        .validation_source()
        .expect("reference error must retain the foundation validation issue");
    assert_eq!(source.code().as_str(), "V106");
    assert_eq!(
        source.phase(),
        vyre_foundation::validate::ValidationPhase::Program
    );
    assert!(matches!(
        source.location(),
        vyre_foundation::validate::ValidationLocation::WorkgroupAxis(0)
    ));
}
