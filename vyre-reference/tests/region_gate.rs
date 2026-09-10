//! Reference interpreter region-gate regression tests.

use vyre_foundation::ir::{BufferDecl, DataType, Expr, Node, Program};

fn raw_program() -> Program {
    Program::from_raw_parts(
        vec![BufferDecl::output("out", 0, DataType::U32).with_count(1)],
        [1, 1, 1],
        vec![Node::store("out", Expr::u32(0), Expr::u32(7)), Node::Return],
    )
}

#[test]
fn reference_eval_rejects_non_region_programs() {
    let error = vyre_reference::ReferenceRequest::standard(
        &raw_program(),
        &[vyre_reference::value::Value::from(vec![0u8; 4])],
    )
    .outputs()
    .expect_err("Fix: reference_eval must reject raw top-level statements");
    assert!(
        error
            .to_string()
            .contains("top-level Region-wrapped Program"),
        "Fix: reference_eval rejection must mention the region invariant, got: {error}"
    );
}
