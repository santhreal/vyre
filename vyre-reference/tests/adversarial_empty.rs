//! Adversarial empty and malformed-boundary coverage for the reference interpreter.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_reference::{reference_eval, value::Value, ReferenceBudget, ReferenceRequest};

#[test]
fn empty_wrapped_program_returns_no_outputs() {
    let program = Program::wrapped(Vec::new(), [1, 1, 1], Vec::new());
    let req = ReferenceRequest::new(&program, &[], ReferenceBudget::standard());
    let _outputs = reference_eval(&req).expect("Fix: empty wrapped Program must evaluate");
}

#[test]
fn raw_empty_program_is_rejected_with_region_context() {
    let program = Program::from_raw_parts(Vec::new(), [1, 1, 1], Vec::new());
    let req = ReferenceRequest::new(&program, &[], ReferenceBudget::standard());
    let err = reference_eval(&req).expect_err("Fix: raw empty Program must be rejected");
    let message = err.to_string();
    assert!(
        message.contains("top-level Region"),
        "expected top-level Region diagnostic, got: {message}"
    );
}

#[test]
fn zero_length_input_does_not_create_implicit_bytes() {
    let decls = vec![
        BufferDecl::storage("input", 0, BufferAccess::ReadOnly, DataType::U32).with_count(1),
        BufferDecl::storage("out", 1, BufferAccess::ReadWrite, DataType::U32).with_count(1),
    ];
    let store_node = Node::store("out", Expr::u32(0), Expr::load("input", Expr::u32(0)));
    let program = Program::wrapped(decls, [1, 1, 1], vec![store_node]);
    let inputs = [Value::Bytes(Vec::new().into())];
    let req = ReferenceRequest::new(&program, &inputs, ReferenceBudget::standard());
    let err = reference_eval(&req)
        .expect_err("Fix: zero-byte input for u32 load must be rejected");
    let message = err.to_string();
    assert!(
        message.contains("input") || message.contains("buffer"),
        "expected actionable buffer diagnostic, got: {message}"
    );
}
