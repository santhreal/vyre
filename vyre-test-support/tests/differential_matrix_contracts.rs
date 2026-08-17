//! Integration tests for differential execution matrix and replay capsules.

use vyre_foundation::ir::{BufferAccess, BufferDecl, DataType, Expr, Node, Program};
use vyre_test_support::differential_matrix::{evaluate_differential, DifferentialDecision};
use vyre_test_support::replay_capsule::ReplayCapsule;

#[test]
fn differential_matrix_evaluates_exact_and_approximate() {
    let program = Program::wrapped(
        vec![
            BufferDecl::storage("in", 0, BufferAccess::ReadOnly, DataType::U32).with_count(1),
            BufferDecl::output("out", 1, DataType::U32).with_count(1),
        ],
        [1, 1, 1],
        vec![Node::store(
            "out",
            Expr::u32(0),
            Expr::add(Expr::load("in", Expr::u32(0)), Expr::u32(10)),
        )],
    );

    let input_bytes = vec![5u32.to_le_bytes().to_vec()];
    let backend_output = vec![15u32.to_le_bytes().to_vec()];

    let decision = evaluate_differential(
        &program,
        "test_add",
        &[vyre_reference::value::Value::Bytes(
            input_bytes[0].clone().into(),
        )],
        &backend_output,
    )
    .expect("evaluation must succeed");

    assert_eq!(decision, DifferentialDecision::ExactByteMatch);
}

#[test]
fn replay_capsule_persists_and_minimizes() {
    let capsule = ReplayCapsule::new(
        "tests/differential.rs",
        "reference",
        "host",
        "1.0",
        "ir-fixtures",
        42,
        vec![0x56, 0x49, 0x52, 0x30],
        vec![vec![1; 32]],
        4,
        "test mismatch",
    );

    let json = capsule.to_json().expect("json serialize");
    let recovered = ReplayCapsule::from_json(&json).expect("json deserialize");
    assert_eq!(capsule, recovered);

    let minimized = capsule.minimize_inputs();
    assert_eq!(minimized.input_bytes[0].len(), 16);
}
